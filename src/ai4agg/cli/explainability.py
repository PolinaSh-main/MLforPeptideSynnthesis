import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tqdm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..utils.loaders import make_agg_point_peptide_set
from ..utils.preprocessors import (
    FingerprintPreprocessor,
    OccurencyVectorPreprocessor,
    PositionalFingerprintPreprocessor,
    ProtectedFingerprintPreprocessor,
    ProtectedWholePeptideFingerprintPreprocessor,
    WholePeptideFingerprintPreprocessor,
)
from ..utils.utils import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Preprocessor registry (SHAP-compatible subset — classification only) ────
PREPROCESSOR_REGISTRY = {
    "occurency": OccurencyVectorPreprocessor,
    "fingerprint": FingerprintPreprocessor,
    "protected_fingerprint": ProtectedFingerprintPreprocessor,
    "positional_fingerprint": PositionalFingerprintPreprocessor,
    "whole_peptide_fingerprint": WholePeptideFingerprintPreprocessor,
    "protected_whole_peptide_fingerprint": ProtectedWholePeptideFingerprintPreprocessor,
}

# ── Model registry (tree models → TreeExplainer; others → KernelExplainer) ──
MODEL_REGISTRY = {
    "xgb": XGBClassifier,
    "rff": RandomForestClassifier,
}

TREE_MODELS = {"xgb", "rff"}

AA_DISPLAY_NAMES = {
    "A": "Ala (A)", "C": "Cys (C)", "D": "Asp (D)", "E": "Glu (E)",
    "F": "Phe (F)", "G": "Gly (G)", "H": "His (H)", "I": "Ile (I)",
    "K": "Lys (K)", "L": "Leu (L)", "M": "Met (M)", "N": "Asn (N)",
    "P": "Pro (P)", "Q": "Gln (Q)", "R": "Arg (R)", "S": "Ser (S)",
    "T": "Thr (T)", "V": "Val (V)", "W": "Trp (W)", "Y": "Tyr (Y)",
}


# ── Motif helpers (unchanged from original) ─────────────────────────────────

def get_motifs(peptide: str, motifs: str) -> List[str]:
    if motifs == "1_2":
        motifs_list = [peptide[i:i + 2] for i in range(len(peptide) - 1)]
    elif motifs == "1_3":
        motifs_list = [peptide[i] + peptide[i + 2] for i in range(len(peptide) - 2)]
    else:
        raise ValueError(f"Unknown motif type: {motifs}")
    return ["".join(sorted(m)) for m in motifs_list]


def get_motif_occurency_vector_1_2(peptide: str, all_motifs: List[str]) -> np.ndarray:
    v = np.zeros(len(all_motifs))
    for i, motif in enumerate(all_motifs):
        v[i] = peptide.count(motif) + peptide.count(f"{motif[1]}{motif[0]}")
    return v


def get_motif_occurency_vector_1_3(peptide: str, all_motifs: List[str]) -> np.ndarray:
    v = np.zeros(len(all_motifs))
    for i, motif in enumerate(all_motifs):
        v[i] = (
            len(re.findall(f"{motif[0]}.{motif[1]}", peptide))
            + len(re.findall(f"{motif[1]}.{motif[0]}", peptide))
        )
    return v


MOTIF_REGISTRY = {
    "1_2": get_motif_occurency_vector_1_2,
    "1_3": get_motif_occurency_vector_1_3,
}


# ── Feature name helpers ─────────────────────────────────────────────────────

def _build_feature_names(
    preprocessor_key: str,
    preprocessor_instance,
    n_fingerprint_bits: int,
    max_sequence_len: int,
) -> List[str]:
    """Return human-readable feature names for the chosen preprocessor."""

    if preprocessor_key == "occurency":
        # Per-AA names, pretty-printed
        return [
            AA_DISPLAY_NAMES.get(aa, aa) for aa in preprocessor_instance.all_aa
        ]

    if preprocessor_key in (
        "fingerprint",
        "protected_fingerprint",
        "positional_fingerprint",
    ):
        # Concatenated per-residue fingerprints: pos{i}_bit{j}
        names = []
        for pos in range(max_sequence_len):
            for bit in range(n_fingerprint_bits):
                names.append(f"pos{pos}_bit{bit}")
        return names

    if preprocessor_key in (
        "whole_peptide_fingerprint",
        "protected_whole_peptide_fingerprint",
    ):
        return [f"bit{i}" for i in range(n_fingerprint_bits)]

    # Fallback: generic numeric names
    sample_fp = preprocessor_instance("A")
    return [f"feat{i}" for i in range(len(sample_fp))]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(
    data_path: Path,
    preprocessor_key: str,
    motifs: str,
    normalise: bool,
    n_fingerprint_bits: int,
    seed: int,
) -> Tuple[List[str], pd.DataFrame]:

    data = make_agg_point_peptide_set(data_path)

    PreprocessorClass = PREPROCESSOR_REGISTRY[preprocessor_key]
    preprocessor = PreprocessorClass(
        data,
        random_state=seed,
        normalise=normalise,
        occurency_vector_normalise=normalise,
        n_fingerprint_bits=n_fingerprint_bits,
    )
    data["input"] = data["peptide"].map(preprocessor)
    data["label"] = data["aggregation"].map(lambda agg: [0, 1] if agg else [1, 0])

    feature_names = _build_feature_names(
        preprocessor_key,
        preprocessor,
        n_fingerprint_bits,
        preprocessor.max_sequence_len,
    )

    # Optional motif augmentation — only sensible with occurency preprocessor
    if motifs in ("1_2", "1_3"):
        if preprocessor_key != "occurency":
            logger.warning(
                "Motif augmentation is designed for the 'occurency' preprocessor. "
                f"Applying it to '{preprocessor_key}' will append motif counts to the "
                "fingerprint vector, which may not be meaningful."
            )

        all_motifs: List[str] = []
        for peptide in data["peptide"]:
            all_motifs.extend(get_motifs(peptide, motifs))

        motif_histogram = pd.value_counts(all_motifs)
        selected_motifs = motif_histogram[motif_histogram.values >= 20].index.to_list()

        data["input"] = data.apply(
            lambda row: np.concatenate(
                [row["input"], MOTIF_REGISTRY[motifs](row["peptide"], selected_motifs)]
            ),
            axis=1,
        )
        feature_names = feature_names + selected_motifs

    return feature_names, data


# ── Training + SHAP ──────────────────────────────────────────────────────────

def _make_explainer(model, model_key: str, x_background: np.ndarray):
    """Return a SHAP explainer appropriate for the model type."""
    if model_key in TREE_MODELS:
        return shap.TreeExplainer(model)
    # Fallback: KernelExplainer with k-means summary to keep it tractable
    background = shap.kmeans(x_background, min(50, len(x_background)))
    return shap.KernelExplainer(model.predict_proba, background)


def _extract_positive_class_shap(shap_values) -> np.ndarray:
    """Extract SHAP values for the positive class (aggregation=1).

    Handles both:
    - 3-D array of shape (n_samples, n_features, n_classes)  — XGB multioutput
    - list of arrays [class_0, class_1]  — sklearn tree ensembles
    - 2-D array (n_samples, n_features) — already single output
    """
    if isinstance(shap_values, list):
        # sklearn TreeExplainer returns [shap_class0, shap_class1]
        return shap_values[1]
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # XGBClassifier with multi-output: shape (n, feats, classes)
            return shap_values[:, :, 1]
        return shap_values  # already 2-D
    # shap.Explanation object (newer SHAP API)
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3:
            return vals[:, :, 1]
        return vals
    raise TypeError(f"Unrecognised SHAP output type: {type(shap_values)}")


def train_and_explain(
    data: pd.DataFrame,
    model_key: str,
    n_repeats: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_test_all: List[np.ndarray] = []
    shap_values_all: List[np.ndarray] = []
    f1_test: List[float] = []

    for _ in tqdm.tqdm(range(n_repeats)):
        train_set, test_set = train_test_split(data, shuffle=True)

        X_train = np.stack(train_set["input"].to_list())
        y_train = np.stack(train_set["label"].to_list())
        X_test = np.stack(test_set["input"].to_list())
        y_test = test_set["label"].to_list()

        model = MODEL_REGISTRY[model_key]()
        model.fit(X_train, y_train)

        pred_test = model.predict(X_test)
        f1_test.append(f1_score(y_test, pred_test, average="micro"))

        explainer = _make_explainer(model, model_key, X_train)
        raw_shap = explainer.shap_values(X_test)
        shap_pos = _extract_positive_class_shap(raw_shap)

        x_test_all.append(X_test)
        shap_values_all.append(shap_pos)

    return (
        np.array(f1_test),
        np.concatenate(x_test_all),
        np.concatenate(shap_values_all),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option(
    "--preprocessor",
    type=click.Choice(list(PREPROCESSOR_REGISTRY.keys())),
    default="occurency",
    show_default=True,
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default="xgb",
    show_default=True,
)
@click.option(
    "--motifs",
    type=click.Choice(["no_motifs", "1_2", "1_3"]),
    default="no_motifs",
    show_default=True,
)
@click.option("--normalise", type=bool, default=True, show_default=True)
@click.option("--n_fingerprint_bits", type=int, default=128, show_default=True)
@click.option("--n_repeats", type=int, default=50, show_default=True)
@click.option("--seed", type=int, default=3245, show_default=True)
@click.option(
    "--color_bar_label",
    type=str,
    default=None,
    help="Override SHAP colour-bar label. Auto-detected from preprocessor if omitted.",
)
def main(
    data_path: Path,
    output_path: Path,
    preprocessor: str,
    model: str,
    motifs: str,
    normalise: bool,
    n_fingerprint_bits: int,
    n_repeats: int,
    seed: int,
    color_bar_label: Optional[str],
) -> None:

    seed_everything(seed)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessor: {preprocessor}  |  Model: {model}")
    logger.info("Loading data")
    feature_names, data = load_data(
        data_path, preprocessor, motifs, normalise, n_fingerprint_bits, seed
    )

    logger.info("Training models + computing SHAP values")
    f1_test_np, x_test, shap_values = train_and_explain(data, model, n_repeats)
    logger.info(f"F1 Score: {np.mean(f1_test_np):.3f}±{np.std(f1_test_np):.3f}")

    # ── SHAP summary plot ────────────────────────────────────────────────────
    custom_colours = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.000, (0.537, 0.627, 0.612)),
            (0.500, (0.741, 0.482, 0.424)),
            (1.000, (0.890, 0.706, 0.275)),
        ],
    )

    _cbar_defaults = {
        "occurency": "Amino Acid Occurrence (length normalised)",
        "fingerprint": "Morgan Fingerprint Bit Value",
        "protected_fingerprint": "Protected Morgan Fingerprint Bit Value",
        "positional_fingerprint": "Position-weighted Fingerprint Bit Value",
        "whole_peptide_fingerprint": "Whole-peptide Fingerprint Bit Value",
        "protected_whole_peptide_fingerprint": "Protected Whole-peptide Fingerprint Bit Value",
    }
    cbar_label = color_bar_label or _cbar_defaults.get(preprocessor, "Feature Value")

    # For fingerprint preprocessors with many bits, cap display at top-20
    max_display = 20

    plt.figure(dpi=300)
    shap.summary_plot(
        shap_values,
        x_test,
        feature_names=feature_names,
        plot_size=(13, 7.5),
        show=False,
        sort=True,
        color_bar_label=cbar_label,
        cmap=custom_colours,
        max_display=max_display,
    )
    plt.xlabel("Impact on Aggregation (higher = more aggregating)")
    plt.title(f"SHAP — {preprocessor} / {model}", fontsize=10, pad=8)

    out_png = output_path / f"explainer_{preprocessor}_{model}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    logger.info(f"Saved SHAP plot → {out_png}")

    # ── Save mean |SHAP| ranking as CSV for easy comparison ─────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Pad feature_names if needed (e.g. motifs were appended)
    fn = list(feature_names)
    if len(fn) < len(mean_abs_shap):
        fn += [f"extra_feat{i}" for i in range(len(mean_abs_shap) - len(fn))]

    shap_df = (
        pd.DataFrame({"feature": fn[:len(mean_abs_shap)], "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    out_csv = output_path / f"shap_ranking_{preprocessor}_{model}.csv"
    shap_df.to_csv(out_csv, index=False)
    logger.info(f"Saved SHAP ranking → {out_csv}")
    logger.info("Top-10 features:\n" + shap_df.head(10).to_string(index=False))
