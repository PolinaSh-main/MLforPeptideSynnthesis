import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier
from xgboost import XGBClassifier, XGBRegressor

from ..utils.loaders import (
    make_reaction_set,
    make_shuffled_peptide_set,
    make_whole_peptide_set,
    make_wof_peptide_set,
)
from ..utils.preprocessors import (
    CompositePreprocessor,
    FingerprintPreprocessor,
    HydrophobicityPreprocessor,
    OccurencyVectorPreprocessor,
    OneHotPreprocessor,
    PositionalFingerprintPreprocessor,
    ProtectedFingerprintPreprocessor,
    ProtectedWholePeptideFingerprintPreprocessor,
    SequencePreprocessor,
    WholePeptideFingerprintPreprocessor,
)
from ..features import (
    ProtectedFingerprintFeature,
    HydrophobicityFeature,
    CouplingAgentFeature,
    MachineFeature,
    TempCouplingFeature,
    CouplingStrokesFeature,
    DeprotectionStrokesFeature,
    FlowRateFeature,
    TempReactorFeature,
    FirstAreaFeature,
    FirstHeightFeature,
    FirstWidthFeature,
    PrevAreaFeature,
    PrevHeightFeature,
    PrevWidthFeature,
    PrevDiffFeature,
)

# All synthesis-metadata features (UZH+MIT columns) plus per-residue hydrophobicity.
ALL_SYNTHESIS_FEATURES = [
    CouplingAgentFeature,
    MachineFeature,
    TempCouplingFeature,
    CouplingStrokesFeature,
    DeprotectionStrokesFeature,
    FlowRateFeature,
    TempReactorFeature,
    FirstAreaFeature,
    FirstHeightFeature,
    FirstWidthFeature,
    PrevAreaFeature,
    PrevHeightFeature,
    PrevWidthFeature,
    PrevDiffFeature,
    HydrophobicityFeature,
]
from ..utils.utils import seed_everything, split_peptide_set
from ..utils.data_logger import log_dataframe
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


LOADER_REGISTRY = {
    "reaction_set": make_reaction_set,
    "wof_set": make_wof_peptide_set,
    "whole_set": make_whole_peptide_set,
    "whole_set_shuffled": make_shuffled_peptide_set,
}

PREPROCESSOR_REGISTRY = {
    "sequence": SequencePreprocessor,
    "one_hot": OneHotPreprocessor,
    "fingerprint": FingerprintPreprocessor,
    "protected_fingerprint": ProtectedFingerprintPreprocessor,
    "positional_fingerprint": PositionalFingerprintPreprocessor,
    "whole_peptide_fingerprint": WholePeptideFingerprintPreprocessor,
    "protected_whole_peptide_fingerprint": ProtectedWholePeptideFingerprintPreprocessor,
    "occurency": OccurencyVectorPreprocessor,
    "hydrophobicity": HydrophobicityPreprocessor,
    # New features can be combined here without touching any other code —
    # see src/ai4agg/features/README.md
    "protected_fp_hydrophob": partial(CompositePreprocessor, [ProtectedFingerprintFeature, HydrophobicityFeature]),
    "all_synthesis_hydrophob": partial(CompositePreprocessor, ALL_SYNTHESIS_FEATURES),
}

# Classification-only models
MODEL_REGISTRY = {
    "rff": RandomForestClassifier,
    "xgb": XGBClassifier,
    "knn": KNeighborsClassifier,
    "gaussian": GaussianProcessClassifier,
    "hc2": partial(HIVECOTEV2, time_limit_in_minutes=10),
    "timeforest": TimeSeriesForestClassifier,
    "weasel": WEASEL,
}

# Regression models — keyed separately so --model can select them under --task regression
REGRESSION_MODEL_REGISTRY = {
    "rff": RandomForestRegressor,
    "xgb": XGBRegressor,
}

# Union of keys exposed to --model CLI option
_ALL_MODEL_KEYS = sorted(set(list(MODEL_REGISTRY) + list(REGRESSION_MODEL_REGISTRY)))


def load_data(
    data_path: Path,
    loader: str,
    preprocessor: str,
    cv_split: int = 0,
    seed: int = 3245,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    logger.info(f"\n[load_data] Starting data loading and preprocessing")
    logger.info(f"  loader={loader}, preprocessor={preprocessor}, cv_split={cv_split}, seed={seed}")
    
    dataset = LOADER_REGISTRY[loader](data_path, seed=seed, **kwargs)  # type: ignore
    logger.info(f"[load_data] After loader: {len(dataset)} rows, columns: {dataset.columns.tolist()}")

    preprocessor_instance = PREPROCESSOR_REGISTRY[preprocessor](
        dataset,
        random_state=seed,
        **kwargs,
    )
    logger.info(f"[load_data] Preprocessor '{preprocessor}' initialized")
    
    if isinstance(preprocessor_instance, CompositePreprocessor):
        dataset["input"] = dataset.apply(preprocessor_instance, axis=1)
    else:
        dataset["input"] = dataset["peptide"].map(preprocessor_instance)
    logger.info(f"[load_data] After preprocessing: input column added with shape examples")
    if len(dataset) > 0:
        first_input = dataset["input"].iloc[0]
        logger.info(f"[load_data] First input shape: {first_input.shape if hasattr(first_input, 'shape') else 'N/A'}")


    if loader == "whole_set_shuffled":
        train_set, test_set = train_test_split(dataset, random_state=seed)
        dataset_dict = {"train": train_set, "test": test_set}
        logger.info(f"[load_data] Train/test split: {len(train_set)} train, {len(test_set)} test")
    else:
        dataset_dict = split_peptide_set(dataset, val=False, cv_split=cv_split, seed=seed)
        logger.info(f"[load_data] CV split {cv_split}: {len(dataset_dict['train'])} train, {len(dataset_dict['test'])} test")

    # Log split statistics
    for split_name in ["train", "test"]:
        if split_name in dataset_dict:
            df = dataset_dict[split_name]
            logger.info(f"[load_data] {split_name.upper()}: {len(df)} rows, columns: {df.columns.tolist()}")
            if "aggregation" in df.columns:
                agg_count = df["aggregation"].sum()
                logger.info(f"[load_data]   aggregation: {agg_count} True, {len(df) - agg_count} False")

    return dataset_dict


def train(dataset_dict: Dict[str, pd.DataFrame], model: str) -> Tuple[Any, Any, pd.DataFrame]:
    logger.info(f"\n[train] Training classification model: {model}")
    logger.info(f"[train] Train set: {len(dataset_dict['train'])} samples")
    
    classifier = MODEL_REGISTRY[model]()
    classifier.fit(
        np.stack(dataset_dict["train"]["input"].to_list()),
        np.stack(dataset_dict["train"]["aggregation"].to_list()),
    )
    logger.info(f"[train] Model trained successfully")

    test_set = dataset_dict["test"].copy()
    test_set["prediction"] = classifier.predict(np.stack(test_set["input"].to_list()))
    logger.info(f"[train] Predictions made on {len(test_set)} test samples")
    
    if hasattr(classifier, "predict_proba"):
        test_set["prediction_probability"] = classifier.predict_proba(
            np.stack(test_set["input"].to_list())
        )[:, 1]
        logger.info(f"[train] Prediction probabilities computed")

    ground_truth = []
    predictions = []
    for serial in test_set["serial"].unique():
        subset = test_set[test_set["serial"] == serial]
        ground_truth.append(np.any(subset["aggregation"]))
        predictions.append(np.any(subset["prediction"]))

    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    logger.info(f"[train] RESULTS — F1: {f1:.4f}, Accuracy: {accuracy:.4f} (on {len(ground_truth)} serials)")
    log_dataframe(test_set, "test_set", "AFTER_PREDICTIONS")
    return f1, accuracy, test_set


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def train_regression(
    dataset_dict: Dict[str, pd.DataFrame],
    model: str,
    target_column: str = "first_diff_clean",
) -> Tuple[float, float, pd.DataFrame]:
    logger.info(f"\n[train_regression] Training regression model: {model}")
    logger.info(f"[train_regression] Train set: {len(dataset_dict['train'])} samples")
    logger.info(f"[train_regression] Target column: {target_column}")

    if model not in REGRESSION_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model}' is not available for regression. "
            f"Available: {list(REGRESSION_MODEL_REGISTRY)}."
        )

    regressor = REGRESSION_MODEL_REGISTRY[model]()
    regressor.fit(
        np.stack(dataset_dict["train"]["input"].to_list()),
        np.stack(dataset_dict["train"][target_column].to_list()),
    )
    logger.info(f"[train_regression] Model trained successfully")

    test_set = dataset_dict["test"].copy()
    test_set["prediction"] = regressor.predict(np.stack(test_set["input"].to_list()))
    logger.info(f"[train_regression] Predictions made on {len(test_set)} test samples")

    y_true = test_set[target_column].to_numpy(dtype=float)
    y_pred = test_set["prediction"].to_numpy(dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    pearson = pearson_correlation(y_true, y_pred)
    logger.info(f"[train_regression] RESULTS — MSE: {mse:.4f}, Pearson: {pearson:.4f}")
    log_dataframe(test_set, "test_set", "AFTER_REGRESSION_PREDICTIONS")
    return mse, pearson, test_set


def add_prediction_frame(
    prediction_frames: list,
    test_predictions: pd.DataFrame,
    split: int,
) -> None:
    test_predictions = test_predictions.copy()
    test_predictions["split"] = split
    prediction_frames.append(test_predictions)


def save_predictions(
    prediction_frames: list,
    output_path: Path,
    target_column: str,
) -> None:
    if not prediction_frames:
        return

    all_predictions = pd.concat(prediction_frames)
    prediction_columns = [
        col for col in [
            "split",
            "serial",
            "peptide",
            "aggregation",
            target_column,
            "prediction",
            "prediction_probability",
        ]
        if col in all_predictions.columns
    ]
    all_predictions[prediction_columns].to_csv(
        output_path / "step_predictions.csv",
        index=False,
    )


@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--loader", type=click.Choice(list(LOADER_REGISTRY.keys())))
@click.option("--preprocessor", type=click.Choice(list(PREPROCESSOR_REGISTRY.keys())))
@click.option("--model", type=click.Choice(_ALL_MODEL_KEYS))
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--target_column", type=str, default="first_diff_clean",
              help="Regression target column (default: first_diff_clean).")
@click.option("--seed", type=int, default=3245)
@click.option("--n_repeats", type=int, default=0,
              help="Number of repeats for whole_set_shuffled loader.")
@click.option("--wof_start", type=int, required=False)
@click.option("--wof_end", type=int, required=False)
@click.option("--wof_drop", type=bool, default=False)
@click.option("--occurency_vector_normalise", type=bool, required=False)
@click.option("--pg_scheme", type=str, default="default_uzh_mit", show_default=True,
              help="Name of the protecting-group scheme (key in resources/pg_scheme.json), "
                   "used by the 'hydrophobicity' and 'protected_fp_hydrophob' preprocessors.")
@click.option("--save_step_predictions", is_flag=True,
              help="Save per-step predictions to step_predictions.csv in output_path.")
def main(
    data_path: Path,
    output_path: Path,
    loader: str,
    preprocessor: str,
    model: str,
    task: str,
    target_column: str,
    seed: int,
    n_repeats: int,
    wof_start: Optional[int],
    wof_end: Optional[int],
    wof_drop: bool,
    occurency_vector_normalise: Optional[bool],
    pg_scheme: str,
    save_step_predictions: bool,
) -> None:
    logger.info("="*80)
    logger.info("TRAIN SKLEARN MODELS - STARTING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Task: {task}")
    logger.info(f"  Loader: {loader}")
    logger.info(f"  Preprocessor: {preprocessor}")
    logger.info(f"  Model: {model}")
    if task == "regression":
        logger.info(f"  Target column: {target_column}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Seed: {seed}")
    if loader == "whole_set_shuffled":
        logger.info(f"  N repeats: {n_repeats}")
    logger.info("="*80)

    # ── Early validation ────────────────────────────────────────────────────
    if task == "regression" and model not in REGRESSION_MODEL_REGISTRY:
        raise click.UsageError(
            f"Model '{model}' is not supported for regression. "
            f"Available regression models: {list(REGRESSION_MODEL_REGISTRY)}."
        )

    if model in ["hc2", "timeforest", "weasel"] and (
        loader != "whole_set" or preprocessor != "sequence"
    ):
        raise click.UsageError(
            f"Model '{model}' requires loader='whole_set' and preprocessor='sequence'."
        )

    if loader == "wof_set" and (wof_start is None or wof_end is None):
        raise click.UsageError(
            "Loader 'wof_set' requires --wof_start and --wof_end to be specified."
        )

    seed_everything(seed)
    output_path.mkdir(parents=True, exist_ok=True)

    f1, accuracy = [], []
    mse, pearson = [], []
    prediction_frames: list = []

    if loader == "whole_set_shuffled":
        for i in tqdm.tqdm(range(n_repeats)):
            dataset_dict = load_data(
                data_path,
                loader,
                preprocessor,
                cv_split=0,
                padding=True,
                wof_start=wof_start,
                wof_end=wof_end,
                wof_drop=wof_drop,
                occurency_vector_normalise=occurency_vector_normalise,
                pg_scheme=pg_scheme,
                seed=i,
            )

            if task == "regression":
                if target_column not in dataset_dict["train"].columns:
                    raise ValueError(
                        f"Target column '{target_column}' not in dataset. "
                        "Check that --loader produces it (e.g. whole_set with first_diff_clean)."
                    )
                mse_i, pearson_i, test_predictions = train_regression(dataset_dict, model, target_column)
                mse.append(mse_i)
                pearson.append(pearson_i)
            else:
                f1_i, acc_i, test_predictions = train(dataset_dict, model)
                f1.append(f1_i)
                accuracy.append(acc_i)

            add_prediction_frame(prediction_frames, test_predictions, i)

    else:
        for cv_split in range(5):
            logger.info(f"Running Split {cv_split + 1}/5")

            dataset_dict = load_data(
                data_path,
                loader,
                preprocessor,
                cv_split=cv_split,
                padding=True,
                wof_start=wof_start,
                wof_end=wof_end,
                wof_drop=wof_drop,
                occurency_vector_normalise=occurency_vector_normalise,
                pg_scheme=pg_scheme,
                seed=seed,
            )

            if task == "regression":
                if target_column not in dataset_dict["train"].columns:
                    raise ValueError(
                        f"Target column '{target_column}' not in dataset. "
                        "Check that --loader produces it (e.g. whole_set with first_diff_clean)."
                    )
                mse_i, pearson_i, test_predictions = train_regression(dataset_dict, model, target_column)
                logger.info(
                    f"Split {cv_split + 1}/5 — MSE: {mse_i:.4f}  Pearson: {pearson_i:.4f}"
                )
                mse.append(mse_i)
                pearson.append(pearson_i)
            else:
                f1_i, acc_i, test_predictions = train(dataset_dict, model)
                logger.info(
                    f"Split {cv_split + 1}/5 — F1: {f1_i:.4f}  Accuracy: {acc_i:.4f}"
                )
                f1.append(f1_i)
                accuracy.append(acc_i)

            add_prediction_frame(prediction_frames, test_predictions, cv_split)

    if save_step_predictions:
        save_predictions(prediction_frames, output_path, target_column)
        logger.info(f"[MAIN] Step predictions saved to {output_path / 'step_predictions.csv'}")

    logger.info(f"\n[MAIN] FINAL RESULTS")
    logger.info("="*80)
    
    with (output_path / "results.json").open("w") as results_file:
        if task == "regression":
            logger.info(
                f"MSE:     {np.mean(mse):.4f} ± {np.std(mse):.4f}"
            )
            logger.info(
                f"Pearson: {np.nanmean(pearson):.4f} ± {np.nanstd(pearson):.4f}"
            )
            logger.info(f"Raw MSE values: {[f'{v:.4f}' for v in mse]}")
            logger.info(f"Raw Pearson values: {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in pearson]}")
            json.dump(
                {
                    "task": "regression",
                    "model": model,
                    "preprocessor": preprocessor,
                    "target_column": target_column,
                    "mse": {
                        "mean": float(np.mean(mse)),
                        "std": float(np.std(mse)),
                    },
                    "pearson": {
                        "mean": float(np.nanmean(pearson)),
                        "std": float(np.nanstd(pearson)),
                    },
                    "raw_mse": list(mse),
                    "raw_pearson": [p if not np.isnan(p) else None for p in pearson],
                },
                results_file,
                indent=2,
            )
        else:
            logger.info(
                f"Accuracy: {np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}"
            )
            logger.info(
                f"F1:       {np.mean(f1):.4f} ± {np.std(f1):.4f}"
            )
            logger.info(f"Raw Accuracy values: {[f'{v:.4f}' for v in accuracy]}")
            logger.info(f"Raw F1 values: {[f'{v:.4f}' for v in f1]}")
            json.dump(
                {
                    "task": "classification",
                    "model": model,
                    "preprocessor": preprocessor,
                    "f1": {
                        "mean": float(np.mean(f1)),
                        "std": float(np.std(f1)),
                    },
                    "accuracy": {
                        "mean": float(np.mean(accuracy)),
                        "std": float(np.std(accuracy)),
                    },
                    "raw_f1": list(f1),
                    "raw_acc": list(accuracy),
                },
                results_file,
                indent=2,
            )
    
    logger.info(f"Results saved to {output_path / 'results.json'}")
    logger.info("="*80)
    logger.info("TRAIN SKLEARN MODELS - COMPLETED")
    logger.info("="*80)

