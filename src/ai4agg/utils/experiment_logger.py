import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Plain-language descriptions, written for readers who are neither
# programmers nor chemists. Used to annotate per-run log files.

LOADER_DESCRIPTIONS = {
    "whole_set": "Looking only at each finished peptide as a whole (one entry per peptide).",
    "reaction_set": "Looking at every step of building each peptide, one amino acid at a time.",
    "whole_set_shuffled": (
        "Same as 'whole peptide', but the order of amino acids in each peptide "
        "was randomly shuffled. This is a sanity check: if the model still "
        "performs well on shuffled peptides, it is probably not learning "
        "anything meaningful about the real sequence."
    ),
    "wof_set": "Looking at the synthesis steps around the point where aggregation first appears.",
}

PREPROCESSOR_DESCRIPTIONS = {
    "sequence": "Represents each peptide as its raw sequence of amino acid letters, turned into numbers.",
    "one_hot": "Represents each amino acid as a simple marker for which of the ~20 amino acids it is.",
    "fingerprint": "Represents each amino acid by its molecular structure (a 'fingerprint').",
    "protected_fingerprint": "Like 'fingerprint', but also includes the chemical protecting groups used during synthesis.",
    "positional_fingerprint": "Like 'fingerprint', but gives more weight to amino acids added earlier during synthesis.",
    "whole_peptide_fingerprint": "Represents the entire peptide as a single molecular fingerprint (not per amino acid).",
    "protected_whole_peptide_fingerprint": "Like 'whole_peptide_fingerprint', but also includes the protecting groups.",
    "occurency": "Counts how many times each amino acid appears in the peptide.",
    "hydrophobicity": "Uses how water-repelling ('hydrophobic') each amino acid is, at every position in the peptide.",
    "protected_fp_hydrophob": "Combines the protected molecular fingerprint with hydrophobicity information.",
    "all_synthesis_hydrophob": (
        "Combines hydrophobicity with every recorded synthesis-machine detail "
        "(coupling agent, machine, temperatures, flow rate, peak shapes, etc.)."
    ),
}

MODEL_DESCRIPTIONS = {
    "rff": "Random Forest -- a large committee of decision trees that vote on the answer.",
    "xgb": "XGBoost -- a strong tree-based model that corrects its own mistakes step by step.",
    "knn": "K-Nearest Neighbours -- predicts based on the most similar peptides seen before.",
    "gaussian": "Gaussian Process -- a probability-based model that also estimates its own uncertainty.",
    "hc2": "HIVE-COTE 2.0 -- an ensemble of many time-series classification methods.",
    "timeforest": "Time Series Forest -- decision trees applied to the peptide as a sequence over time.",
    "weasel": "WEASEL -- a time-series classification method based on recurring symbolic patterns.",
}


def _describe(mapping: dict, key: str) -> str:
    return mapping.get(key, f"(no description available for '{key}')")


@contextmanager
def experiment_run_log(
    run_dir: Path, loader: str, preprocessor: str, model: str, pg_scheme: str
) -> Iterator[Path]:
    """Write a clean, annotated, non-technical log file to `run_dir/log.txt`.

    While the context is active, INFO-level log records from the whole
    `ai4agg` package are written to this file (in addition to whatever the
    caller's console handlers show), and the very verbose raw-dataframe
    dumps from `data_pipeline` are suppressed everywhere.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    header = [
        "=" * 70,
        "EXPERIMENT LOG",
        "=" * 70,
        "",
        "'Aggregation' means the peptide clumped together during synthesis",
        "instead of staying dissolved -- something chemists want to predict",
        "and avoid. This experiment checks how well a model can predict, from",
        "the peptide and its synthesis conditions, whether aggregation happens.",
        "",
        f"Data split:              {loader}",
        f"  -> {_describe(LOADER_DESCRIPTIONS, loader)}",
        "",
        f"Data representation:    {preprocessor}",
        f"  -> {_describe(PREPROCESSOR_DESCRIPTIONS, preprocessor)}",
        "",
        f"Prediction model:        {model}",
        f"  -> {_describe(MODEL_DESCRIPTIONS, model)}",
        "",
        f"Protecting-group scheme: {pg_scheme}",
        "",
        "-" * 70,
        "DETAILED RUN LOG",
        "-" * 70,
        "",
    ]

    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("\n".join(header) + "\n")
    log_file.flush()

    file_handler = logging.StreamHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
    )

    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    data_pipeline_logger = logging.getLogger("data_pipeline")
    previous_data_pipeline_level = data_pipeline_logger.level
    data_pipeline_logger.setLevel(logging.WARNING)

    try:
        yield log_path
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_root_level)
        data_pipeline_logger.setLevel(previous_data_pipeline_level)
        log_file.close()


def append_results_summary(run_dir: Path, results: dict) -> None:
    """Append a plain-language summary of `results` (results.json) to log.txt."""
    log_path = run_dir / "log.txt"
    lines = ["", "-" * 70, "RESULT SUMMARY", "-" * 70, ""]

    if not results:
        lines.append("This run did not finish successfully -- no results were produced.")
        lines.append("See the detailed run log above for error messages.")
    elif results.get("task") == "regression":
        mse = results.get("mse", {})
        pearson = results.get("pearson", {})
        lines.append(
            "The model tried to predict a numeric value (how the peptide's "
            "synthesis 'first difference' changes)."
        )
        lines.append(
            f"On data it had never seen before, its average squared error was "
            f"{mse.get('mean', float('nan')):.4f} "
            f"(+/- {mse.get('std', float('nan')):.4f}). Lower is better."
        )
        lines.append(
            f"The correlation between predicted and actual values was "
            f"{pearson.get('mean', float('nan')):.3f} "
            f"(1.0 = perfect agreement, 0 = no relationship)."
        )
    else:
        acc = results.get("accuracy", {})
        f1 = results.get("f1", {})
        acc_mean = acc.get("mean", float("nan"))
        lines.append(
            f"Across repeated train/test splits, the model correctly predicted "
            f"whether a peptide would aggregate in {acc_mean * 100:.1f}% of cases "
            f"on average (+/- {acc.get('std', float('nan')) * 100:.1f} percentage points)."
        )
        lines.append(
            f"Its F1 score (a balance between catching aggregating peptides and "
            f"not over-predicting aggregation) was {f1.get('mean', float('nan')):.3f} "
            f"(+/- {f1.get('std', float('nan')):.3f}); 1.0 is perfect, 0 is worst."
        )

    lines.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
