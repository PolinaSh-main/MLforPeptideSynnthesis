import json
import logging
from itertools import product
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd

from .explainability import main as explain_main
from .explainability import PREPROCESSOR_REGISTRY as EXPLAIN_PREPROCESSOR_REGISTRY
from .train_sklearn_models import main as train_main
from .train_sklearn_models import LOADER_REGISTRY, PREPROCESSOR_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Loaders to sweep. "wof_set" is excluded — it needs extra --wof_start/--wof_end
# parameters that don't make sense to grid-search generically.
DEFAULT_LOADERS = ("whole_set", "reaction_set", "whole_set_shuffled")

# Fast classification models only (rff, xgb, knn). hc2/timeforest/weasel/gaussian
# are excluded — they're slow on the full grid and (for hc2/timeforest/weasel)
# only support loader="whole_set" + preprocessor="sequence".
DEFAULT_MODELS = ("rff", "xgb", "knn")

# All vector preprocessors registered for training.
DEFAULT_PREPROCESSORS = tuple(PREPROCESSOR_REGISTRY.keys())

DEFAULT_PG_SCHEMES = ("default_uzh_mit",)


def _read_results_json(run_dir: Path) -> dict:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}
    with results_path.open() as f:
        return json.load(f)


@click.command()
@click.option("--data_path", type=Path, required=True,
               help="Path to combined dataset CSV (see create_combined_data).")
@click.option("--output_path", type=Path, required=True,
               help="Directory where per-combination results and the summary CSV are written.")
@click.option("--loaders", multiple=True, default=DEFAULT_LOADERS, show_default=True,
               type=click.Choice(list(LOADER_REGISTRY.keys())))
@click.option("--preprocessors", multiple=True, default=DEFAULT_PREPROCESSORS, show_default=True,
               type=click.Choice(list(PREPROCESSOR_REGISTRY.keys())))
@click.option("--models", multiple=True, default=DEFAULT_MODELS, show_default=True,
               type=click.Choice(list(DEFAULT_MODELS)))
@click.option("--pg_schemes", multiple=True, default=DEFAULT_PG_SCHEMES, show_default=True,
               help="Protecting-group schemes (keys in resources/pg_scheme.json) to sweep.")
@click.option("--seed", type=int, default=3245, show_default=True)
@click.option("--shuffled_n_repeats", type=int, default=5, show_default=True,
               help="Number of repeats used for the 'whole_set_shuffled' loader.")
@click.option("--shap_n_repeats", type=int, default=20, show_default=True,
               help="Number of train/test repeats used for SHAP analysis.")
@click.option("--skip_existing", is_flag=True, default=False,
               help="Skip combinations whose results.json already exists.")
def main(
    data_path: Path,
    output_path: Path,
    loaders: Tuple[str, ...],
    preprocessors: Tuple[str, ...],
    models: Tuple[str, ...],
    pg_schemes: Tuple[str, ...],
    seed: int,
    shuffled_n_repeats: int,
    shap_n_repeats: int,
    skip_existing: bool,
) -> None:
    """Run a full grid of (loader, preprocessor, model, pg_scheme) combinations,
    collect their results into a single comparison CSV, and run SHAP analysis
    for the best model of each (preprocessor, pg_scheme) combination.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    train_dir = output_path / "train"
    shap_dir = output_path / "shap"

    combinations = list(product(loaders, preprocessors, models, pg_schemes))
    logger.info(f"Running {len(combinations)} (loader, preprocessor, model, pg_scheme) combinations")

    rows = []
    for loader, preprocessor, model, pg_scheme in combinations:
        tag = f"{loader}__{preprocessor}__{model}__{pg_scheme}"
        run_dir = train_dir / tag

        if skip_existing and (run_dir / "results.json").exists():
            logger.info(f"[SKIP] {tag} (results.json already exists)")
        else:
            logger.info(f"[RUN] {tag}")
            args = [
                "--data_path", str(data_path),
                "--output_path", str(run_dir),
                "--loader", loader,
                "--preprocessor", preprocessor,
                "--model", model,
                "--pg_scheme", pg_scheme,
                "--seed", str(seed),
            ]
            if loader == "whole_set_shuffled":
                args += ["--n_repeats", str(shuffled_n_repeats)]

            try:
                train_main(args, standalone_mode=False)
            except Exception:
                logger.exception(f"[FAIL] {tag}")

        results = _read_results_json(run_dir)
        row = {
            "loader": loader,
            "preprocessor": preprocessor,
            "model": model,
            "pg_scheme": pg_scheme,
            "task": results.get("task"),
            "accuracy_mean": results.get("accuracy", {}).get("mean", np.nan),
            "accuracy_std": results.get("accuracy", {}).get("std", np.nan),
            "f1_mean": results.get("f1", {}).get("mean", np.nan),
            "f1_std": results.get("f1", {}).get("std", np.nan),
            "status": "success" if results else "failed",
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        ["preprocessor", "pg_scheme", "accuracy_mean"], ascending=[True, True, False]
    )
    results_csv = output_path / "experiment_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved comparison CSV -> {results_csv}")

    # ── SHAP analysis: best model (rff/xgb) per (preprocessor, pg_scheme) ───
    shap_models = [m for m in models if m in ("xgb", "rff")]
    shap_preprocessors = [p for p in preprocessors if p in EXPLAIN_PREPROCESSOR_REGISTRY]

    if not shap_models:
        logger.warning("None of --models are SHAP-compatible (xgb/rff) — skipping SHAP analysis")
    else:
        for preprocessor in shap_preprocessors:
            for pg_scheme in pg_schemes:
                subset = results_df[
                    (results_df["preprocessor"] == preprocessor)
                    & (results_df["pg_scheme"] == pg_scheme)
                    & (results_df["model"].isin(shap_models))
                    & (results_df["status"] == "success")
                ]
                if subset.empty:
                    logger.warning(f"[SHAP SKIP] {preprocessor} / {pg_scheme}: no successful runs")
                    continue

                best_model = subset.sort_values("accuracy_mean", ascending=False).iloc[0]["model"]
                tag = f"{preprocessor}__{pg_scheme}__{best_model}"
                run_dir = shap_dir / tag
                logger.info(f"[SHAP] {tag} (best model by accuracy)")

                args = [
                    "--data_path", str(data_path),
                    "--output_path", str(run_dir),
                    "--preprocessor", preprocessor,
                    "--model", best_model,
                    "--n_repeats", str(shap_n_repeats),
                    "--seed", str(seed),
                    "--pg_scheme", pg_scheme,
                ]
                try:
                    explain_main(args, standalone_mode=False)
                except Exception:
                    logger.exception(f"[SHAP FAIL] {tag}")

    logger.info("Experiment grid complete")
    logger.info(f"  Comparison CSV: {results_csv}")
    logger.info(f"  Per-combination results: {train_dir}")
    logger.info(f"  SHAP rankings/plots: {shap_dir}")


if __name__ == "__main__":
    main()
