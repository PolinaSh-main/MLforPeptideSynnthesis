import json
from pathlib import Path
from importlib.resources import files
import logging

import click
import pandas as pd

from ..utils.data_logger import log_dataframe
from ..utils.dataset_loader import load_dataset, METADATA_COLUMNS, SYNTH_STEP_COLUMNS

logger = logging.getLogger(__name__)

DATASET_CONFIGS_DIR = files("ai4agg") / "resources/dataset_configs"


def clean_cys_his(synthesis_data: pd.DataFrame, row: object) -> float:
    """Interpolate first_diff for Cys/His to clean temperature artifacts."""
    if not (row["amino_acid"] == "H" or row["amino_acid"] == "C"): # type: ignore
        return row["first_diff"] # type: ignore
    else:
        first_index = synthesis_data.iloc[0].name
        last_index = synthesis_data.iloc[-1].name
        # Cys His is the second amino acid -> Interpolate between 0 and the second
        if row.name == first_index: # type: ignore
            interpolated_diff = synthesis_data.loc[row.name + 1]["first_diff"] / 2 # type: ignore
        # Cys His is the last amino acid -> Use the previous diff
        elif row.name == last_index: # type: ignore
            interpolated_diff = synthesis_data.loc[row.name - 1]["first_diff"] # type: ignore
        # Interpolate
        else:
            interpolated_diff = (
                synthesis_data.loc[row.name - 1]["first_diff"] # type: ignore
                + synthesis_data.loc[row.name + 1]["first_diff"] # type: ignore
            ) / 2

        return interpolated_diff


def get_clean_unique_peptides(peptide_df: pd.DataFrame):
    logger.info(f"[get_clean_unique_peptides] Input: {len(peptide_df)} rows, {peptide_df['serial'].nunique()} serials")

    unique_peptides = list()
    unique_synth_sets = list()
    duplicates_removed = 0

    for serial in peptide_df['serial'].unique():
        subset = peptide_df[peptide_df['serial'] == serial]
        final_peptide = subset['peptide'].iloc[-1]

        if final_peptide in unique_peptides:
            duplicates_removed += 1
            continue

        unique_peptides.append(final_peptide)
        unique_synth_sets.append(subset)

    result = pd.concat(unique_synth_sets)
    logger.info(f"[get_clean_unique_peptides] Output: {len(result)} rows | Duplicates removed: {duplicates_removed}")
    return result


@click.command()
@click.option("--dataset", "datasets", multiple=True,
               type=(Path, Path),
               help="Pair of: path to a dataset CSV and path to its JSON config "
                    "(see src/ai4agg/resources/dataset_configs/). Can be given multiple times.")
@click.option("--uzh_dataset_path", type=Path, required=False,
               help="Deprecated alias for --dataset <path> dataset_configs/uzh.json")
@click.option("--mit_dataset_path", type=Path, required=False,
               help="Deprecated alias for --dataset <path> dataset_configs/mit.json")
@click.option("--save_path", type=Path, required=True)
@click.option("--aggregation_threshold", type=float, default=-0.2)
@click.option("--max_length", type=int, default=20, help="Maximum peptide length (default: 20)")
@click.option("--truncate/--no-truncate", default=True,
               help="Truncate each synthesis to the first `max_length` amino acids (default: enabled). "
                    "With --no-truncate, full-length peptides are kept regardless of max_length.")
def main(
    datasets: tuple,
    uzh_dataset_path: Path,
    mit_dataset_path: Path,
    save_path: Path,
    aggregation_threshold: float,
    max_length: int,
    truncate: bool,
):
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    dataset_pairs = list(datasets)
    if uzh_dataset_path is not None:
        dataset_pairs.append((uzh_dataset_path, Path(DATASET_CONFIGS_DIR / "uzh.json")))
    if mit_dataset_path is not None:
        dataset_pairs.append((mit_dataset_path, Path(DATASET_CONFIGS_DIR / "mit.json")))

    if not dataset_pairs:
        raise click.UsageError(
            "No datasets provided. Use --dataset <csv> <config.json> (repeatable), "
            "or the deprecated --uzh_dataset_path / --mit_dataset_path."
        )

    logger.info("="*80)
    logger.info("Starting combined dataset creation")
    for csv_path, config_path in dataset_pairs:
        logger.info(f"  Dataset: {csv_path}  (config: {config_path})")
    logger.info(f"  Save path: {save_path}")
    logger.info(f"  Aggregation threshold: {aggregation_threshold}")
    logger.info(f"  Max peptide length: {max_length}")
    logger.info(f"  Truncate: {truncate}")
    logger.info("="*80)

    # Load + process each dataset via its config
    logger.info("\n[STEP 1] Loading and processing datasets")
    processed_datasets = []
    for csv_path, config_path in dataset_pairs:
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"[STEP 1] Loading '{config.get('name', csv_path)}' from {csv_path}")
        processed = load_dataset(csv_path, config, max_length=max_length, truncate=truncate)
        logger.info(f"[STEP 1] '{config.get('name', csv_path)}': {len(processed)} rows, {processed['serial'].nunique()} serials")
        processed_datasets.append(processed)

    # Combine datasets
    logger.info("\n[STEP 2] Combining datasets")
    combined_dataset = pd.concat(processed_datasets, ignore_index=True)
    logger.info(f"[STEP 2] Combined dataset: {len(combined_dataset)} rows, {combined_dataset['serial'].nunique()} serials")
    log_dataframe(combined_dataset, "combined_dataset", "AFTER_CONCAT")

    # Clean Cysteine and Histidine peaks (Temperature change causes artifacts)
    logger.info("\n[STEP 3] Cleaning Cys/His artifacts via interpolation")
    combined_dataset["first_diff_clean"] = combined_dataset.apply(
        lambda row: clean_cys_his(
            combined_dataset[combined_dataset["serial"] == row["serial"]], row
        ),
        axis=1,
    )
    cys_his_affected = (combined_dataset["first_diff"] != combined_dataset["first_diff_clean"]).sum()
    logger.info(f"[STEP 3] Cys/His interpolation: {cys_his_affected} rows modified")
    log_dataframe(combined_dataset, "combined_dataset", "AFTER_CYS_HIS_CLEAN")

    # Drop Duplicates
    logger.info("\n[STEP 4] Removing duplicate peptides")
    pre_dedup_count = len(combined_dataset)
    combined_dataset = get_clean_unique_peptides(combined_dataset)
    logger.info(f"[STEP 4] Deduplication: {pre_dedup_count} → {len(combined_dataset)} rows")

    # Add aggregation
    logger.info("\n[STEP 5] Adding aggregation labels")
    combined_dataset['aggregation'] = (
        combined_dataset['first_diff_clean'] < aggregation_threshold
    )
    agg_count = combined_dataset['aggregation'].sum()
    logger.info(f"[STEP 5] Aggregation labels: {agg_count} True, {len(combined_dataset) - agg_count} False (threshold={aggregation_threshold})")
    log_dataframe(combined_dataset, "combined_dataset", "AFTER_AGGREGATION")

    # Save
    logger.info("\n[STEP 6] Saving to CSV")
    output_columns = (
        ['serial', 'peptide', 'first_diff_clean', 'aggregation']
        + METADATA_COLUMNS
        + SYNTH_STEP_COLUMNS
    )
    combined_dataset = combined_dataset[output_columns]
    combined_dataset.to_csv(save_path)
    logger.info(f"[STEP 6] Saved {len(combined_dataset)} rows to {save_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
