import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.data_logger import log_step, log_dataframe

logger = logging.getLogger(__name__)

METADATA_COLUMNS = ["coupling_agent", "solvent", "resin", "temp_coupling", "pg_scheme"]


# Reaction set: Data as is, one amino acid addition at a time
def make_reaction_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    logger.info(f"\n[make_reaction_set] Loading data from {data_path}")
    data = pd.read_csv(data_path, index_col=0)
    data = data.reset_index(drop=True)
    log_dataframe(data, "make_reaction_set", "OUTPUT")
    return data

# Whole Peptide set: Only consider whole peptides
@log_step("make_whole_peptide_set")
def make_whole_peptide_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    data = make_reaction_set(data_path)
    logger.info(f"[make_whole_peptide_set] Processing {data['serial'].nunique()} unique serials")

    whole_peptide_indices = list()
    aggregation = list()
    first_diff_clean = list()
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]
        whole_peptide_indices.append(subset.iloc[-1].name)
        aggregation.append(subset['aggregation'].sum() >= 1)
        if "first_diff_clean" in subset:
            first_diff_clean.append(subset["first_diff_clean"].min())

    whole_peptide_set = data.loc[whole_peptide_indices]
    logger.info(f"[make_whole_peptide_set] Extracted {len(whole_peptide_set)} whole peptide indices")
    
    whole_peptide_set['aggregation'] = aggregation
    columns = ["peptide", "serial", "aggregation"]
    if first_diff_clean:
        whole_peptide_set["first_diff_clean"] = first_diff_clean
        columns.append("first_diff_clean")
    columns += [col for col in METADATA_COLUMNS if col in whole_peptide_set.columns]
    whole_peptide_set = whole_peptide_set[columns]
    
    logger.info(f"[make_whole_peptide_set] Before dedup: {len(whole_peptide_set)} rows")
    whole_peptide_set = whole_peptide_set.drop_duplicates(subset="peptide")
    logger.info(f"[make_whole_peptide_set] After dedup: {len(whole_peptide_set)} rows")

    return whole_peptide_set


# Whole Peptide Set shuffled: Consider whole peptides but shuffle them
def make_shuffled_peptide_set(data_path: Path, seed: int, **kwargs) -> pd.DataFrame: # noqa
    logger.info(f"[make_shuffled_peptide_set] Loading and shuffling with seed={seed}")
    peptide_set = make_whole_peptide_set(data_path)
    logger.info(f"[make_shuffled_peptide_set] Original peptides (first 3):\n{peptide_set['peptide'].head(3).tolist()}")
    
    peptide_set['peptide'] = peptide_set['peptide'].map(lambda peptide : ''.join(random.sample(list(peptide), len(peptide))))
    logger.info(f"[make_shuffled_peptide_set] Shuffled peptides (first 3):\n{peptide_set['peptide'].head(3).tolist()}")
    log_dataframe(peptide_set, "make_shuffled_peptide_set", "OUTPUT")
    return peptide_set

# Shift point of aggregation
def get_aggregation_label_wof(
    subset: pd.DataFrame, start: int, end: int, drop: bool = False
):
    logger.debug(f"[get_aggregation_label_wof] Processing subset with {len(subset)} rows, start={start}, end={end}, drop={drop}")
    label = np.zeros(len(subset))
    
    # Find first aggregation point, or None if no aggregation exists
    agg_indices = np.argwhere(subset["aggregation"]).flatten()
    agg_point = agg_indices[0] if len(agg_indices) > 0 else None

    if agg_point is None:
        # No aggregation in this synthesis — keep all zeros (negative class)
        subset["aggregation"] = label.astype(int)
        logger.debug(f"[get_aggregation_label_wof] No aggregation points found — returning all False")
        return subset

    logger.debug(f"[get_aggregation_label_wof] Aggregation point at index {agg_point}")
    label[max(0, int(agg_point - start)) : min(int(agg_point + end), len(subset))] = 1
    subset["aggregation"] = label.astype(bool)

    if drop:
        subset = subset.iloc[: min(int(agg_point + end), len(subset))]
        logger.debug(f"[get_aggregation_label_wof] Dropped to {len(subset)} rows")

    columns = ["peptide", "serial", "aggregation"]
    if "first_diff_clean" in subset:
        columns.append("first_diff_clean")
    columns += [col for col in METADATA_COLUMNS if col in subset.columns]
    return subset[columns]

def make_wof_peptide_set(
    data_path: Path, wof_start: int, wof_end: int, wof_drop: bool = False, **kwargs # noqa
) -> pd.DataFrame:
    logger.info(f"[make_wof_peptide_set] Creating WOF set with start={wof_start}, end={wof_end}, drop={wof_drop}")
    data = make_reaction_set(data_path)
    logger.info(f"[make_wof_peptide_set] Loaded {data['serial'].nunique()} serials")
    
    chunks = list()
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]
        chunks.append(get_aggregation_label_wof(subset.copy(), wof_start, wof_end, wof_drop))

    result = pd.concat(chunks)
    log_dataframe(result, "make_wof_peptide_set", "OUTPUT")
    return result


# Aggregation Point: Drop amino acids after aggregation point
def make_agg_point_peptide_set(data_path: Path, **kwargs) -> pd.DataFrame: # noqa
    logger.info("[make_agg_point_peptide_set] Creating aggregation point set")
    data = make_reaction_set(data_path)
    logger.info(f"[make_agg_point_peptide_set] Initial: {len(data)} rows from {data['serial'].nunique()} serials")

    subset_indices = list()
    skipped_count = 0
    for serial in data["serial"].unique():
        subset = data[data["serial"] == serial]

        if subset['aggregation'].sum() == 0:
            subset_indices.append(subset.iloc[-1].name)
            continue
        
        agg_point = min(np.argwhere(subset['aggregation']).flatten())
        if agg_point <= 5:
            skipped_count += 1
            continue

        subset_indices.append(subset.iloc[agg_point].name)

    logger.info(f"[make_agg_point_peptide_set] Kept {len(subset_indices)} indices, skipped {skipped_count} (agg_point <= 5)")
    data = data.loc[subset_indices]
    log_dataframe(data, "make_agg_point_peptide_set", "OUTPUT")
    return data


        
