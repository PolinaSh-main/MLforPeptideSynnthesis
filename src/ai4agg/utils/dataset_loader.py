import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import PG_SCHEME_PATH
from .data_logger import log_dataframe

logger = logging.getLogger(__name__)

REQUIRED_COLUMN_MAP_KEYS = ["serial", "amino_acid", "pre_chain", "first_diff"]
METADATA_COLUMNS = ["coupling_agent", "solvent", "resin", "temp_coupling", "pg_scheme"]
STANDARD_COLUMNS = ["serial", "peptide", "amino_acid", "first_diff"] + METADATA_COLUMNS


def filter_synthesis(synthesis_data: pd.DataFrame, max_length: int = 20, truncate: bool = True) -> pd.DataFrame:
    logger.info(f"[filter_synthesis] Input: {len(synthesis_data)} rows, {synthesis_data['serial'].nunique()} serials")
    logger.info(f"[filter_synthesis] Max peptide length: {max_length}, truncate: {truncate}")

    metadata_cols = [c for c in METADATA_COLUMNS if c in synthesis_data.columns]
    keep_cols = ['serial', 'peptide', 'amino_acid', 'first_diff'] + metadata_cols

    subsets = list()
    filtered_reasons = {"invalid_aa": 0, "not_from_scratch": 0, "short_peptide": 0, "too_long": 0, "kept": 0}

    for serial in synthesis_data['serial'].unique():
        subset = synthesis_data[synthesis_data['serial'] == serial]

        # Filter out any invalid amino acids / pre-chains
        if len(subset[subset['amino_acid'].map(lambda aa: isinstance(aa, str))]) != len(subset) or len(subset[subset['pre-chain'].map(lambda pc: isinstance(pc, str))]) != len(subset):
            filtered_reasons["invalid_aa"] += 1
            continue

        # Filter out all synthesis not starting from scratch
        if len(subset) != (len(subset['peptide'].iloc[-1]) - 1):
            filtered_reasons["not_from_scratch"] += 1
            continue

        final_peptide = subset['peptide'].iloc[-1]

        # Filter out all short peptides
        if len(final_peptide) < 5:
            filtered_reasons["short_peptide"] += 1
            continue

        if truncate:
            # Filter out peptides that exceed max_length
            if len(final_peptide) > max_length:
                filtered_reasons["too_long"] += 1
                continue

            # Truncate to max_length steps (AA additions)
            subset = subset[:min(len(subset), max_length - 1)]

        subsets.append(subset[keep_cols])
        filtered_reasons["kept"] += 1

    result = pd.concat(subsets)
    logger.info(f"[filter_synthesis] Output: {len(result)} rows | Filters: {filtered_reasons}")
    return result


def load_dataset(csv_path: Path, config: Dict[str, Any], max_length: int = 20, truncate: bool = True) -> pd.DataFrame:
    """Load a synthesis dataset CSV using a dataset config dict (see resources/dataset_configs/*.json).

    Returns a DataFrame with the standard columns: serial, peptide, amino_acid,
    first_diff, coupling_agent, solvent, resin, temp_coupling, pg_scheme.
    Missing columns are filled with NaN.
    """
    column_map: Dict[str, str] = config.get("column_map", {})

    for key in REQUIRED_COLUMN_MAP_KEYS:
        if key not in column_map:
            raise ValueError(
                f"Dataset config '{config.get('name', csv_path)}' is missing required "
                f"column_map key: '{key}'"
            )

    df = pd.read_csv(csv_path, index_col=config.get("index_col"))

    for std_name, csv_col in column_map.items():
        if csv_col not in df.columns:
            raise ValueError(
                f"Column '{csv_col}' not found in CSV. Available columns: {df.columns.tolist()}"
            )

    rename_map = {
        csv_col: ("pre-chain" if std_name == "pre_chain" else std_name)
        for std_name, csv_col in column_map.items()
    }
    df = df.rename(columns=rename_map)

    df["peptide"] = df["pre-chain"] + df["amino_acid"]

    serial_dtype = int if config.get("serial_dtype", "str") == "int" else str
    prefix = config.get("prefix", "")
    df["serial"] = prefix + df["serial"].astype(serial_dtype).astype(str)

    for column, value in config.get("constants", {}).items():
        if column not in df.columns:
            df[column] = value

    pg_scheme_name = config.get("pg_scheme")
    if pg_scheme_name is not None:
        with open(PG_SCHEME_PATH) as f:
            schemes = json.load(f)
        if pg_scheme_name not in schemes:
            raise ValueError(
                f"Unknown pg_scheme '{pg_scheme_name}' in dataset config "
                f"'{config.get('name', csv_path)}'. Available schemes: {list(schemes.keys())}"
            )
        df["pg_scheme"] = pg_scheme_name

    df = filter_synthesis(df, max_length=max_length, truncate=truncate)

    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df = df[STANDARD_COLUMNS]
    log_dataframe(df, "load_dataset", f"OUTPUT[{config.get('name', csv_path)}]")
    return df
