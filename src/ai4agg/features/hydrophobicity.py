import numpy as np
import pandas as pd

from ..utils.utils import HYDROPHOBICITY_PATH, load_pg_scheme
from .base import BaseFeature


class HydrophobicityFeature(BaseFeature):
    """Per-residue hydrophobicity (Consensus logP, corrected for outliers).

    Looks up logP for each residue based on (one-letter code, protecting group),
    where the protecting group is taken from a pg_scheme (abbrev -> pg).
    If a given (AA, PG) combination is missing from the table, falls back to
    the mean logP over the whole table rather than failing.
    """

    name = "logP"

    def __init__(
        self,
        max_sequence_len: int,
        padding: bool = True,
        pg_scheme: str | dict | None = "default_uzh_mit",
        **kwargs,
    ) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)

        if isinstance(pg_scheme, dict):
            self.pg_map = pg_scheme
        else:
            self.pg_map = load_pg_scheme(pg_scheme or "default_uzh_mit")

        table = pd.read_csv(HYDROPHOBICITY_PATH)
        table = table.dropna(subset=["AA (1-code)"])

        self.logp_dict = dict(
            zip(
                zip(table["AA (1-code)"], table["side chain PG"]),
                table["Consensus_logP_corrected"],
            )
        )
        self.mean_logp = float(table["Consensus_logP_corrected"].mean())

    def transform(self, row: pd.Series) -> np.ndarray:
        peptide = row["peptide"]
        values = []
        for aa in peptide:
            pg = self.pg_map.get(aa, "none")
            value = self.logp_dict.get((aa, pg), np.nan)
            if np.isnan(value):
                value = self.mean_logp
            values.append(value)

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            values.extend([0.0] * n_pad)

        return np.array(values, dtype=float)

    def feature_names(self) -> list[str]:
        return [f"pos{pos}_{self.name}" for pos in range(self.max_sequence_len)]
