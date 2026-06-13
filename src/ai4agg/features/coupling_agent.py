import numpy as np
import pandas as pd

from .base import BaseFeature


class CouplingAgentFeature(BaseFeature):
    """One-hot encoding of the synthesis coupling agent (e.g. HATU, PyAOP).

    If the dataset has no "coupling_agent" column (or it is entirely NaN),
    transform() returns a single NaN feature instead of failing.
    """

    name = "coupling_agent"

    def __init__(self, max_sequence_len: int, padding: bool = True, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        self.categories: list[str] = []

    def fit(self, data: pd.DataFrame) -> None:
        if "coupling_agent" in data.columns and data["coupling_agent"].notna().any():
            self.categories = sorted(data["coupling_agent"].dropna().unique().tolist())

    def transform(self, row: pd.Series) -> np.ndarray:
        if not self.categories:
            return np.array([np.nan])

        vector = np.zeros(len(self.categories))
        value = row.get("coupling_agent", np.nan)
        if pd.isna(value):
            vector[:] = np.nan
        elif value in self.categories:
            vector[self.categories.index(value)] = 1.0
        return vector

    def feature_names(self) -> list[str]:
        if not self.categories:
            return [f"{self.name}_unknown"]
        return [f"{self.name}_{category}" for category in self.categories]
