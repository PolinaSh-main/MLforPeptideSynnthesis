import numpy as np
import pandas as pd

from .base import BaseFeature


class CategoricalMetadataFeature(BaseFeature):
    """One-hot encoding of a single scalar metadata column (e.g. coupling_agent, machine).

    If `column` is missing from the dataset (or entirely NaN), transform()
    returns a single NaN feature instead of failing.
    """

    column: str = "base_column"

    def __init__(self, max_sequence_len: int, padding: bool = True, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        self.name = self.column
        self.categories: list = []

    def fit(self, data: pd.DataFrame) -> None:
        if self.column in data.columns and data[self.column].notna().any():
            self.categories = sorted(data[self.column].dropna().unique().tolist())

    def transform(self, row: pd.Series) -> np.ndarray:
        if not self.categories:
            return np.array([np.nan])

        # Unknown/missing category -> all-zero vector (graceful degradation,
        # e.g. for datasets that don't record this metadata at all).
        vector = np.zeros(len(self.categories))
        value = row.get(self.column, np.nan)
        if not pd.isna(value) and value in self.categories:
            vector[self.categories.index(value)] = 1.0
        return vector

    def feature_names(self) -> list[str]:
        if not self.categories:
            return [f"{self.name}_unknown"]
        return [f"{self.name}_{category}" for category in self.categories]
