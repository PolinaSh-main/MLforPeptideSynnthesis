import numpy as np
import pandas as pd

from .base import BaseFeature


class NumericMetadataFeature(BaseFeature):
    """A single scalar numeric metadata value (e.g. temp_coupling).

    Falls back to the dataset-wide mean if `column` is missing/NaN for a row
    (or for the whole dataset), so this feature degrades gracefully instead
    of producing NaN inputs for models that can't handle them.
    """

    column: str = "base_column"

    def __init__(self, max_sequence_len: int, padding: bool = True, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        self.name = self.column
        self.mean_value = 0.0

    def fit(self, data: pd.DataFrame) -> None:
        if self.column in data.columns:
            values = pd.to_numeric(data[self.column], errors="coerce").dropna()
            if len(values) > 0:
                self.mean_value = float(values.mean())

    def transform(self, row: pd.Series) -> np.ndarray:
        value = row.get(self.column, np.nan)
        if pd.isna(value):
            value = self.mean_value
        return np.array([value], dtype=float)

    def feature_names(self) -> list[str]:
        return [self.name]
