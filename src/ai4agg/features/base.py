import numpy as np
import pandas as pd


class BaseFeature:
    """Interface every feature in src/ai4agg/features/ must implement.

    A feature turns one dataset row (or, for simple sequence-only features,
    a peptide string wrapped into a pd.Series with a "peptide" key) into a
    fixed-length numeric vector. ``CompositePreprocessor`` concatenates the
    outputs of a list of features.
    """

    name: str = "base"

    def __init__(self, max_sequence_len: int, padding: bool = True, **kwargs) -> None:  # noqa
        self.max_sequence_len = max_sequence_len
        self.padding = padding

    def fit(self, data: pd.DataFrame) -> None:
        """Prepare any lookup tables / encoders needed by transform()."""

    def transform(self, row: pd.Series) -> np.ndarray:
        raise NotImplementedError

    def feature_names(self) -> list[str]:
        raise NotImplementedError
