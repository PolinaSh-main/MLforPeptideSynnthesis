import numpy as np
import pandas as pd

from .base import BaseFeature


class PositionalWeightFeature(BaseFeature):
    """Wraps another per-residue feature and weights each residue's block.

    weight = (position_from_C_terminus + 1) / peptide_length
    Position 0 (N-terminus, last added in SPPS) -> lowest weight.
    Last position (C-terminus, first added) -> weight = 1.0.
    Padding blocks get weight 0.

    Usage: PositionalWeightFeature(max_sequence_len, wrapped_feature=HydrophobicityFeature, ...)
    """

    name = "positional"

    def __init__(self, max_sequence_len: int, padding: bool = True, wrapped_feature: type[BaseFeature] | None = None, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        if wrapped_feature is None:
            raise ValueError("PositionalWeightFeature requires a `wrapped_feature` BaseFeature class")

        self.feature = wrapped_feature(max_sequence_len, padding, **kwargs)
        self.name = f"positional_{self.feature.name}"

    def fit(self, data: pd.DataFrame) -> None:
        self.feature.fit(data)

    def transform(self, row: pd.Series) -> np.ndarray:
        vector = self.feature.transform(row).astype(float)
        peptide_length = len(row["peptide"])
        block_size = len(vector) // self.max_sequence_len

        weighted = vector.copy()
        for position in range(self.max_sequence_len):
            start, end = position * block_size, (position + 1) * block_size
            if position < peptide_length:
                position_from_c = peptide_length - 1 - position
                weight = (position_from_c + 1) / peptide_length
            else:
                weight = 0.0
            weighted[start:end] *= weight

        return weighted

    def feature_names(self) -> list[str]:
        return [f"weighted_{name}" for name in self.feature.feature_names()]
