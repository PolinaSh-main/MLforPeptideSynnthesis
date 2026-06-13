import numpy as np
import pandas as pd

from .base import BaseFeature
from .categorical_metadata import CategoricalMetadataFeature


class SynthesisStepFeature(BaseFeature):
    """Per-residue synthesis-machine readout (e.g. coupling_strokes, flow_rate, first_area, ...).

    Reads `row[column]`, a per-residue sequence aligned with the peptide's
    amino acid addition order, built by `make_whole_peptide_set()`. Missing
    values (e.g. for datasets without this column, like UZH, or a row with no
    data at all) fall back to the dataset-wide mean; padding positions beyond
    the peptide length are filled with 0.0.
    """

    column: str = "base_step"

    def __init__(self, max_sequence_len: int, padding: bool = True, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        self.name = self.column
        self.mean_value = 0.0

    def fit(self, data: pd.DataFrame) -> None:
        if self.column in data.columns:
            all_values: list = []
            for arr in data[self.column]:
                if isinstance(arr, (list, np.ndarray, pd.Series)):
                    all_values.extend(arr)
            values = pd.to_numeric(pd.Series(all_values), errors="coerce").dropna()
            if len(values) > 0:
                self.mean_value = float(values.mean())

    def transform(self, row: pd.Series) -> np.ndarray:
        values = row.get(self.column, np.nan)
        if not isinstance(values, (list, np.ndarray, pd.Series)):
            values = []
        values = list(values)[: self.max_sequence_len]
        n_real = len(values)

        if self.padding:
            values = values + [np.nan] * max(self.max_sequence_len - n_real, 0)

        result = [
            self.mean_value if (i < n_real and pd.isna(v)) else (0.0 if pd.isna(v) else float(v))
            for i, v in enumerate(values)
        ]
        return np.array(result, dtype=float)

    def feature_names(self) -> list[str]:
        return [f"pos{pos}_{self.name}" for pos in range(self.max_sequence_len)]


class CouplingStrokesFeature(SynthesisStepFeature):
    column = "coupling_strokes"


class DeprotectionStrokesFeature(SynthesisStepFeature):
    column = "deprotection_strokes"


class FlowRateFeature(SynthesisStepFeature):
    column = "flow_rate"


class TempReactorFeature(SynthesisStepFeature):
    column = "temp_reactor_1"


class FirstAreaFeature(SynthesisStepFeature):
    column = "first_area"


class FirstHeightFeature(SynthesisStepFeature):
    column = "first_height"


class FirstWidthFeature(SynthesisStepFeature):
    column = "first_width"


class PrevAreaFeature(SynthesisStepFeature):
    column = "prev_area"


class PrevHeightFeature(SynthesisStepFeature):
    column = "prev_height"


class PrevWidthFeature(SynthesisStepFeature):
    column = "prev_width"


class PrevDiffFeature(SynthesisStepFeature):
    column = "prev_diff"


class MachineFeature(CategoricalMetadataFeature):
    """One-hot encoding of the synthesis machine."""

    name = "machine"
    column = "machine"
