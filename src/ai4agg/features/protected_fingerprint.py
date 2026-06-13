import numpy as np
import pandas as pd

from ..utils.utils import FingerPrintCalculator, PROTECTED_AA_TO_SMILES_PATH
from .base import BaseFeature


class ProtectedFingerprintFeature(BaseFeature):
    """Per-residue Morgan fingerprints using Fmoc-AA(PG)-OH SMILES."""

    name = "protected_fp"

    def __init__(self, max_sequence_len: int, padding: bool = True, n_fingerprint_bits: int = 128, **kwargs) -> None:  # noqa
        super().__init__(max_sequence_len, padding, **kwargs)
        self.n_fingerprint_bits = n_fingerprint_bits
        self.fingerprint_calculator = FingerPrintCalculator(
            n_fingerprint_bits,
            smiles_path=PROTECTED_AA_TO_SMILES_PATH,
            smiles_column="smiles",
        )

    def transform(self, row: pd.Series) -> np.ndarray:
        peptide = row["peptide"]
        fingerprint_sequence = []
        for aa in peptide:
            smiles = self.fingerprint_calculator.onelet_smiles_dict[aa]
            fingerprint_sequence.append(
                self.fingerprint_calculator.morgan_fingerprint_from_smiles(smiles)
            )

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            fingerprint_sequence.append(np.zeros(n_pad * self.n_fingerprint_bits))

        return np.concatenate(fingerprint_sequence, axis=0).flatten()

    def feature_names(self) -> list[str]:
        names = []
        for pos in range(self.max_sequence_len):
            for bit in range(self.n_fingerprint_bits):
                names.append(f"pos{pos}_{self.name}_bit{bit}")
        return names
