import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .utils import FingerPrintCalculator, PROTECTED_AA_TO_SMILES_PATH

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CorePreprocessor:

    def __init__(self, data: pd.DataFrame, padding: bool = True, random_state: int = 3245, **kwargs) -> None:  # noqa
        self.padding = padding
        self.random_state = random_state
        self.max_sequence_len = max(data['peptide'].map(len))

    def __call__(self, peptide: str) -> np.ndarray:  # noqa
        raise NotImplementedError


class SequencePreprocessor(CorePreprocessor):

    def __call__(self, peptide: str) -> np.ndarray:
        processed_peptide = np.zeros(self.max_sequence_len if self.padding else len(peptide))
        for i, aa in enumerate(peptide):
            processed_peptide[i] = ord(aa)
        return processed_peptide


class OneHotPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, padding: bool = True, random_state: int = 3245, **kwargs) -> None:  # noqa
        super().__init__(data, padding, random_state)

        aa_frequencies = []
        for peptide in data["peptide"].to_list():
            amino_acids = [[aa] for aa in peptide]
            aa_frequencies.extend(amino_acids + [["pad"]])

        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(aa_frequencies)
        self.one_hot_encoder = encoder

    def __call__(self, peptide: str) -> np.ndarray:
        return self.one_hot_peptide(peptide)

    def one_hot_peptide(self, peptide: str) -> np.ndarray:
        one_hot_sequence = []
        for aa in peptide:
            one_hot_aa = self.one_hot_encoder.transform([[aa]]).reshape(1, -1)
            one_hot_sequence.append(one_hot_aa)

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            pad_vector = self.one_hot_encoder.transform([["pad"]])
            shaped_pad_vector = np.repeat(pad_vector, n_pad, 0)
            one_hot_sequence.append(shaped_pad_vector)

        return np.concatenate(one_hot_sequence, axis=0).flatten()


@dataclass
class FingerprintPreprocessor(CorePreprocessor):

    def __init__(self, data: pd.DataFrame, padding: bool = True, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs):  # noqa
        super().__init__(data, padding, random_state)
        self.n_fingerprint_bits = n_fingerprint_bits
        self.fingerprint_calculator = FingerPrintCalculator(n_fingerprint_bits)

    def __call__(self, peptide: str) -> np.ndarray:
        fingerprint_sequence = []
        for aa in peptide:
            fingerprint_sequence.append(self.fingerprint_calculator.morgan_fingerprint(aa))

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            fingerprint_sequence.append(np.zeros(n_pad * self.n_fingerprint_bits))  # type: ignore

        return np.concatenate(fingerprint_sequence, axis=0).flatten()


class ProtectedFingerprintPreprocessor(FingerprintPreprocessor):
    """Per-residue Morgan fingerprints using Fmoc-AA(PG)-OH SMILES."""

    def __init__(self, data: pd.DataFrame, padding: bool = True, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs):  # noqa
        super().__init__(data, padding, n_fingerprint_bits, random_state)
        self.fingerprint_calculator = FingerPrintCalculator(
            n_fingerprint_bits,
            smiles_path=PROTECTED_AA_TO_SMILES_PATH,
            smiles_column="smiles",
        )

    def __call__(self, peptide: str) -> np.ndarray:
        fingerprint_sequence = []
        for aa in peptide:
            smiles = self.fingerprint_calculator.onelet_smiles_dict[aa]
            fingerprint_sequence.append(
                self.fingerprint_calculator.morgan_fingerprint_from_smiles(smiles)
            )

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            fingerprint_sequence.append(np.zeros(n_pad * self.n_fingerprint_bits))  # type: ignore

        return np.concatenate(fingerprint_sequence, axis=0).flatten()


class PositionalFingerprintPreprocessor(FingerprintPreprocessor):
    """Per-residue Morgan fingerprints weighted by position from C-terminus.

    weight = (position_from_C_terminus + 1) / peptide_length
    Position 0 = N-terminus (last added in SPPS) → weight = 1/L (lowest)
    Position L-1 = C-terminus (first added) → weight = 1.0 (highest)
    """

    def __call__(self, peptide: str) -> np.ndarray:
        fingerprint_sequence = []
        peptide_length = len(peptide)
        for position, aa in enumerate(peptide):
            # position 0 = N-terminus in the sequence string
            # C-terminus position index = peptide_length - 1 - position
            position_from_c = peptide_length - 1 - position
            weight = (position_from_c + 1) / peptide_length
            fp = np.array(self.fingerprint_calculator.morgan_fingerprint(aa))
            fingerprint_sequence.append(fp * weight)

        if self.padding:
            n_pad = self.max_sequence_len - len(peptide)
            fingerprint_sequence.append(np.zeros(n_pad * self.n_fingerprint_bits))  # type: ignore

        return np.concatenate(fingerprint_sequence, axis=0).flatten()


class WholePeptideFingerprintPreprocessor(CorePreprocessor):
    """Single Morgan fingerprint for the entire peptide SMILES (native AA).

    Uses smilifier() to build the full peptide SMILES from native AA SMILES,
    then computes one Morgan fingerprint for the whole molecule.
    Output shape: (n_fingerprint_bits,) — fixed regardless of peptide length.
    """

    def __init__(self, data: pd.DataFrame, padding: bool = True, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs):  # noqa
        super().__init__(data, False, random_state)  # no padding — fixed-size output
        self.n_fingerprint_bits = n_fingerprint_bits
        self.fingerprint_calculator = FingerPrintCalculator(n_fingerprint_bits)

    def __call__(self, peptide: str) -> np.ndarray:
        peptide_smiles = self.fingerprint_calculator.smilifier(peptide)
        return np.array(
            self.fingerprint_calculator.morgan_fingerprint_from_smiles(peptide_smiles)
        )


class ProtectedWholePeptideFingerprintPreprocessor(CorePreprocessor):
    """Single Morgan fingerprint for the entire peptide built from Fmoc-AA(PG)-OH SMILES.

    Builds a pseudo-whole-peptide SMILES by concatenating protected residue SMILES
    (C-terminus to N-terminus, same order as smilifier), then computes one Morgan
    fingerprint for the whole molecule.
    Output shape: (n_fingerprint_bits,) — fixed regardless of peptide length.

    Note: The concatenated protected SMILES is a heuristic representation of the
    on-resin intermediate. It does not model full amide bond formation between
    protected residues, but captures the chemical environment of protecting groups
    which is the main signal added vs. native fingerprints.
    """

    def __init__(self, data: pd.DataFrame, padding: bool = True, n_fingerprint_bits: int = 128, random_state: int = 3245, **kwargs):  # noqa
        super().__init__(data, False, random_state)
        self.n_fingerprint_bits = n_fingerprint_bits
        self.fingerprint_calculator = FingerPrintCalculator(
            n_fingerprint_bits,
            smiles_path=PROTECTED_AA_TO_SMILES_PATH,
            smiles_column="smiles",
        )

    def __call__(self, peptide: str) -> np.ndarray:
        # Build concatenated protected SMILES C→N (same direction as smilifier)
        sequence_reversed = list(peptide)[::-1]
        smiles_parts = [
            self.fingerprint_calculator.onelet_smiles_dict[aa]
            for aa in sequence_reversed
        ]
        # Append free amine cap as in smilifier
        smiles_parts.append("N")
        concatenated_smiles = "".join(smiles_parts)
        return np.array(
            self.fingerprint_calculator.morgan_fingerprint_from_smiles(concatenated_smiles)
        )


@dataclass
class OccurencyVectorPreprocessor(CorePreprocessor):

    def __init__(
        self,
        data: pd.DataFrame,
        random_state: int = 3245,
        normalise: bool = True,
        occurency_vector_normalise: bool | None = None,
        **kwargs,
    ):  # noqa
        super().__init__(data, False, random_state)  # no padding needed

        all_aa = []
        for peptide in data['peptide']:
            all_aa.extend(list(peptide))
        self.all_aa = np.unique(all_aa)

        self.normalise = normalise if occurency_vector_normalise is None else occurency_vector_normalise

    def __call__(self, peptide: str) -> np.ndarray:
        occurency_vector = self.build_occurency_vector(peptide)
        if self.normalise:
            occurency_vector = occurency_vector / len(peptide)
        return occurency_vector

    def build_occurency_vector(self, peptide: str) -> np.ndarray:
        occurency_vector = np.zeros(len(self.all_aa))
        for i, aa in enumerate(self.all_aa):
            occurency_vector[i] = peptide.count(aa)
        return occurency_vector
