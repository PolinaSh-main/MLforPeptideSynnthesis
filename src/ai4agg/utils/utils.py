import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from importlib.resources import files
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold, train_test_split

AA_TO_SMILES_PATH = files("ai4agg") / "resources/onelet_to_smiles.csv"
PROTECTED_AA_TO_SMILES_PATH = files("ai4agg") / "resources/protected_onelet_to_smiles.csv"
HYDROPHOBICITY_PATH = files("ai4agg") / "resources/hydrophobicity_clean.csv"
PG_SCHEME_PATH = files("ai4agg") / "resources/pg_scheme.json"


def load_pg_scheme(scheme_name: str, pg_scheme_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Load a protecting-group scheme (abbrev -> pg) by name from pg_scheme.json."""
    with open(pg_scheme_path or PG_SCHEME_PATH) as f:
        schemes = json.load(f)

    if scheme_name not in schemes:
        raise ValueError(
            f"Unknown pg_scheme '{scheme_name}'. Available schemes: {list(schemes.keys())}"
        )

    return schemes[scheme_name]["pg_map"]


def seed_everything(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def split_peptide_set(dataset: pd.DataFrame, val: bool = False, cv_split: int = 0, seed: int = 3245) -> Dict[str, pd.DataFrame]:
    
    serials = dataset['serial'].unique()

    kfold = KFold(shuffle=True, random_state=seed)
    train_indices, test_indices = list(kfold.split(serials))[cv_split]
    train_serials, test_serials = serials[train_indices], serials[test_indices]

    train_set, test_set = dataset[dataset['serial'].isin(train_serials)], dataset[dataset['serial'].isin(test_serials)]

    dataset_dict = {'train': train_set, 'test': test_set}

    if val:
        train_set, val_set = train_test_split(dataset_dict['train'], test_size=0.1, random_state=seed)
        dataset_dict['train'] = train_set
        dataset_dict['val'] = val_set

    return dataset_dict



class FingerPrintCalculator:

    def __init__(
        self,
        n_bits: int,
        smiles_path: Optional[Union[str, Path]] = None,
        smiles_column: Optional[str] = None,
    ) -> None:

        df = pd.read_csv(smiles_path or AA_TO_SMILES_PATH)
        df = df.replace(np.nan, "", regex=True)
        if smiles_column is not None:
            df["AASMILES"] = df[smiles_column]
        else:
            df["AASMILES"] = df["left"] + df["sidechain"] + df["right"]
        onelet_smiles_dict = dict(zip(df.abbrev, df.AASMILES))
        
        self.onelet_smiles_dict = onelet_smiles_dict
        self.n_bits = n_bits
        self.fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=n_bits)


    def smilifier(self, sequence: str) -> str:
        sequence_list = [*sequence][::-1]
        sequence_smiles_list = [
            self.onelet_smiles_dict.get(item, item) for item in sequence_list
        ]
        sequence_smiles_list += ["N"]
        sequence_smiles = "".join(sequence_smiles_list)
        return sequence_smiles

    def morgan_fingerprint(self, amino_acid: str) -> List[float]:
        smile = self.smilifier(amino_acid)
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Could not build molecule from SMILES {smile!r} for {amino_acid!r}.")
        fp = self.fingerprint.GetFingerprintAsNumPy(mol)
        return list(fp)

    def morgan_fingerprint_from_smiles(self, smiles: str) -> List[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not build molecule from SMILES {smiles!r}.")
        fp = self.fingerprint.GetFingerprintAsNumPy(mol)
        return list(fp)



