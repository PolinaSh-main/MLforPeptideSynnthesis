# AI4Aggregation

This repo contains code accompanying the publication [Amino Acid Composition drives Peptide Aggregation: Predicting Aggregation for Improved Synthesis](https://chemrxiv.org/engage/chemrxiv/article-details/67a9af9ffa469535b9b67865), including scripts to reproduce all results.

<p align='center'>
  <img src='figures/GraphicalAbstract.png' width="1000px">
</p>


## Installation

The project was developed and tested on Python 3.10 and utilises poetry as package manager. To install the package run the commands provided below. The installation is expected to take less than five minutes.  

```console
pip install poetry
poetry install
```

## Preprocessing data

To create the dataset used to train models, use the following script:

```console
poetry run create_combined_data \
    --dataset <Path to UZH dataset> src/ai4agg/resources/dataset_configs/uzh.json \
    --dataset <Path to MIT dataset> src/ai4agg/resources/dataset_configs/mit.json \
    --save_path data/combined_data.csv
```

Each `--dataset` flag takes a pair of paths: the raw CSV and a JSON config describing how to
read it (column names, synthesis constants, protecting-group scheme). You can pass any number
of `--dataset` pairs to combine multiple sources. For the UZH dataset download the file
`uzh_data_clean.csv` from the [Zenodo record corresponding (version 1)](https://zenodo.org/records/14824562)
to this publication. The MIT dataset can be found in the corresponding GitHub repo
[here (last accessed 10.02.2025)](https://github.com/learningmatter-mit/peptimizer/blob/master/dataset/data_synthesis/synthesis_data.csv).

The old `--uzh_dataset_path` / `--mit_dataset_path` flags still work as deprecated aliases for
the commands above using the bundled `uzh.json` / `mit.json` configs.

By default, syntheses are truncated to the first `--max_length` (default 20) amino acids. Pass
`--no-truncate` to keep full-length peptides regardless of `--max_length`.

### Adding a new dataset (no code changes needed)

1. Create a JSON config in `src/ai4agg/resources/dataset_configs/`, e.g. `new_lab.json`:

   ```json
   {
     "name": "new_lab",
     "prefix": "newlab_",
     "column_map": {
       "serial": "serial",
       "amino_acid": "amino_acid",
       "pre_chain": "pre-chain",
       "first_diff": "first_diff"
     },
     "constants": {
       "coupling_agent": "HATU",
       "solvent": "DMF",
       "resin": "Rink"
     },
     "pg_scheme": "default_uzh_mit",
     "serial_dtype": "str",
     "index_col": null
   }
   ```

   - `column_map` maps the standard names (`serial`, `amino_acid`, `pre_chain`, `first_diff`
     are required; `coupling_agent`, `temp_coupling`, `solvent`, `resin` are optional) to the
     actual column names in your CSV.
   - `constants` fills in columns that aren't in your CSV with a fixed value for every row.
   - `pg_scheme` must be a key in `src/ai4agg/resources/pg_scheme.json` describing which
     protecting groups were used for this dataset.
   - `serial_dtype` is `"int"` or `"str"` (the type of the raw serial column before the prefix
     is added), `index_col` is passed straight to `pd.read_csv` (`null` or `0`).

2. Run:

   ```console
   poetry run create_combined_data \
       --dataset path/to/new_data.csv src/ai4agg/resources/dataset_configs/new_lab.json \
       --save_path data/combined_data.csv
   ```

That's it — no Python code needs to change. If a required column is missing or `pg_scheme` is
unknown, `create_combined_data` will fail with a clear error message naming the problem.

## Training models

### Classical Machine learning models

The following script allows training of classical machine learning models on the peptide aggregation data. This includes models ranging from a Random Forest to time series classification models:

```console
poetry run train_sklearn_model 
    --data_path: Path to UZH/MIT combined dataset
    --output_path: Path where the result of the training are going to be saved
    --loader: Data loader to represent the data. Choose from: [reaction_set (stepwise representation), whole_set (whole peptide) and whole_set_shuffled]
    --preprocessor: Preprocessor for the data. Choose from: [sequence, one_hot, fingerprint, protected_fingerprint, positional_fingerprint, whole_peptide_fingerprint, protected_whole_peptide_fingerprint, occurency, hydrophobicity, protected_fp_hydrophob]
    --model: Model to train. Choose from: [rff (Random Forest), xgb (XGBoost), knn (K-Nearest Neighbour), gaussian (Gaussian Processes), hc2 (Hive Cote 2), timeforest (Time series forest), weasel (WEASEL)]
    --pg_scheme: Name of the protecting-group scheme (key in src/ai4agg/resources/pg_scheme.json),
                 used by the 'hydrophobicity' and 'protected_fp_hydrophob' preprocessors. Default: default_uzh_mit
```

As an example run the following. The folder `results/xgb` needs to exist:

```console
poetry run train_sklearn_model --data_path data/combined_data.csv --output_path results/xgb  --loader whole_set --preprocessor occurency --model xgb
```

The script takes around a minute to run and the expected accuracy of the model is `0.596±0.019`.

### Training HuggingFace models

To train HuggingFace models on the aggregation data use the following script. A GPU is recommended to train the models but at the expense of time the models can also be run locally:

```console
poetry run train_hf_model 
    --data_path: Path to UZH/MIT combined dataset
    --output_path: Path where the result of the training are going to be saved
    --model: Model to train e.g. facebook/esm2_t33_650M_UR50D. We evaluated ESM 2.0 and BERT.
    --pretrained: Wether to train the model from scratch or use the pretrained one from HuggingFace. [True False]
```


## Reproducing Training results

We provide a set of scripts to replicate the results obtained in the paper. All scripts require a path to where the results of the experiments are saved. A GPU is recommended to reproduce the Transformer based results:
```console
bash scripts/run_hf_models.sh <Path to Experiment Folder>
bash scripts/run_sklearn_models.sh <Path to Experiment Folder>
bash scripts/run_sklearn_shuffled.sh <Path to Experiment Folder>
bash scripts/run_wof_sklearn.sh <Path to Experiment Folder>
```

## Explainability

To explain the predictions of the models we use Shap values. To reproduce our results use the following scripts: 

```console
poetry run explain_model --data_path data/combined_data.csv --output_path <Path where results should be stored>
```

`explain_model` also accepts `--preprocessor` (same choices as `train_sklearn_model`) and
`--pg_scheme`. With `--preprocessor hydrophobicity` or `protected_fp_hydrophob`, the SHAP
ranking CSV uses readable names like `pos0_logP`, `pos1_logP`, ... so you can directly compare
which positions/features drive the model versus the `occurency` baseline.

## Feature engineering: adding a new feature

All "synthesis-aware" features (protecting groups, hydrophobicity, coupling agent, ...) live in
`src/ai4agg/features/`. Adding a new feature requires **one new file and one line in a
registry** — no other code needs to change.

### 1. Protecting groups (`pg_scheme.json`)

`src/ai4agg/resources/pg_scheme.json` is the single place that defines which protecting group
(PG) is used for each amino acid in a given experiment. Each top-level key is a named scheme:

```json
{
  "default_uzh_mit": {
    "description": "Standard Fmoc-SPPS, both UZH and MIT datasets",
    "solvent": "DMF",
    "resin": "Rink",
    "coupling_agent": "HATU",
    "pg_map": { "G": "none", "C": "Trt", "R": "Pbf", "...": "..." }
  }
}
```

Select a scheme at training/explanation time with `--pg_scheme <name>`. Any feature that needs
to know which PG is attached to a residue (e.g. `HydrophobicityFeature`) reads `pg_map` from
this file — you never hardcode PGs in feature code.

### 2. Writing a new feature

Create `src/ai4agg/features/my_feature.py` implementing the `BaseFeature` interface
(`src/ai4agg/features/base.py`):

```python
import numpy as np
import pandas as pd
from .base import BaseFeature


class MyFeature(BaseFeature):
    name = "my_feature"

    def fit(self, data: pd.DataFrame) -> None:
        # Build any lookup tables / encoders here, e.g. from `data` columns.
        ...

    def transform(self, row: pd.Series) -> np.ndarray:
        # `row` has at least a "peptide" column, plus any metadata columns
        # (coupling_agent, solvent, resin, temp_coupling, pg_scheme) if present.
        ...
        return np.array([...])

    def feature_names(self) -> list[str]:
        return [f"pos{i}_{self.name}" for i in range(self.max_sequence_len)]
```

If your CSV doesn't have the column you need, return `np.nan` values instead of raising —
downstream code expects features to degrade gracefully, not crash.

### 3. Registering the feature

Add it to `CompositePreprocessor` registry entries in
`src/ai4agg/cli/train_sklearn_models.py` and `src/ai4agg/cli/explainability.py`:

```python
from ..features import MyFeature

PREPROCESSOR_REGISTRY = {
    ...,
    "protected_fp_hydrophob_myfeature": partial(
        CompositePreprocessor,
        [ProtectedFingerprintFeature, HydrophobicityFeature, MyFeature],
    ),
}
```

Then run, e.g.:

```console
poetry run train_sklearn_model --preprocessor protected_fp_hydrophob_myfeature --model xgb ...
poetry run explain_model --preprocessor protected_fp_hydrophob_myfeature --model xgb ...
```

The SHAP ranking will automatically include your feature's names from `feature_names()`.

### Existing building-block features

- `ProtectedFingerprintFeature` — per-residue Morgan fingerprint of the protected (Fmoc-AA(PG)-OH) amino acid.
- `HydrophobicityFeature` — per-residue PG-corrected Consensus logP (`pos{i}_logP`). Falls back to the
  table-wide mean if a given (AA, PG) combination isn't in `hydrophobicity_clean.csv`.
- `CouplingAgentFeature` — one-hot encoding of the `coupling_agent` metadata column (NaN if absent).
- `PositionalWeightFeature` — wraps another feature and multiplies each residue's block by a
  position-from-C-terminus weight, e.g. `PositionalWeightFeature(max_sequence_len, wrapped_feature=HydrophobicityFeature)`.
