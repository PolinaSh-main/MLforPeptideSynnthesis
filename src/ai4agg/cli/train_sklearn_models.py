import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import TimeSeriesForestClassifier
from xgboost import XGBClassifier, XGBRegressor

from ..utils.loaders import (
    make_reaction_set,
    make_shuffled_peptide_set,
    make_whole_peptide_set,
    make_wof_peptide_set,
)
from ..utils.preprocessors import (
    FingerprintPreprocessor,
    OccurencyVectorPreprocessor,
    OneHotPreprocessor,
    PositionalFingerprintPreprocessor,
    ProtectedFingerprintPreprocessor,
    SequencePreprocessor,
    WholePeptideFingerprintPreprocessor,
)
from ..utils.utils import seed_everything, split_peptide_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


LOADER_REGISTRY = {
    "reaction_set": make_reaction_set,
    "wof_set": make_wof_peptide_set,
    "whole_set": make_whole_peptide_set,
    "whole_set_shuffled": make_shuffled_peptide_set,
}

PREPROCESSOR_REGISTRY = {
    "sequence": SequencePreprocessor,
    "one_hot": OneHotPreprocessor,
    "fingerprint": FingerprintPreprocessor,
    "protected_fingerprint": ProtectedFingerprintPreprocessor,
    "positional_fingerprint": PositionalFingerprintPreprocessor,
    "whole_peptide_fingerprint": WholePeptideFingerprintPreprocessor,
    "occurency": OccurencyVectorPreprocessor,
}

MODEL_REGISTRY = {
    "rff": RandomForestClassifier,
    "xgb": XGBClassifier,
    "knn": KNeighborsClassifier,
    "gaussian": GaussianProcessClassifier,
    "hc2": partial(HIVECOTEV2, time_limit_in_minutes=10),
    "timeforest": TimeSeriesForestClassifier,
    "weasel": WEASEL,
}

REGRESSION_MODEL_REGISTRY = {
    "rff": RandomForestRegressor,
    "xgb": XGBRegressor,
}


def load_data(
    data_path: Path,
    loader: str,
    preprocessor: str,
    cv_split: int = 0,
    seed: int = 3245,
    **kwargs,
) -> Dict[str, pd.DataFrame]:

    dataset = LOADER_REGISTRY[loader](data_path, seed=seed, **kwargs)  # type: ignore

    preprocessor_instance = PREPROCESSOR_REGISTRY[preprocessor](
        dataset,
        random_state=seed,
        **kwargs,
    )
    dataset["input"] = dataset["peptide"].map(preprocessor_instance)

    if loader == "whole_set_shuffled":
        train_set, test_set = train_test_split(dataset, random_state=seed)
        dataset_dict = {"train": train_set, "test": test_set}
    else:
        dataset_dict = split_peptide_set(dataset, val=False, cv_split=cv_split, seed=seed)

    return dataset_dict


def train(dataset_dict: Dict[str, pd.DataFrame], model: str) -> Tuple[Any, Any, pd.DataFrame]:

    classifier = MODEL_REGISTRY[model]()
    classifier.fit(
        np.stack(dataset_dict["train"]["input"].to_list()),
        np.stack(dataset_dict["train"]["aggregation"].to_list()),
    )

    test_set = dataset_dict["test"].copy()
    test_set["prediction"] = classifier.predict(np.stack(test_set["input"].to_list()))
    if hasattr(classifier, "predict_proba"):
        test_set["prediction_probability"] = classifier.predict_proba(
            np.stack(test_set["input"].to_list())
        )[:, 1]

    ground_truth = list()
    predictions = list()
    for serial in test_set["serial"].unique():
        subset = test_set[test_set["serial"] == serial]

        ground_truth.append(np.any(subset["aggregation"]))
        predictions.append(np.any(subset["prediction"]))

    f1 = f1_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    return f1, accuracy, test_set


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def train_regression(
    dataset_dict: Dict[str, pd.DataFrame],
    model: str,
    target_column: str = "first_diff_clean",
) -> Tuple[float, float, pd.DataFrame]:

    if model not in REGRESSION_MODEL_REGISTRY:
        raise ValueError(
            f"Model {model} is not available for regression. "
            f"Use one of: {list(REGRESSION_MODEL_REGISTRY)}."
        )

    regressor = REGRESSION_MODEL_REGISTRY[model]()
    regressor.fit(
        np.stack(dataset_dict["train"]["input"].to_list()),
        np.stack(dataset_dict["train"][target_column].to_list()),
    )

    test_set = dataset_dict["test"].copy()
    test_set["prediction"] = regressor.predict(np.stack(test_set["input"].to_list()))

    y_true = test_set[target_column].to_numpy(dtype=float)
    y_pred = test_set["prediction"].to_numpy(dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    pearson = pearson_correlation(y_true, y_pred)
    return mse, pearson, test_set


def add_prediction_frame(
    prediction_frames: list[pd.DataFrame],
    test_predictions: pd.DataFrame,
    split: int,
) -> None:
    test_predictions = test_predictions.copy()
    test_predictions["split"] = split
    prediction_frames.append(test_predictions)


def save_predictions(
    prediction_frames: list[pd.DataFrame],
    output_path: Path,
    target_column: str,
) -> None:
    if not prediction_frames:
        return

    all_predictions = pd.concat(prediction_frames)
    prediction_columns = [
        column for column in [
            "split",
            "serial",
            "peptide",
            "aggregation",
            target_column,
            "prediction",
            "prediction_probability",
        ]
        if column in all_predictions.columns
    ]
    all_predictions[prediction_columns].to_csv(
        output_path / "step_predictions.csv",
        index=False,
    )


@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--loader", type=click.Choice(list(LOADER_REGISTRY.keys())))
@click.option("--preprocessor", type=click.Choice(list(PREPROCESSOR_REGISTRY.keys())))
@click.option("--model", type=click.Choice(list(MODEL_REGISTRY.keys())))
@click.option("--task", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--target_column", type=str, default="first_diff_clean")
@click.option("--seed", type=int, default=3245)
@click.option("--n_repeats", type=int, default=0)
@click.option("--wof_start", type=int, required=False)
@click.option("--wof_end", type=int, required=False)
@click.option("--wof_drop", type=bool, required=False)
@click.option("--occurency_vector_normalise", type=bool, required=False)
@click.option("--save_step_predictions", is_flag=True)
def main(
    data_path: Path,
    output_path: Path,
    loader: str,
    preprocessor: str,
    model: str,
    task: str,
    target_column: str,
    seed: int,
    n_repeats: int,
    wof_start: Optional[int],
    wof_end: Optional[int],
    wof_drop: Optional[bool],
    occurency_vector_normalise: Optional[bool],
    save_step_predictions: bool,
) -> None:

    seed_everything(seed)

    f1, accuracy = list(), list()
    mse, pearson = list(), list()
    prediction_frames: list[pd.DataFrame] = list()

    if loader == "whole_set_shuffled":
        for i in tqdm.tqdm(range(n_repeats)):
            dataset_dict = load_data(
                data_path,
                loader,
                preprocessor,
                cv_split=0,
                padding=True,
                wof_start=wof_start,
                wof_end=wof_end,
                wof_drop=wof_drop,
                occurency_vector_normalise=occurency_vector_normalise,
                seed=i,
            )

            if task == "regression":
                mse_result, pearson_result, test_predictions = train_regression(
                    dataset_dict,
                    model,
                    target_column,
                )
                mse.append(mse_result)
                pearson.append(pearson_result)
            else:
                f1_result, accuracy_results, test_predictions = train(dataset_dict, model)
                f1.append(f1_result)
                accuracy.append(accuracy_results)

            add_prediction_frame(prediction_frames, test_predictions, i)

    else:
        for cv_split in range(5):
            logger.info(f"Running Split {cv_split}/5")

            if task == "regression" and model not in REGRESSION_MODEL_REGISTRY:
                raise ValueError(
                    f"Model {model} is not available for regression. "
                    f"Use one of: {list(REGRESSION_MODEL_REGISTRY)}."
                )

            if model in ["hc2", "timeforest", "weasel"] and (
                loader != "whole_set" or preprocessor != "sequence"
            ):
                raise ValueError(
                    f"Incompatible loader {loader} or preprocessor {preprocessor} "
                    f"for model {model}."
                )

            dataset_dict = load_data(
                data_path,
                loader,
                preprocessor,
                cv_split=cv_split,
                padding=True,
                wof_start=wof_start,
                wof_end=wof_end,
                wof_drop=wof_drop,
                occurency_vector_normalise=occurency_vector_normalise,
                seed=seed,
            )

            if task == "regression":
                if target_column not in dataset_dict["train"]:
                    raise ValueError(
                        f"Target column {target_column!r} is not present in the loaded dataset."
                    )
                mse_result, pearson_result, test_predictions = train_regression(
                    dataset_dict,
                    model,
                    target_column,
                )
                logger.info(
                    f"Finished Running Split {cv_split}/5 "
                    f"MSE: {mse_result:.3f} Pearson: {pearson_result:.3f}"
                )
                mse.append(mse_result)
                pearson.append(pearson_result)
            else:
                f1_result, accuracy_results, test_predictions = train(dataset_dict, model)
                logger.info(f"Finished Running Split {cv_split}/5 F1: {f1_result:.3f}")
                f1.append(f1_result)
                accuracy.append(accuracy_results)

            add_prediction_frame(prediction_frames, test_predictions, cv_split)

    if save_step_predictions:
        save_predictions(prediction_frames, output_path, target_column)

    with (output_path / "results.json").open("w") as results_file:
        if task == "regression":
            logger.info(f"MSE: {np.mean(mse):.3f}+-{np.std(mse):.3f}")
            logger.info(f"Pearson: {np.nanmean(pearson):.3f}+-{np.nanstd(pearson):.3f}")
            json.dump(
                {
                    "mse": {
                        "mean": np.mean(mse).astype(float),
                        "std": np.std(mse).astype(float),
                    },
                    "pearson": {
                        "mean": np.nanmean(pearson).astype(float),
                        "std": np.nanstd(pearson).astype(float),
                    },
                    "raw_mse": list(mse),
                    "raw_pearson": list(pearson),
                },
                results_file,
            )
        else:
            logger.info(f"Accuracy: {np.mean(accuracy):.3f}+-{np.std(accuracy):.3f}")
            json.dump(
                {
                    "f1": {
                        "mean": np.mean(f1).astype(float),
                        "std": np.std(f1).astype(float),
                    },
                    "raw_f1": list(f1),
                    "raw_acc": list(accuracy),
                },
                results_file,
            )
