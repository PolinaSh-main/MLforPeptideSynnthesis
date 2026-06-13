import pandas as pd
import numpy as np
from scipy.stats import zscore

# загрузка
df = pd.read_csv("hydrophobicity_with_outliers.csv")

# удалить пустую колонку iLOGP (если есть)
df = df.loc[:, ~df.columns.str.contains(r"iLOGP|Unnamed", na=False)]

logp_cols = [
    "logP (XLOGP3)",
    "logP (WLOGP)",
    "logP (MLOGP)",
    "logP (SILICOS-IT)"
]

df[logp_cols] = df[logp_cols].apply(pd.to_numeric, errors="coerce")


# -----------------------------
# leave-one-out outlier check
# -----------------------------
def loo_filter(values, threshold=1.5):
    """
    values: array shape (4,)
    threshold: допустимое отклонение от mean остальных (logP units)
    """
    values = np.array(values, dtype=float)

    outlier_mask = np.zeros(len(values), dtype=bool)

    for i in range(len(values)):
        others = np.delete(values, i)

        # если нет валидных значений — пропуск
        if np.isnan(values[i]) or np.isnan(others).all():
            outlier_mask[i] = True
            continue

        others_mean = np.nanmean(others)

        if np.abs(values[i] - others_mean) > threshold:
            outlier_mask[i] = True

    return outlier_mask


def compute_consensus(row, threshold=1.5):
    vals = row[logp_cols].values.astype(float)

    mask = loo_filter(vals, threshold=threshold)

    cleaned = vals.copy()
    cleaned[mask] = np.nan

    return np.nanmean(cleaned)


def count_outliers(row, threshold=1.5):
    vals = row[logp_cols].values.astype(float)
    return loo_filter(vals, threshold=threshold).sum()


# -----------------------------
# расчёты
# -----------------------------

df["outlier_count"] = df.apply(lambda r: count_outliers(r, threshold=1.5), axis=1)

df["Consensus_logP_corrected"] = df.apply(
    lambda r: compute_consensus(r, threshold=1.5),
    axis=1
)

# флаг: если выброшено ≥1 метода
df["has_outlier"] = df["outlier_count"] > 0

# сохранение
df.to_csv("hydrophobicity_clean.csv", index=False)