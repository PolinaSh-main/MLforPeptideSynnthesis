# utils/data_logger.py
import logging
import functools
import numpy as np
import pandas as pd

logger = logging.getLogger("data_pipeline")


def _format_dataframe_info(df: pd.DataFrame, name: str, step: str) -> str:
    """Форматирует полную информацию о DataFrame."""
    lines = [f"\n{'='*80}"]
    lines.append(f"[{name}] {step}")
    lines.append(f"{'='*80}")
    
    # Размер
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Колонки
    lines.append(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        lines.append(f"  {i:2d}. {col:20s} | dtype: {str(dtype):15s} | nulls: {null_count:5d} ({null_pct:5.1f}%)")
    
    # Null-значения
    null_summary = df.isnull().sum()
    if null_summary.sum() > 0:
        lines.append(f"\nNull values summary:")
        for col, count in null_summary[null_summary > 0].items():
            lines.append(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Статистика по числовым колонкам
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        lines.append(f"\nNumeric columns statistics:")
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            lines.append(f"  {col}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    # Sample данных
    lines.append(f"\nSample (first 3 rows):")
    lines.append(df.head(3).to_string())
    lines.append(f"{'='*80}\n")
    
    return "\n".join(lines)


def log_dataframe(df: pd.DataFrame, name: str, step: str = "TRANSFORM"):
    """Логирует полную информацию о DataFrame."""
    if isinstance(df, pd.DataFrame):
        info_text = _format_dataframe_info(df, name, step)
        logger.info(info_text)


def log_step(step_name: str = None):
    """Декоратор для функций, принимающих и возвращающих DataFrame."""
    def decorator(func):
        name = step_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Найти DataFrame среди аргументов
            df_in = next(
                (a for a in args if isinstance(a, pd.DataFrame)),
                kwargs.get("data") or kwargs.get("df") or kwargs.get("subset"),
            )
            if df_in is not None:
                log_dataframe(df_in, name, "INPUT")

            result = func(*args, **kwargs)

            if isinstance(result, pd.DataFrame):
                log_dataframe(result, name, "OUTPUT")

            return result
        return wrapper
    return decorator