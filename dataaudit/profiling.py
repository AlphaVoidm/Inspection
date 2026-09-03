"""Dataset profiling: structure, previews, per-column descriptive stats
(spec section 6)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .utils import example_values, is_numeric_dtype


def _column_stats(series: pd.Series) -> dict:
    n = len(series)
    non_null = int(series.count())
    missing = int(n - non_null)
    missing_pct = round(100.0 * missing / n, 4) if n else 0.0
    nunique = int(series.nunique(dropna=True))
    uniq_ratio = round(nunique / non_null, 6) if non_null else 0.0

    row = {
        "column": str(series.name),
        "dtype": str(series.dtype),
        "non_null": non_null,
        "missing": missing,
        "missing_pct": missing_pct,
        "unique": nunique,
        "uniqueness_ratio": uniq_ratio,
        "examples": example_values(series, 5),
    }
    if is_numeric_dtype(series.dtype):
        num = pd.to_numeric(series, errors="coerce")
        row.update({
            "min": float(num.min()) if non_null else None,
            "max": float(num.max()) if non_null else None,
            "mean": float(num.mean()) if non_null else None,
            "median": float(num.median()) if non_null else None,
            "std": float(num.std(ddof=0)) if non_null else None,
            "q05": float(num.quantile(0.05)) if non_null else None,
            "q25": float(num.quantile(0.25)) if non_null else None,
            "q75": float(num.quantile(0.75)) if non_null else None,
            "q95": float(num.quantile(0.95)) if non_null else None,
        })
    else:
        row.update({
            "min": None, "max": None, "mean": None, "median": None,
            "std": None, "q05": None, "q25": None, "q75": None, "q95": None,
            "top_values": example_values(series, 8),
        })
    return row


def profile_dataset(ctx: AuditContext, ds: Dataset) -> None:
    """Compute the per-column profile for a tabular dataset."""
    if not ds.is_tabular or ds.df is None:
        ds.profile = pd.DataFrame()
        return
    df = ds.df
    rows = [_column_stats(df[c]) for c in df.columns]
    ds.profile = pd.DataFrame(rows)
    # attach a column-type flag for downstream consumers
    ds.profile["numeric"] = [is_numeric_dtype(df[c].dtype) for c in df.columns]


def previews(ds: Dataset, n: int = 8) -> Dict[str, pd.DataFrame]:
    """Return head / tail / random previews for a tabular dataset."""
    if not ds.is_tabular or ds.df is None:
        return {}
    df = ds.df
    out = {"head": df.head(n), "tail": df.tail(n)}
    if len(df) > n:
        out["sample"] = df.sample(min(n, len(df)), random_state=42)
    return out


def structure_summary(ds: Dataset) -> dict:
    if ds.is_scientific:
        info = ds.scientific_info
        return {
            "file_id": ds.file_id,
            "kind": "scientific",
            "dims": info.get("dims", {}),
            "coords": info.get("coords", []),
            "data_vars": info.get("data_vars", []),
            "n_variables": len(info.get("data_vars", [])),
        }
    if not ds.is_tabular:
        return {"file_id": ds.file_id, "kind": ds.kind}
    df = ds.df
    return {
        "file_id": ds.file_id,
        "kind": "tabular",
        "shape": (ds.n_rows, ds.n_cols),
        "rows": ds.n_rows,
        "columns": ds.n_cols,
        "inspection_mode": ds.inspection_mode,
        "memory_bytes": ds.memory_bytes,
        "index": str(df.index.name) if df.index.name is not None else "RangeIndex",
        "index_type": type(df.index).__name__,
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
    }


def build_dataset_profiles_table(ctx: AuditContext) -> pd.DataFrame:
    rows = []
    for ds in ctx.datasets:
        if not ds.readable:
            rows.append({
                "dataset": ds.display_name, "file_type": ds.file_type,
                "status": ds.status, "inspection_mode": ds.inspection_mode,
                "rows": None, "columns": None, "kind": ds.kind,
            })
            continue
        rows.append({
            "dataset": ds.display_name,
            "file_type": ds.file_type,
            "status": ds.status,
            "inspection_mode": ds.inspection_mode,
            "rows": ds.n_rows,
            "columns": ds.n_cols,
            "kind": ds.kind,
            "memory_bytes": ds.memory_bytes,
            "n_variables": len(ds.columns) if ds.is_tabular else len(ds.scientific_info.get("data_vars", [])),
        })
    return pd.DataFrame(rows)
