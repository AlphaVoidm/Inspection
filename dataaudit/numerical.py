"""Numerical-variable audit (spec section 12)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .utils import is_numeric_dtype, is_constant, is_near_constant, skewness, iqr_bounds


def _num_audit_row(series: pd.Series, col: str) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    n = len(s)
    non_null = int(s.count())
    missing = int(s.isna().sum())
    if non_null == 0:
        return {
            "dataset": "", "variable": col, "non_null": 0, "missing": n,
            "missing_pct": 100.0, "min": None, "max": None, "mean": None,
            "median": None, "std": None, "variance": None, "q1": None, "q3": None,
            "iqr": None, "p05": None, "p95": None, "zeros": 0, "negatives": 0,
            "positives": 0, "infinite": 0, "nan": missing, "constant": None,
            "near_constant": None, "skewness": None, "n_iqr_outliers": None,
            "n_z_outliers": None,
        }
    lo, hi = iqr_bounds(s)
    z = (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and s.std(ddof=0) > 0 else pd.Series(0, index=s.index)
    return {
        "dataset": "",
        "variable": col,
        "non_null": non_null,
        "missing": missing,
        "missing_pct": round(100.0 * missing / n, 4) if n else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "variance": float(s.var(ddof=0)),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
        "zeros": int((s == 0).sum()),
        "negatives": int((s < 0).sum()),
        "positives": int((s > 0).sum()),
        "infinite": int(np.isinf(s).sum()),
        "nan": int(s.isna().sum()),
        "constant": bool(is_constant(series)),
        "near_constant": bool(is_near_constant(series)),
        "skewness": skewness(series),
        "n_iqr_outliers": int(((s < lo) | (s > hi)).sum()),
        "n_z_outliers": int((z.abs() > 3).sum()),
    }


def run_numerical_audit(ctx: AuditContext) -> None:
    for ds in ctx.tabular_datasets:
        df = ds.df
        rows = []
        for col in df.columns:
            if is_numeric_dtype(df[col].dtype):
                r = _num_audit_row(df[col], str(col))
                r["dataset"] = ds.display_name
                r["file_id"] = ds.file_id
                rows.append(r)
        ds.numerical_audit = pd.DataFrame(rows)
