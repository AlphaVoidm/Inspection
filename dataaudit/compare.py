"""Cross-dataset comparison for overlapping variables (spec section 22)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .overlap import _unit_for


def _entity_col(ds: Dataset) -> Optional[str]:
    for c in ds.entity_columns:
        if ds.df[c].nunique() > 1:
            return c
    return None


def _time_col(ds: Dataset) -> Optional[str]:
    cov = ds.temporal.get("coverage")
    if cov:
        return cov.get("column")
    return None


def _metrics(x: pd.Series, y: pd.Series) -> dict:
    m = pd.concat([x, y], axis=1).dropna()
    xv, yv = m.iloc[:, 0].astype(float), m.iloc[:, 1].astype(float)
    n = len(m)
    if n < 2:
        return {"n_overlap": n, "correlation": None, "mae": None, "rmse": None,
                "mean_diff": None, "median_diff": None, "mean_pct_diff": None,
                "median_ratio": None, "max_diff": None, "agreement_rate": None}
    diff = (xv - yv).abs()
    pct = diff / xv.replace(0, np.nan).abs() * 100.0
    ratio = xv / yv.replace(0, np.nan)
    agree = (xv == yv).mean() * 100.0
    return {
        "n_overlap": n,
        "correlation": round(float(xv.corr(yv)), 4),
        "mae": round(float(diff.mean()), 4),
        "rmse": round(float(np.sqrt((diff ** 2).mean())), 4),
        "mean_diff": round(float((xv - yv).mean()), 4),
        "median_diff": round(float((xv - yv).median()), 4),
        "mean_pct_diff": round(float(pct.mean()), 4),
        "median_ratio": round(float(ratio.median()), 4),
        "max_diff": round(float(diff.max()), 4),
        "agreement_rate": round(float(agree), 4),
    }


def _align(dsA: Dataset, colA: str, dsB: Dataset, colB: str) -> tuple:
    """Align two variables on (entity, time) keys. Returns (a, b, keys_desc)."""
    ea, eb = _entity_col(dsA), _entity_col(dsB)
    ta, tb = _time_col(dsA), _time_col(dsB)

    def build(ds, col, ent, time):
        cols = []
        for c in [col, ent, time]:
            if c is not None and c not in cols:
                cols.append(c)
        df = ds.df[cols].copy()
        df = df.rename(columns={col: "__value"})
        if ent:
            df["__entity"] = df[ent].astype(str)
        else:
            df["__entity"] = "ALL"
        if time:
            t = pd.to_datetime(df[time], errors="coerce")
            # align to month period for comparability across daily/hourly
            df["__period"] = t.dt.to_period("M").astype(str)
        else:
            df["__period"] = "NA"
        df["__value"] = pd.to_numeric(df["__value"], errors="coerce")
        return df.dropna(subset=["__value"])

    fa = build(dsA, colA, ea, ta)
    fb = build(dsB, colB, eb, tb)
    key_cols = ["__entity", "__period"]
    m = fa.merge(fb, on=key_cols, suffixes=("_a", "_b"))
    # if no entity distinction was possible, collapse
    if m.empty and ea is None and eb is None:
        m = fa[["__value"]].join(fb[["__value"]], how="inner", rsuffix="_b")
    return m, f"entity={'yes' if (ea or eb) else 'no'} time={'month' if (ta or tb) else 'no'}"


def run_comparison(ctx: AuditContext) -> None:
    overlap = ctx.get_table("10_overlap_analysis.csv")
    rows = []
    if overlap is None or overlap.empty:
        ctx.add_table("11_cross_dataset_comparison.csv", pd.DataFrame())
        return
    exact = overlap[overlap["match_kind"] == "name_exact"]
    by_id = ctx.dataset_by_id

    for _, ov in exact.iterrows():
        dsA = next((d for d in ctx.datasets if d.display_name == ov["dataset_a"]), None)
        dsB = next((d for d in ctx.datasets if d.display_name == ov["dataset_b"]), None)
        if dsA is None or dsB is None or not dsA.is_tabular or not dsB.is_tabular:
            continue
        colA, colB = ov["variable_a"], ov["variable_b"]
        if colA not in dsA.df.columns or colB not in dsB.df.columns:
            continue
        # only numeric variables are compared; categorical name-overlaps are
        # already documented in the overlap table.
        if not (pd.api.types.is_numeric_dtype(dsA.df[colA]) and
                pd.api.types.is_numeric_dtype(dsB.df[colB])):
            continue

        # validity: units must be compatible, frequency must be comparable
        ua, ub = _unit_for(dsA, colA), _unit_for(dsB, colB)
        if ua and ub and ua != ub:
            rows.append({
                "variable_a": colA, "dataset_a": dsA.display_name,
                "variable_b": colB, "dataset_b": dsB.display_name,
                "n_overlap": 0, "valid": False,
                "reason": f"COMPARISON NOT VALID — unit mismatch ({ua} vs {ub})",
            })
            continue

        try:
            m, keys_desc = _align(dsA, colA, dsB, colB)
        except Exception as exc:
            rows.append({
                "variable_a": colA, "dataset_a": dsA.display_name,
                "variable_b": colB, "dataset_b": dsB.display_name,
                "n_overlap": 0, "valid": False,
                "reason": f"COMPARISON NOT VALID — alignment error: {exc}",
            })
            continue
        if m.empty or "__value_a" not in m.columns:
            rows.append({
                "variable_a": colA, "dataset_a": dsA.display_name,
                "variable_b": colB, "dataset_b": dsB.display_name,
                "n_overlap": 0, "valid": False,
                "reason": "COMPARISON NOT VALID — no overlapping (entity, period) keys",
            })
            continue
        met = _metrics(m["__value_a"], m["__value_b"])
        rows.append({
            "variable_a": colA, "dataset_a": dsA.display_name,
            "variable_b": colB, "dataset_b": dsB.display_name,
            "n_overlap": met["n_overlap"], "valid": True,
            "reason": f"aligned by {keys_desc}",
            "correlation": met["correlation"], "mae": met["mae"], "rmse": met["rmse"],
            "mean_diff": met["mean_diff"], "median_diff": met["median_diff"],
            "mean_pct_diff": met["mean_pct_diff"], "median_ratio": met["median_ratio"],
            "max_diff": met["max_diff"], "agreement_rate": met["agreement_rate"],
        })
    ctx.add_table("11_cross_dataset_comparison.csv", pd.DataFrame(rows))
