"""Missing-data audit (spec section 13)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .temporal import detect_temporal_columns, infer_frequency, _period_key


def _longest_missing_run(series: pd.Series) -> int:
    isna = series.isna()
    best = 0
    run = 0
    for v in isna:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _classify_pattern(ds: Dataset, per_var: pd.DataFrame, time_col: Optional[str]) -> str:
    """Heuristic missingness-pattern classifier (evidence-based, low confidence)."""
    n_cols = ds.n_cols or 0
    if n_cols == 0:
        return "UNKNOWN"
    # variable-based: some columns almost fully missing, others complete
    high_missing = (per_var["missing_pct"] > 60).sum()
    low_missing = (per_var["missing_pct"] < 5).sum()
    if high_missing >= 1 and low_missing >= 1:
        return "VARIABLE-BASED"
    # block missingness: contiguous row blocks missing across many columns
    df = ds.df
    row_missing = df.isna().mean(axis=1)
    if row_missing.max() > 0.5 and (row_missing > 0.5).sum() >= 5:
        # are they contiguous?
        flags = (row_missing > 0.5).astype(int)
        changes = (flags.diff() != 0).sum()
        if changes <= 4 and flags.sum() > 0:
            return "BLOCK MISSINGNESS"
    # time-based: missing concentrated at beginning/end
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        missing_rows = df.isna().any(axis=1)
        if missing_rows.any() and t.notna().any():
            first_half = missing_rows.loc[t <= t.median()].mean()
            second_half = missing_rows.loc[t > t.median()].mean()
            if abs(first_half - second_half) > 0.3:
                return "TIME-BASED"
    # entity-based is checked in run_missingness when entity cols exist
    return "RANDOM-LOOKING" if df.isna().any().any() else "NONE"


def _scientific_missingness(ds: Dataset) -> Optional[dict]:
    """Sample the primary NetCDF/HDF5 variable for NaN / fill-value share."""
    try:
        import xarray as xr
        with xr.open_dataset(ds.path) as d:
            primary = None
            for v in d.data_vars:
                if v not in ("time_bnds", "lat_bnds", "lon_bnds"):
                    primary = v
                    break
            if primary is None:
                return None
            da = d[primary]
            flat = da.values.ravel()
            n = flat.size
            sampled = False
            if n > 200_000:
                idx = np.random.default_rng(0).choice(n, size=200_000, replace=False)
                flat = flat[idx]
                sampled = True
            nan_frac = 0.0
            if np.issubdtype(flat.dtype, np.number):
                nan_frac = float(np.isnan(flat).mean())
            fv = da.attrs.get("_FillValue", da.attrs.get("missing_value"))
            fill_frac = float((flat == fv).mean()) if fv is not None else 0.0
            return {"variable": primary, "n_total": int(n), "sampled": sampled,
                    "nan_frac": nan_frac, "fill_value": fv, "fill_frac": fill_frac}
    except Exception:
        return None


def run_missingness_audit(ctx: AuditContext) -> None:
    global_rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        n_rows, n_cols = len(df), df.shape[1]
        total_cells = n_rows * n_cols
        missing_cells = int(df.isna().sum().sum())
        total_missing_pct = round(100.0 * missing_cells / total_cells, 4) if total_cells else 0.0
        complete_rows = int((~df.isna().any(axis=1)).sum())
        complete_cols = [c for c in df.columns if df[c].notna().all()]

        per_var = pd.DataFrame({
            "column": [str(c) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [round(100.0 * df[c].isna().mean(), 4) for c in df.columns],
            "longest_missing_run": [_longest_missing_run(df[c]) for c in df.columns],
        })

        # time column for time-based analysis
        tcols = detect_temporal_columns(ds)
        time_col = tcols[0]["column"] if tcols else None
        by_time = None
        if time_col:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            freq, _, _ = infer_frequency(ts.dropna())
            keys = _period_key(ts, freq)
            tmp = pd.DataFrame({"period": keys, "missing": df.isna().any(axis=1)})
            by_time = tmp.groupby("period")["missing"].agg(["count", "sum", "mean"]).reset_index()
            by_time.columns = ["period", "n_rows", "n_rows_with_missing", "missing_share"]
            by_time = by_time.sort_values("period").tail(50)  # most recent periods for report

        # entity-based missingness
        by_entity = None
        entity_col = None
        for ec in ds.entity_columns:
            if df[ec].nunique() > 1 and df[ec].nunique() <= 2000:
                entity_col = ec
                break
        if entity_col:
            g = df.groupby(entity_col, dropna=True)
            m = g.apply(lambda x: x.isna().mean().mean() * 100.0)
            n = g.size()
            by_entity = pd.DataFrame({"entity": m.index, "n_rows": n.values, "missing_pct": m.values})
            by_entity = by_entity.sort_values("missing_pct", ascending=False)

        # entity x time coverage matrix (sampled for size)
        coverage_matrix = None
        if entity_col and time_col:
            try:
                ts = pd.to_datetime(df[time_col], errors="coerce")
                freq, _, _ = infer_frequency(ts.dropna())
                keys = _period_key(ts, freq)
                cm = pd.crosstab(df[entity_col], keys)
                if cm.shape[0] > 60:
                    cm = cm.sample(min(60, cm.shape[0]), random_state=0)
                if cm.shape[1] > 80:
                    cm = cm.iloc[:, -80:]
                coverage_matrix = cm
            except Exception:
                coverage_matrix = None

        pattern = _classify_pattern(ds, per_var, time_col)

        ds.missingness = {
            "n_rows": n_rows, "n_cols": n_cols,
            "missing_cells": missing_cells,
            "total_missing_pct": total_missing_pct,
            "complete_rows": complete_rows,
            "complete_rows_pct": round(100.0 * complete_rows / n_rows, 4) if n_rows else 0.0,
            "complete_columns": complete_cols,
            "per_variable": per_var,
            "by_time": by_time,
            "by_entity": by_entity,
            "coverage_matrix": coverage_matrix,
            "pattern": pattern,
            "time_col": time_col,
            "entity_col": entity_col,
        }
        global_rows.append({
            "dataset": ds.display_name,
            "file_id": ds.file_id,
            "rows": n_rows,
            "columns": n_cols,
            "missing_cells": missing_cells,
            "total_missing_pct": total_missing_pct,
            "complete_rows": complete_rows,
            "complete_rows_pct": ds.missingness["complete_rows_pct"],
            "n_complete_columns": len(complete_cols),
            "pattern": pattern,
            "time_col": time_col,
            "entity_col": entity_col,
        })

    # scientific (NetCDF/HDF5) datasets: metadata-sampled missingness
    for ds in ctx.scientific_datasets:
        sci = _scientific_missingness(ds)
        if sci is None:
            ds.missingness = {"n_rows": None, "n_cols": None, "missing_cells": None,
                              "total_missing_pct": None, "complete_rows": None,
                              "complete_rows_pct": None, "complete_columns": [],
                              "per_variable": None, "by_time": None, "by_entity": None,
                              "coverage_matrix": None, "pattern": "UNKNOWN",
                              "time_col": None, "entity_col": None}
            global_rows.append({
                "dataset": ds.display_name, "file_id": ds.file_id, "rows": None,
                "columns": None, "missing_cells": None, "total_missing_pct": None,
                "complete_rows": None, "complete_rows_pct": None,
                "n_complete_columns": None, "pattern": "UNKNOWN",
                "time_col": None, "entity_col": None,
            })
            continue
        pct = round(100.0 * (sci["nan_frac"] + sci["fill_frac"]), 4)
        ds.missingness = {
            "n_rows": None, "n_cols": None, "missing_cells": None,
            "total_missing_pct": pct, "complete_rows": None, "complete_rows_pct": None,
            "complete_columns": [], "per_variable": None, "by_time": None, "by_entity": None,
            "coverage_matrix": None, "pattern": "METADATA-SAMPLE",
            "time_col": None, "entity_col": None,
            "scientific": sci,
        }
        global_rows.append({
            "dataset": ds.display_name, "file_id": ds.file_id, "rows": sci["n_total"],
            "columns": None, "missing_cells": None,
            "total_missing_pct": pct, "complete_rows": None, "complete_rows_pct": None,
            "n_complete_columns": None, "pattern": "METADATA-SAMPLE (sampled)",
            "time_col": None, "entity_col": None,
        })
    ctx.add_table("06_missingness.csv", pd.DataFrame(global_rows))
