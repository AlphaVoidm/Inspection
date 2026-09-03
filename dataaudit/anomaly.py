"""Outlier inspection and structural-break detection (spec sections 34, 35)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .temporal import detect_temporal_columns, infer_frequency, _period_key
from .utils import is_numeric_dtype, iqr_bounds

MAX_NUM_VARS_OUTLIERS = 15


def _classify_outlier(value, s: pd.Series, temporal: bool) -> str:
    lo, hi = iqr_bounds(s, k=3.0)  # extreme fence
    if not np.isnan(lo) and not np.isnan(hi) and (value < lo or value > hi):
        return "POSSIBLE_ERROR" if not temporal else "POSSIBLE_DATA_ERROR"
    lo1, hi1 = iqr_bounds(s, k=1.5)
    if not np.isnan(lo1) and not np.isnan(hi1) and (value < lo1 or value > hi1):
        return "STATISTICAL_OUTLIER"
    return "UNKNOWN"


def _outlier_rows(ds: Dataset, df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    rows = []
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c].dtype)]
    for col in numeric_cols[:MAX_NUM_VARS_OUTLIERS]:
        s = pd.to_numeric(df[col], errors="coerce")
        nonna = s.dropna()
        if len(nonna) < 10:
            continue
        lo, hi = iqr_bounds(s)
        outlier_idx = s[(s < lo) | (s > hi)].sort_values()
        if len(outlier_idx) == 0:
            continue
        # sample the most extreme values for the report
        s_abs_dev = (s - s.median()).abs()
        extreme = s_abs_dev.nlargest(min(10, len(s_abs_dev))).index
        for idx in extreme:
            v = s.loc[idx]
            if pd.isna(v):
                continue
            row = {
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "variable": str(col),
                "value": float(v),
                "kind": "STATISTICAL_OUTLIER",
                "classification": _classify_outlier(v, s, temporal=time_col is not None),
                "method": "IQR (1.5×) / extreme deviance",
                "evidence": f"value {v:g} vs median {s.median():g}, IQR [{lo:g}, {hi:g}]",
            }
            if time_col:
                row["timestamp"] = str(df.loc[idx, time_col])[:19]
            rows.append(row)
    return pd.DataFrame(rows)


def _structural_breaks(ds: Dataset, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    rows = []
    ts = pd.to_datetime(df[time_col], errors="coerce")
    freq, _, _ = infer_frequency(ts.dropna())
    keys = _period_key(ts, freq)
    tmp = df.copy()
    tmp["__period"] = keys
    tmp = tmp[tmp["__period"].notna()]

    # choose a primary numeric variable to monitor (prefer value-like)
    value_col = None
    for c in df.columns:
        if is_numeric_dtype(df[c].dtype) and str(c).lower() in ("value", "load", "demand", "generation", "price", "gdp", "temperature", "temp", "population"):
            value_col = c
            break
    if value_col is None:
        value_col = next((c for c in df.columns if is_numeric_dtype(df[c].dtype)), None)
    if value_col is None:
        return pd.DataFrame()

    g = tmp.groupby("__period")
    means = g[value_col].mean()
    stds = g[value_col].std()
    counts = g[value_col].count()
    miss = g[value_col].apply(lambda s: s.isna().mean())

    # level shift: mean jump vs rolling std of the mean
    if len(means) >= 6:
        roll_std = means.rolling(6, min_periods=3).std()
        for i in range(1, len(means)):
            prev, cur = means.iloc[i - 1], means.iloc[i]
            rstd = roll_std.iloc[i]
            if pd.notna(rstd) and rstd > 0 and abs(cur - prev) > 3 * rstd:
                rows.append({
                    "dataset": ds.display_name, "variable": value_col,
                    "period": str(means.index[i]),
                    "signal": "SUDDEN LEVEL SHIFT",
                    "evidence": f"mean {prev:g} → {cur:g} (>{3:g}× rolling std {rstd:g})",
                })
    # variance shift
    if len(stds) >= 6:
        for i in range(1, len(stds)):
            prev, cur = stds.iloc[i - 1], stds.iloc[i]
            if pd.notna(prev) and pd.notna(cur) and prev > 0:
                ratio = cur / prev
                if ratio > 4 or ratio < 0.25:
                    rows.append({
                        "dataset": ds.display_name, "variable": value_col,
                        "period": str(stds.index[i]),
                        "signal": "SUDDEN VARIANCE SHIFT",
                        "evidence": f"std {prev:g} → {cur:g} (ratio {ratio:g})",
                    })
    # missingness regime change
    for i in range(1, len(miss)):
        prev, cur = miss.iloc[i - 1], miss.iloc[i]
        if prev < 0.05 and cur > 0.3:
            rows.append({
                "dataset": ds.display_name, "variable": value_col,
                "period": str(miss.index[i]),
                "signal": "NEW MISSINGNESS REGIME",
                "evidence": f"missing share {prev:.0%} → {cur:.0%}",
            })
    # coverage change (entity count per period)
    if ds.entity_columns:
        ec = ds.entity_columns[0]
        ents = g[ec].nunique()
        if len(ents) >= 4:
            base = ents.iloc[: min(4, len(ents))].median()
            if base and base > 0:
                for i in range(len(ents)):
                    if ents.iloc[i] < 0.7 * base:
                        rows.append({
                            "dataset": ds.display_name, "variable": str(ec),
                            "period": str(ents.index[i]),
                            "signal": "COVERAGE CHANGE",
                            "evidence": f"entities dropped from ~{base:.0f} to {ents.iloc[i]:.0f}",
                        })
                        break
    return pd.DataFrame(rows)


def run_anomaly_audit(ctx: AuditContext) -> None:
    frames = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        tcols = detect_temporal_columns(ds)
        time_col = tcols[0]["column"] if tcols else None
        out = _outlier_rows(ds, df, time_col)
        br = _structural_breaks(ds, df, time_col) if time_col else pd.DataFrame()
        ds.anomaly_report = out
        if len(out):
            frames.append(out)
        if len(br):
            frames.append(br.rename(columns={"variable": "variable", "signal": "kind"}))
    if frames:
        ctx.add_table("16_anomaly_report.csv", pd.concat(frames, ignore_index=True))
    else:
        ctx.add_table("16_anomaly_report.csv", pd.DataFrame())
