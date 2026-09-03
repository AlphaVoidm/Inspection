"""Date/time detection, frequency inference and temporal coverage
(spec sections 8, 9)."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

from .core import AuditContext, Dataset
from .utils import is_datetime_dtype, is_numeric_dtype
from .variables import is_year_like, is_month_like


def _median_diff(ts: pd.Series) -> Optional[pd.Timedelta]:
    if len(ts) < 2:
        return None
    diffs = ts.diff().dropna()
    if diffs.empty:
        return None
    return diffs.median()


def _to_datetime_column(series: pd.Series) -> Optional[pd.Series]:
    if is_datetime_dtype(series.dtype):
        return series
    if is_numeric_dtype(series.dtype):
        return None
    s = series.dropna().astype(str)
    if s.empty:
        return None
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().mean() >= 0.85:
        return pd.Series(parsed.values, index=series.index)
    return None


def infer_frequency(ts: pd.Series) -> Tuple[str, str, str]:
    """Infer temporal frequency from actual spacing (never from first rows only).

    Returns (label, confidence, description).
    """
    ts = pd.Series(pd.to_datetime(ts, errors="coerce")).dropna().sort_values().drop_duplicates()
    n = len(ts)
    if n < 2:
        return "UNKNOWN", "UNKNOWN", "fewer than 2 distinct timestamps"

    diffs = ts.diff().dropna()
    median = diffs.median()
    total = diffs.count()

    # dominant step (most common diff, tolerance-bucketed)
    step_sec = median.total_seconds()

    label = "IRREGULAR"
    conf = "MEDIUM"

    if step_sec < 3600:
        label, conf = "SUB-HOURLY", "MEDIUM"
        desc = f"median step {median}"
    elif 3600 <= step_sec <= 7200:
        label, conf = "HOURLY", "HIGH" if (diffs == pd.Timedelta(hours=1)).mean() > 0.8 else "MEDIUM"
        desc = f"median step {median}"
    elif pd.Timedelta(days=1) <= median <= pd.Timedelta(days=1, hours=12):
        # daily or sub-daily; check dominance of exactly 1 day
        frac_day = (diffs == pd.Timedelta(days=1)).mean()
        label = "DAILY"
        conf = "HIGH" if frac_day > 0.8 else "MEDIUM"
        desc = f"{frac_day:.0%} of steps are 1 day"
    elif pd.Timedelta(days=6) <= median <= pd.Timedelta(days=8):
        frac_wk = ((diffs >= pd.Timedelta(days=6)) & (diffs <= pd.Timedelta(days=8))).mean()
        label = "WEEKLY"
        conf = "HIGH" if frac_wk > 0.8 else "MEDIUM"
        desc = f"{frac_wk:.0%} of steps are ~7 days"
    elif pd.Timedelta(days=27) <= median <= pd.Timedelta(days=32):
        # check calendar months (ordinal integers of PeriodIndex)
        months = ts.dt.to_period("M").astype("int64")
        msteps = pd.Series(months).diff().dropna()
        frac_month = (msteps == 1).mean()
        label = "MONTHLY"
        conf = "HIGH" if frac_month > 0.8 else "MEDIUM"
        desc = f"{frac_month:.0%} of steps are 1 month"
    elif pd.Timedelta(days=88) <= median <= pd.Timedelta(days=96):
        quarters = ts.dt.to_period("Q").astype("int64")
        qsteps = pd.Series(quarters).diff().dropna()
        frac_q = (qsteps == 1).mean()
        label = "QUARTERLY"
        conf = "HIGH" if frac_q > 0.8 else "MEDIUM"
        desc = f"{frac_q:.0%} of steps are 1 quarter"
    elif median >= pd.Timedelta(days=360):
        years = ts.dt.year
        ysteps = years.diff().dropna()
        frac_y = (ysteps == 1).mean()
        label = "ANNUAL"
        conf = "HIGH" if frac_y > 0.8 else "MEDIUM"
        desc = f"{frac_y:.0%} of steps are 1 year"
    else:
        # irregular-ish; refine with calendar pattern
        label = "IRREGULAR"
        conf = "MEDIUM"
        desc = f"median step {median} (no clean pattern)"
    return label, conf, desc


def _expected_periods(start, end, freq_label: str) -> Optional[int]:
    """Number of periods between start and end inclusive at a given freq."""
    try:
        if freq_label == "SUB-HOURLY":
            return None
        if freq_label == "HOURLY":
            rng = pd.date_range(start, end, freq="h")
            return len(rng)
        if freq_label == "DAILY":
            rng = pd.date_range(start, end, freq="D")
            return len(rng)
        if freq_label == "WEEKLY":
            rng = pd.date_range(start, end, freq="W")
            return len(rng)
        if freq_label == "MONTHLY":
            return len(pd.period_range(start, end, freq="M"))
        if freq_label == "QUARTERLY":
            return len(pd.period_range(start, end, freq="Q"))
        if freq_label == "ANNUAL":
            return len(pd.period_range(start, end, freq="Y"))
    except Exception:
        return None
    return None


def _period_key(ts: pd.Series, freq_label: str) -> pd.Series:
    ts = pd.to_datetime(ts)
    if freq_label == "MONTHLY":
        return ts.dt.to_period("M").astype(str)
    if freq_label == "QUARTERLY":
        return ts.dt.to_period("Q").astype(str)
    if freq_label == "ANNUAL":
        return ts.dt.year.astype(str)
    if freq_label == "DAILY":
        return ts.dt.strftime("%Y-%m-%d")
    if freq_label == "WEEKLY":
        return ts.dt.to_period("W").astype(str)
    if freq_label == "HOURLY":
        return ts.dt.strftime("%Y-%m-%d %H")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S")


def coverage_analysis(ts: pd.Series, freq_label: str) -> dict:
    """Temporal coverage metrics for one series of timestamps."""
    ts = pd.Series(pd.to_datetime(ts, errors="coerce")).dropna().sort_values()
    out: dict = {
        "n_observations": int(len(ts)),
        "start": ts.min() if len(ts) else None,
        "end": ts.max() if len(ts) else None,
        "frequency": freq_label,
        "n_unique_timestamps": int(ts.nunique()),
        "n_duplicate_timestamps": int((ts.duplicated()).sum()),
    }
    if len(ts) < 2:
        out.update({"expected_periods": None, "coverage_pct": None,
                    "n_gaps": None, "longest_gap": None, "missing_periods": None})
        return out

    start, end = ts.min(), ts.max()
    expected = _expected_periods(start, end, freq_label)
    keys = _period_key(ts, freq_label)
    uniq = set(keys.dropna())
    observed = len(uniq)
    out["expected_periods"] = expected
    out["observed_periods"] = observed
    out["coverage_pct"] = round(100.0 * observed / expected, 2) if expected else None

    # gaps
    missing = []
    if expected is not None and freq_label in ("MONTHLY", "QUARTERLY", "ANNUAL", "DAILY"):
        try:
            if freq_label == "MONTHLY":
                all_periods = set(pd.period_range(start, end, freq="M").astype(str))
            elif freq_label == "QUARTERLY":
                all_periods = set(pd.period_range(start, end, freq="Q").astype(str))
            elif freq_label == "ANNUAL":
                all_periods = set(str(y) for y in range(int(start.year), int(end.year) + 1))
            elif freq_label == "DAILY":
                all_periods = set(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))
            else:
                all_periods = set()
            missing = sorted(all_periods - uniq)
        except Exception:
            missing = []
    out["n_gaps"] = len(missing)
    out["missing_periods"] = missing[:200]
    out["missing_periods_truncated"] = len(missing) > 200
    # longest continuous gap in terms of missing periods
    if freq_label == "MONTHLY" and missing:
        longest = 1
        run = 1
        parsed = sorted(pd.Period(x, freq="M") for x in missing)
        for a, b in zip(parsed, parsed[1:]):
            if b - a == 1:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        out["longest_gap"] = longest
    else:
        out["longest_gap"] = len(missing) if missing else 0
    return out


def detect_temporal_columns(ds: Dataset) -> List[dict]:
    """Find candidate temporal columns (datetime, year/month parts, unix ts)."""
    out = []
    if not ds.is_tabular or ds.df is None:
        return out
    df = ds.df
    for col in df.columns:
        series = df[col]
        parsed = _to_datetime_column(series)
        if parsed is not None:
            out.append({
                "column": str(col), "kind": "datetime",
                "n_parsed": int(parsed.notna().sum()),
                "n_total": int(len(series)),
                "parsed": parsed,
            })
        elif is_numeric_dtype(series.dtype):
            # unix timestamp?
            s = series.dropna()
            name_lc = str(col).lower().strip()
            if (len(s) and name_lc.find("timestamp") >= 0) or \
               (len(s) and 1e9 <= s.median() <= 4e12 and
                name_lc in ("ts", "timestamp", "time", "unix", "epoch", "date")):
                try:
                    unit = "s" if s.median() < 1e11 else "ms"
                    parsed = pd.to_datetime(s, unit=unit, errors="coerce")
                    out.append({
                        "column": str(col), "kind": f"unix_timestamp({unit})",
                        "n_parsed": int(parsed.notna().sum()),
                        "n_total": int(len(series)),
                        "parsed": pd.Series(parsed.values, index=s.index),
                    })
                except Exception:
                    pass
            else:
                # separate year / month column?
                y_like, y_frac = is_year_like(series)
                m_like, _ = is_month_like(series)
                name_hint_year = name_lc in ("year", "yr", "jahr", "anno", "annee") or "year" in name_lc
                if y_like and name_hint_year:
                    try:
                        yrs = pd.to_numeric(series, errors="coerce").astype("Int64")
                        parsed = pd.to_datetime(yrs.astype(str) + "-01-01", errors="coerce")
                        out.append({
                            "column": str(col), "kind": "year",
                            "n_parsed": int(parsed.notna().sum()),
                            "n_total": int(len(series)),
                            "parsed": pd.Series(parsed.values, index=series.index),
                        })
                    except Exception:
                        pass
                elif m_like and str(col).lower().strip() in ("month", "monat", "mes"):
                    # month-only column can't form a full timestamp alone
                    out.append({"column": str(col), "kind": "month_only",
                                "n_parsed": int(series.notna().sum()),
                                "n_total": int(len(series)), "parsed": None})
    return out


def run_temporal_audit(ctx: AuditContext) -> None:
    """Populate ds.temporal for every readable dataset and the global table."""
    rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        tcols = detect_temporal_columns(ds)
        ds.temporal = {"columns": tcols, "primary": None, "coverage": None, "by_entity": None}

        primary = None
        for prefer in ("datetime", "year", "unix_timestamp"):
            for t in tcols:
                if t["kind"].startswith(prefer) and t.get("parsed") is not None:
                    primary = t
                    break
            if primary is not None:
                break
        if primary is None:
            for t in tcols:
                if t.get("parsed") is not None:
                    primary = t
                    break

        if primary is not None:
            ts = primary["parsed"].dropna()
            freq, conf, desc = infer_frequency(ts)
            cov = coverage_analysis(ts, freq)
            cov["column"] = primary["column"]
            cov["kind"] = primary["kind"]
            cov["frequency_confidence"] = conf
            cov["frequency_evidence"] = desc
            ds.temporal["primary"] = primary
            ds.temporal["coverage"] = cov
            rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "column": cov["column"],
                "kind": cov["kind"],
                "start": str(cov["start"])[:19],
                "end": str(cov["end"])[:19],
                "frequency": freq,
                "frequency_confidence": conf,
                "n_observations": cov["n_observations"],
                "n_unique_timestamps": cov["n_unique_timestamps"],
                "duplicate_timestamps": cov["n_duplicate_timestamps"],
                "expected_periods": cov["expected_periods"],
                "observed_periods": cov["observed_periods"],
                "coverage_pct": cov["coverage_pct"],
                "n_gaps": cov["n_gaps"],
                "longest_gap": cov["longest_gap"],
                "missing_periods_sample": ", ".join(map(str, cov.get("missing_periods", [])[:20])),
            })
        else:
            ds.temporal["coverage"] = None

    # scientific (NetCDF/HDF5) datasets: temporal coverage from the time coord
    for ds in ctx.scientific_datasets:
        info = ds.scientific_info
        ds.temporal = ds.temporal or {"columns": [], "primary": None, "coverage": None, "by_entity": None}
        if info.get("time_start") and info.get("time_end"):
            t0 = pd.to_datetime(info["time_start"], errors="coerce")
            t1 = pd.to_datetime(info["time_end"], errors="coerce")
            n = info.get("time_n", 0)
            freq, conf, desc = ("UNKNOWN", "UNKNOWN", "no frequency inferred")
            if n and pd.notna(t0) and pd.notna(t1):
                try:
                    rng = pd.date_range(t0, t1, periods=n)
                    freq, conf, desc = infer_frequency(pd.Series(rng))
                except Exception:
                    pass
            ds.temporal["coverage"] = {
                "column": "time (NetCDF coordinate)",
                "kind": "netcdf_time",
                "start": t0, "end": t1,
                "frequency": freq, "frequency_confidence": conf,
                "frequency_evidence": desc,
                "n_observations": n,
                "n_unique_timestamps": n,
                "duplicate_timestamps": 0,
                "expected_periods": None, "observed_periods": None,
                "coverage_pct": None, "n_gaps": None, "longest_gap": None,
                "missing_periods": [],
            }
            rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "column": "time (NetCDF coordinate)",
                "kind": "netcdf_time",
                "start": str(t0)[:19],
                "end": str(t1)[:19],
                "frequency": freq,
                "frequency_confidence": conf,
                "n_observations": n,
                "n_unique_timestamps": n,
                "duplicate_timestamps": 0,
                "expected_periods": None,
                "observed_periods": None,
                "coverage_pct": None,
                "n_gaps": None,
                "longest_gap": None,
                "missing_periods_sample": "",
            })
    ctx.add_table("04_temporal_coverage.csv", pd.DataFrame(rows))


def run_entity_temporal(ctx: AuditContext, ds: Dataset, entity_cols: List[str]) -> Optional[pd.DataFrame]:
    """Per-entity temporal coverage (spec section 9 'panel data')."""
    if not ds.is_tabular or ds.temporal.get("primary") is None:
        return None
    primary = ds.temporal["primary"]
    ts = primary["parsed"]
    freq, _, _ = infer_frequency(ts.dropna())
    df = ds.df.copy()
    df["__ts"] = pd.to_datetime(ts, errors="coerce")
    out_rows = []
    for col in entity_cols[:3]:  # at most 3 entity cols
        for ent, grp in df.groupby(col, dropna=True):
            gts = grp["__ts"].dropna().sort_values()
            if len(gts) < 2:
                continue
            cov = coverage_analysis(gts, freq)
            out_rows.append({
                "dataset": ds.display_name,
                "entity_column": str(col),
                "entity": str(ent),
                "start": str(cov["start"])[:19],
                "end": str(cov["end"])[:19],
                "frequency": freq,
                "observations": cov["n_observations"],
                "expected_periods": cov["expected_periods"],
                "coverage_pct": cov["coverage_pct"],
                "n_gaps": cov["n_gaps"],
                "longest_gap": cov["longest_gap"],
            })
    ds.temporal["by_entity"] = pd.DataFrame(out_rows) if out_rows else None
    return ds.temporal["by_entity"]
