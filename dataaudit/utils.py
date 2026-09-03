"""Small shared helpers used across the audit pipeline."""
from __future__ import annotations

import hashlib
import io
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for audit reproducibility."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_notebook() -> bool:
    """Best-effort detection of whether we are running inside a notebook."""
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            return True
    except Exception:
        pass
    return False


def display(obj: Any, max_rows: int = 20, max_cols: int = 30) -> Any:
    """Display an object in a notebook; print a compact form in a script.

    Returns the object so callers can chain/inspect it.
    """
    if is_notebook():
        from IPython.display import display as _display, Markdown  # type: ignore
        if isinstance(obj, str):
            _display(Markdown(obj))
        elif isinstance(obj, pd.DataFrame):
            with pd.option_context(
                "display.max_rows", max_rows,
                "display.max_columns", max_cols,
                "display.width", 140,
                "display.float_format", "{:,.4g}".format,
            ):
                _display(obj)
        else:
            _display(obj)
    else:
        if isinstance(obj, str):
            print(obj)
        elif isinstance(obj, pd.DataFrame):
            with pd.option_context(
                "display.max_rows", max_rows,
                "display.max_columns", max_cols,
                "display.width", 140,
                "display.float_format", "{:,.4g}".format,
            ):
                print(obj.to_string())
        else:
            print(obj)
    return obj


# ---------------------------------------------------------------------------
# Formatting / numeric helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n: Optional[float]) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "N/A"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0 or unit == "PB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} PB"


def df_memory(df: pd.DataFrame) -> int:
    """Approximate memory usage of a DataFrame in bytes (deep-ish)."""
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        try:
            return int(df.memory_usage().sum())
        except Exception:
            return -1


def hash_file(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, read in chunks so large files do not blow memory."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def as_float_series(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric where possible, NaN otherwise (no data loss)."""
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * len(series), index=series.index, dtype="float64")


def is_numeric_dtype(dtype) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def is_datetime_dtype(dtype) -> bool:
    return pd.api.types.is_datetime64_any_dtype(dtype)


def safe_round(x: Any, digits: int = 3) -> Any:
    if x is None:
        return None
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), digits)
    except (TypeError, ValueError):
        return x


def truncate_list(items: Iterable[Any], limit: int = 12) -> str:
    items = list(items)
    if len(items) <= limit:
        return "; ".join(str(i) for i in items)
    shown = "; ".join(str(i) for i in items[:limit])
    return f"{shown}; … (+{len(items) - limit} more)"


def example_values(series: pd.Series, n: int = 5) -> str:
    """A few distinct, printable example values for a column."""
    try:
        uniq = series.dropna().astype(str).unique()
    except Exception:
        uniq = series.dropna().map(str).unique()
    vals = list(uniq[:n])
    return truncate_list(vals, limit=n)


def error_string(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def error_record(path: str, stage: str, exc: BaseException) -> dict:
    return {
        "file": path,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "traceback": traceback.format_exc(limit=3),
    }


def python_env_summary() -> dict:
    """Environment info for reproducibility section."""
    env = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    for mod in ("pandas", "numpy", "pyarrow", "openpyxl", "xarray", "h5py",
                "netCDF4", "scipy", "matplotlib", "pycountry"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = "not installed"
    return env


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def iqr_bounds(series: pd.Series, k: float = 1.5):
    """Return (lower, upper) Tukey fence bounds for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def is_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= 1


def is_near_constant(series: pd.Series, ratio: float = 0.98) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    try:
        top = s.value_counts(normalize=True).iloc[0]
        return top >= ratio
    except Exception:
        return False


def skewness(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return None
    try:
        return float(s.skew())
    except Exception:
        return None


def safe_correlation(a: pd.Series, b: pd.Series) -> Optional[float]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 3:
        return None
    try:
        return float(m.iloc[:, 0].corr(m.iloc[:, 1]))
    except Exception:
        return None


def mode_of_diffs(values: Iterable[Any], max_vals: int = 2000) -> Optional[float]:
    """Most common value in a list of numeric diffs (frequency hint)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if arr.size > max_vals:  # sample for huge inputs
        rng = np.random.default_rng(0)
        arr = rng.choice(arr, size=max_vals, replace=False)
    try:
        counts = np.unique(np.round(arr, 6), return_counts=True)
        return float(counts[0][int(np.argmax(counts[1]))])
    except Exception:
        return None


def markdown_table(df: pd.DataFrame, float_format: str = "{:,.4g}") -> str:
    """Render a small DataFrame as a GitHub-flavoured markdown table string."""
    if df is None or df.empty:
        return "_No records._"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_format.format(v))
    cols = [str(c) for c in d.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row.get(c, "")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = ""
            cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)
