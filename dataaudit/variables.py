"""Column semantic-role inference and the global variable catalog
(spec sections 7, 20).  All inferences carry a confidence label."""
from __future__ import annotations

import re
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

from .core import AuditContext, Dataset
from .static_data import (
    AGGREGATE_ENTITIES, COUNTRY_ALIASES, LAT_RANGE, LON_RANGE, MONTH_RANGE,
    ROLE_KEYWORDS, YEAR_RANGE,
)
from .utils import example_values, is_numeric_dtype

try:
    import pycountry
    _PC = True
except Exception:  # pragma: no cover
    pycountry = None
    _PC = False


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------------
# Reference lookups
# ---------------------------------------------------------------------------

def _iso3_lookup(term: str) -> Optional[str]:
    """Map a country name/code to ISO3 (pycountry + curated aliases)."""
    t = term.strip()
    low = _norm(t)
    if low in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[low]
    if len(t) == 2:
        if _PC:
            c = pycountry.countries.get(alpha_2=t.upper())
            return c.alpha_3 if c else None
    if len(t) == 3:
        if _PC:
            c = pycountry.countries.get(alpha_3=t.upper())
            if c:
                return c.alpha_3
        # check alias table by ISO3 itself
        for iso in COUNTRY_ALIASES.values():
            if iso == t.upper():
                return iso
        return None
    if _PC:
        try:
            c = pycountry.countries.lookup(t)
            return c.alpha_3
        except Exception:
            c = pycountry.countries.get(name=t)
            return c.alpha_3 if c else None
    return None


def _iso2_lookup(term: str) -> Optional[str]:
    t = term.strip()
    if len(t) == 2:
        if _PC:
            c = pycountry.countries.get(alpha_2=t.upper())
            if c:
                return c.alpha_2
        # reverse alias
        iso3 = COUNTRY_ALIASES.get(_norm(t))
        return iso3 if iso3 else None
    if _PC:
        try:
            c = pycountry.countries.lookup(t)
            return c.alpha_2
        except Exception:
            pass
    iso3 = _iso3_lookup(t)
    if iso3 and _PC:
        c = pycountry.countries.get(alpha_3=iso3)
        return c.alpha_2 if c else None
    return None


# ---------------------------------------------------------------------------
# Value-based detectors
# ---------------------------------------------------------------------------

def _sample_nonnull(series: pd.Series, n: int = 200) -> pd.Series:
    s = series.dropna()
    if len(s) > n:
        return s.sample(n, random_state=0)
    return s


def is_datetime_like(series: pd.Series) -> Tuple[bool, float]:
    """Try to parse non-null values as datetimes. Returns (bool, frac)."""
    if is_numeric_dtype(series.dtype):
        return False, 0.0
    s = _sample_nonnull(series, 200).astype(str)
    if s.empty:
        return False, 0.0
    parsed = pd.to_datetime(s, errors="coerce")
    frac = float(parsed.notna().mean())
    return frac >= 0.85, frac


def is_year_like(series: pd.Series) -> Tuple[bool, float]:
    if not is_numeric_dtype(series.dtype):
        s = pd.to_numeric(series, errors="coerce")
    else:
        s = series
    s = s.dropna()
    if s.empty:
        return False, 0.0
    frac = float(((s >= YEAR_RANGE[0]) & (s <= YEAR_RANGE[1]) & (s == s.astype(int))).mean())
    return frac >= 0.9, frac


def is_month_like(series: pd.Series) -> Tuple[bool, float]:
    if not is_numeric_dtype(series.dtype):
        s = pd.to_numeric(series, errors="coerce")
    else:
        s = series
    s = s.dropna()
    if s.empty:
        return False, 0.0
    frac = float(((s >= MONTH_RANGE[0]) & (s <= MONTH_RANGE[1]) & (s == s.astype(int))).mean())
    return frac >= 0.9, frac


def is_lat_like(series: pd.Series) -> Tuple[bool, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False, 0.0
    frac = float(((s >= LAT_RANGE[0]) & (s <= LAT_RANGE[1])).mean())
    return frac >= 0.95, frac


def is_lon_like(series: pd.Series) -> Tuple[bool, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False, 0.0
    frac = float(((s >= LON_RANGE[0]) & (s <= LON_RANGE[1])).mean())
    return frac >= 0.95, frac


def country_match_frac(series: pd.Series, sample: int = 300) -> Tuple[float, bool]:
    """Fraction of values that look like country names/codes."""
    s = _sample_nonnull(series, sample).astype(str).str.strip()
    if s.empty:
        return 0.0, False
    n = len(s)
    matched = 0
    for v in s:
        if _norm(v) in AGGREGATE_ENTITIES:
            matched += 1
            continue
        if _iso3_lookup(v) is not None:
            matched += 1
    return matched / n, matched > 0


def _keyword_role(name: str) -> Optional[str]:
    n = _norm(name)
    if not n:
        return None
    # exact-ish token match: longest keyword wins to avoid "id" matching "valid"
    best = None
    best_len = -1
    for role, kws in ROLE_KEYWORDS.items():
        for kw in kws:
            kwn = _norm(kw)
            if not kwn:
                continue
            # token-boundary match within the name
            if kwn == n or n.endswith(kwn) or n.startswith(kwn) or f"_{kwn}" in n or kwn in n:
                if len(kwn) > best_len:
                    best = role
                    best_len = len(kwn)
    return best


# ---------------------------------------------------------------------------
# Main role inference
# ---------------------------------------------------------------------------

def infer_column_role(df: pd.DataFrame, col: str) -> dict:
    """Infer the likely semantic role of one column (never certain)."""
    series = df[col]
    name = str(col)
    n_name = _norm(name)
    kw_role = _keyword_role(name)

    dt_like, dt_frac = is_datetime_like(series)
    year_like, year_frac = is_year_like(series)
    month_like, month_frac = is_month_like(series)
    lat_like, lat_frac = is_lat_like(series)
    lon_like, lon_frac = is_lon_like(series)
    ctry_frac, ctry_seen = country_match_frac(series)

    role = "UNKNOWN"
    conf = "UNKNOWN"
    evidence = []

    def set_role(r, c, ev):
        nonlocal role, conf
        role, conf = r, c
        evidence.append(ev)

    # datetime
    if dt_like:
        set_role("DATETIME", "HIGH", f"{dt_frac:.0%} of sampled values parse as datetime")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}
    if kw_role in ("DATETIME", "DATE", "TIME"):
        set_role(kw_role, "HIGH", f"name token '{name}' suggests {kw_role}")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # explicit country columns
    if kw_role in ("COUNTRY", "COUNTRY_CODE"):
        if ctry_seen:
            set_role(kw_role, "HIGH", f"name + {ctry_frac:.0%} values match country names/codes")
        else:
            set_role(kw_role, "MEDIUM", "name suggests country; values not verified")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}
    if kw_role == "ISO3":
        iso_frac = float(series.dropna().astype(str).str.len().eq(3).mean()) if series.notna().any() else 0.0
        set_role("ISO3", "HIGH" if iso_frac >= 0.8 else "MEDIUM", f"name suggests ISO3; {iso_frac:.0%} values are 3 chars")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}
    if kw_role == "ISO2":
        iso_frac = float(series.dropna().astype(str).str.len().eq(2).mean()) if series.notna().any() else 0.0
        set_role("ISO2", "HIGH" if iso_frac >= 0.8 else "MEDIUM", f"name suggests ISO2; {iso_frac:.0%} values are 2 chars")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # coordinates
    if kw_role == "LATITUDE" or (lat_like and ("lat" in n_name)):
        set_role("LATITUDE", "HIGH", f"name/value in [-90,90]: {lat_frac:.0%}")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}
    if kw_role == "LONGITUDE" or (lon_like and ("lon" in n_name or "lng" in n_name or "long" in n_name)):
        set_role("LONGITUDE", "HIGH", f"name/value in [-180,180]: {lon_frac:.0%}")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # year / month
    if kw_role == "YEAR":
        set_role("YEAR", "HIGH" if year_like else "MEDIUM", f"name suggests year; {year_frac:.0%} values in {YEAR_RANGE}")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}
    if kw_role == "MONTH":
        set_role("MONTH", "HIGH" if month_like else "MEDIUM", f"name suggests month; {month_frac:.0%} values in 1-12")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # country inferred from values even if name is generic
    if ctry_seen and ctry_frac >= 0.9 and kw_role is None:
        set_role("COUNTRY", "HIGH", f"{ctry_frac:.0%} of values match country names/codes")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # geographic/entity roles by name
    for kw, target in [("REGION", "REGION"), ("CONTINENT", "CONTINENT"),
                       ("LOCATION", "LOCATION"), ("STATION", "LOCATION")]:
        if kw_role == kw:
            set_role(target, "MEDIUM", f"name token '{name}' suggests {target.lower()}")
            return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # unit
    if kw_role == "UNIT":
        set_role("UNIT", "MEDIUM", f"name token '{name}' suggests a unit column")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # numeric roles
    if is_numeric_dtype(series.dtype):
        if kw_role == "TARGET":
            set_role("TARGET", "MEDIUM", f"name token '{name}' suggests a target/label")
        elif kw_role == "VALUE":
            set_role("VALUE", "MEDIUM", f"name token '{name}' suggests a measurement value")
        elif kw_role in ("FORECAST", "ACTUAL"):
            set_role("VALUE", "MEDIUM", f"name token '{name}' marks actual/forecast status")
        else:
            set_role("NUMERICAL_FEATURE", "MEDIUM", "numeric column with no specific semantic signal")
        return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}

    # identifier / categorical / text
    if kw_role == "IDENTIFIER":
        set_role("IDENTIFIER", "MEDIUM", f"name token '{name}' suggests identifier")
    elif kw_role == "INDEX":
        set_role("INDEX", "MEDIUM", f"name token '{name}' suggests index")
    elif kw_role == "TEXT":
        set_role("TEXT", "MEDIUM", f"name token '{name}' suggests free text")
    else:
        nunique = int(series.nunique(dropna=True))
        if series.notna().any() and nunique <= max(50, 0.1 * len(series)):
            set_role("CATEGORICAL_FEATURE", "MEDIUM", f"low cardinality ({nunique} unique values)")
        else:
            set_role("TEXT", "LOW", "free-form text (unverified)")
    return {"role": role, "confidence": conf, "evidence": "; ".join(evidence)}


# ---------------------------------------------------------------------------
# Variable catalog
# ---------------------------------------------------------------------------

def build_variable_catalog(ctx: AuditContext) -> pd.DataFrame:
    rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        roles = []
        for col in df.columns:
            r = infer_column_role(df, col)
            roles.append(r)
            prof = ds.profile[ds.profile["column"] == str(col)] if ds.profile is not None and len(ds.profile) else None
            p = prof.iloc[0].to_dict() if prof is not None and len(prof) else {}
            rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "variable": str(col),
                "role": r["role"],
                "role_confidence": r["confidence"],
                "role_evidence": r["evidence"],
                "dtype": p.get("dtype"),
                "non_null": p.get("non_null"),
                "missing_pct": p.get("missing_pct"),
                "unique": p.get("unique"),
                "uniqueness_ratio": p.get("uniqueness_ratio"),
                "examples": p.get("examples"),
                "min": p.get("min"),
                "max": p.get("max"),
                "mean": p.get("mean"),
                "median": p.get("median"),
                "std": p.get("std"),
            })
        ds.semantic_roles = pd.DataFrame(roles)
        ds.semantic_roles["column"] = [str(c) for c in df.columns]
    return pd.DataFrame(rows)


def column_role_for(ds: Dataset, col: str) -> str:
    if ds.semantic_roles is None:
        return "UNKNOWN"
    m = ds.semantic_roles[ds.semantic_roles["column"] == str(col)]
    if len(m):
        return str(m.iloc[0]["role"])
    return "UNKNOWN"
