"""Feature-availability matrix and leakage audit (spec sections 30–33)."""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from .core import AuditContext, Dataset


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _feature_match(feature: str, columns: List[str]) -> Optional[str]:
    """Best-match a project feature term to a discovered column name.

    Exact normalized match wins; otherwise the fraction of the feature's word
    tokens present in a column name is used (handles 'electricity demand' →
    'demand_mw', 'GDP' → 'gdp_usd', …).
    """
    feat_n = _norm(feature)
    feat_toks = set(re.findall(r"[a-z0-9]+", feature.lower()))
    best, best_score = None, 0.0
    for c in columns:
        cn = _norm(c)
        if feat_n == cn:
            return c
        ctoks = set(re.findall(r"[a-z0-9]+", str(c).lower()))
        inter = feat_toks & ctoks
        if not inter:
            continue
        score = len(inter) / len(feat_toks)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


def run_feature_availability(ctx: AuditContext) -> None:
    proj = ctx.config.resolve_project()
    candidates = proj["candidate_features"] or []
    rows = []

    # collect all variables across datasets
    all_vars = []
    for ds in ctx.tabular_datasets:
        for c in ds.df.columns:
            all_vars.append((ds, str(c)))

    if candidates:
        for feat in candidates:
            found = False
            for ds, col in all_vars:
                if _feature_match(feat, [col]):
                    rows.append(_feature_row(ctx, ds, col, feat))
                    found = True
            if not found:
                rows.append({
                    "feature": feat, "dataset": "(none)", "available": "NO",
                    "frequency": None, "start": None, "end": None,
                    "entities": None, "missing_pct": None, "unit": None,
                    "quality": None, "note": "no matching variable discovered",
                })
    else:
        # auto catalog of discovered variables
        for ds, col in all_vars:
            rows.append(_feature_row(ctx, ds, col, None))

    ctx.add_table("14_feature_availability.csv", pd.DataFrame(rows))


def _feature_row(ctx: AuditContext, ds: Dataset, col: str, feature: Optional[str]) -> dict:
    freq = ds.temporal["coverage"]["frequency"] if ds.temporal.get("coverage") else "N/A"
    start = str(ds.temporal["coverage"]["start"])[:19] if ds.temporal.get("coverage") else None
    end = str(ds.temporal["coverage"]["end"])[:19] if ds.temporal.get("coverage") else None
    missing = None
    if ds.profile is not None and len(ds.profile):
        m = ds.profile[ds.profile["column"] == col]
        if len(m):
            missing = m.iloc[0]["missing_pct"]
    unit = "UNIT_UNKNOWN"
    if ds.unit_catalog is not None and len(ds.unit_catalog):
        um = ds.unit_catalog[ds.unit_catalog["variable"] == col]
        if len(um):
            unit = um.iloc[0]["unit"]
    n_entities = None
    if ds.entity_columns and ds.entity_columns[0] in ds.df.columns:
        n_entities = int(ds.df[ds.entity_columns[0]].nunique())
    quality = None
    if ds.quality_scores:
        quality = ds.quality_scores.get("total")
    return {
        "feature": feature or col,
        "dataset": ds.display_name,
        "available": "YES",
        "frequency": freq,
        "start": start,
        "end": end,
        "entities": n_entities,
        "missing_pct": missing,
        "unit": unit,
        "quality": quality,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Leakage audit (spec section 32)
# ---------------------------------------------------------------------------

# Leakage heuristics: regex patterns matched against the RAW lowercased column
# name (word boundaries), so '_y' and 'y_' style suffixes/prefixes behave
# correctly.  These are name-based signals only — never proof of leakage.
LEAKAGE_PATTERNS = [
    (r"\bforecast\b", "forecast"),
    (r"\bprojection\b", "projection"),
    (r"\bpredicted\b", "predicted"),
    (r"\bprediction\b", "prediction"),
    (r"\bscenario\b", "scenario"),
    (r"\bfuture\b", "future"),
    (r"\btarget\b", "target"),
    (r"\blabel\b", "label"),
    (r"\bresponse\b", "response"),
    (r"\bactual\b", "actual"),
    (r"\brealized\b", "realized"),
    (r"\bnormalized\b", "normalized"),
    (r"\bstandardized\b", "standardized"),
    (r"\bscaled\b", "scaled"),
    (r"\bimputed\b", "imputed"),
    (r"\binterpolated\b", "interpolated"),
    (r"\bsmoothed\b", "smoothed"),
    (r"\bcumulative\b", "cumulative"),
    (r"\bcumsum\b", "cumsum"),
    (r"\bmoving_average\b", "moving average"),
    (r"\blagged\b", "lagged"),
    (r"\blead\b", "lead"),
    (r"\bnext_(?=[a-z0-9])", "next_"),
    (r"\byear_ahead\b", "year_ahead"),
    (r"\bgrowth_rate\b", "growth_rate"),
    (r"\bchange_pct\b", "change_pct"),
    (r"\bdiff_(?=[a-z0-9])", "diff_"),
    (r"(?<=[a-z0-9])_y\b", "_y (target-derived suffix)"),
    (r"\by_(?=[a-z0-9])", "y_ (target-derived prefix)"),
]


def run_leakage_audit(ctx: AuditContext) -> None:
    rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        for col in df.columns:
            c = str(col).lower()
            hits = [label for pat, label in LEAKAGE_PATTERNS if re.search(pat, c)]
            if not hits:
                continue
            futureish = any(t in hits for t in ("forecast", "projection", "predicted",
                                                "prediction", "scenario", "future",
                                                "target", "label", "response"))
            reason = ("column name indicates future or target-derived information"
                      if futureish else "column name indicates derived/transformed information")
            rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "variable": str(col),
                "leakage_risk": "LEAKAGE RISK",
                "reason": reason,
                "evidence": f"name patterns matched: {hits}",
                "confidence": "MEDIUM",
                "note": "Name-based signal only — verify the variable's construction before use in forecasting.",
            })
    ctx.add_table("15_leakage_flags.csv", pd.DataFrame(rows))


def annual_monthly_flags(ctx: AuditContext) -> pd.DataFrame:
    """Flag annual/quarterly variables when a monthly target is required (33)."""
    proj = ctx.config.resolve_project()
    required = (proj.get("required_frequency") or "").lower()
    rows = []
    if required != "monthly":
        return pd.DataFrame()
    for ds in ctx.tabular_datasets:
        cov = ds.temporal.get("coverage")
        freq = cov["frequency"] if cov else "UNKNOWN"
        if freq in ("ANNUAL", "QUARTERLY", "IRREGULAR"):
            rows.append({
                "dataset": ds.display_name,
                "frequency": freq,
                "required_frequency": "monthly",
                "flag": "FREQUENCY MISMATCH",
                "note": ("Annual variable detected. Possible treatments: retain annual; forward-fill; "
                         "step function; interpolate; lagged annual feature; derive another feature; exclude. "
                         "Do NOT silently convert."),
            })
    return pd.DataFrame(rows)
