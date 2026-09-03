"""Dataset purpose inference and per-variable semantics (spec sections 19, 20)."""
from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .core import AuditContext, Dataset
from .static_data import PURPOSE_KEYWORDS
from .utils import example_values

# Known conceptual confusions we should flag, never silently resolve.
SEMANTIC_ALERTS = [
    ("capacity", ["generation", "production", "output", "energy", "demand", "load"],
     "CAPACITY ≠ PRODUCTION/GENERATION — capacity is a stock (MW installed), generation is a flow (MWh)."),
    ("consumption", ["demand", "load"],
     "CONSUMPTION ≠ DEMAND — final consumption can exclude losses/own-use that demand includes."),
    ("per capita", ["gdp", "income"],
     "GDP ≠ GDP PER CAPITA — a per-capita series is divided by population."),
    ("cdd", ["temperature", "temp", "tas"],
     "TEMPERATURE ≠ CDD/HDD — cooling/heating degree days are derived from temperature."),
    ("hdd", ["temperature", "temp", "tas"],
     "TEMPERATURE ≠ CDD/HDD — cooling/heating degree days are derived from temperature."),
    ("renewable capacity", ["renewable generation", "renewable share"],
     "RENEWABLE CAPACITY ≠ RENEWABLE GENERATION — installed capacity (MW) vs actual output (MWh)."),
    ("annual", ["monthly", "daily", "hourly"],
     "ANNUAL VALUE ≠ MONTHLY OBSERVATION — cannot be downsampled without assumptions."),
    ("price", ["volume", "quantity", "demand"],
     "PRICE ≠ VOLUME — monetary value vs physical quantity."),
    ("net", ["gross"],
     "NET ≠ GROSS — check whether losses/own-use are excluded."),
    ("forecast", ["actual", "observed", "realized"],
     "FORECAST ≠ ACTUAL — projected values are not observed history."),
]


def infer_purpose(ds: Dataset) -> Dict:
    """Infer what a dataset appears to contain, with evidence + confidence."""
    haystack_parts = [ds.filename, ds.rel_path]
    if ds.is_tabular and ds.df is not None:
        haystack_parts.append(" ".join(str(c) for c in ds.df.columns))
        if ds.metadata:
            haystack_parts.append(" ".join(str(v) for v in ds.metadata.values() if isinstance(v, str)))
    if ds.is_scientific:
        haystack_parts.append(" ".join(str(v) for v in ds.scientific_info.get("data_vars", [])))
        haystack_parts.append(" ".join(str(v) for v in ds.scientific_info.get("attrs", {}).values()))
    haystack = " ".join(haystack_parts).lower()

    scored = []
    for purpose, keywords in PURPOSE_KEYWORDS:
        hits = [k for k in keywords if k.lower() in haystack]
        if hits:
            scored.append({"purpose": purpose, "n_hits": len(hits), "hits": hits})
    scored.sort(key=lambda x: x["n_hits"], reverse=True)

    if not scored:
        return {"purpose": "UNKNOWN", "confidence": "UNKNOWN",
                "evidence": "no keyword evidence in filename/columns/metadata",
                "alternatives": []}
    best = scored[0]
    confidence = "HIGH" if best["n_hits"] >= 3 else ("MEDIUM" if best["n_hits"] >= 2 else "LOW")
    return {
        "purpose": best["purpose"],
        "confidence": confidence,
        "evidence": f"keywords {best['hits']} found in filename/columns/metadata",
        "alternatives": [s["purpose"] for s in scored[1:4]],
    }


def _measurement_type(col: str) -> str:
    c = str(col).lower()
    if any(k in c for k in ("forecast", "projection", "predicted", "scenario", "future")):
        return "FORECAST/PROJECTED"
    if any(k in c for k in ("model", "modelled", "modeled", "simulated")):
        return "MODELED"
    if any(k in c for k in ("estimate", "estimated", "est")):
        return "ESTIMATED"
    if any(k in c for k in ("derived", "index", "ratio", "share", "per_capita", "pct", "percent")):
        return "DERIVED"
    if any(k in c for k in ("actual", "observed", "realized", "real", "historic", "historical")):
        return "OBSERVED"
    return "UNKNOWN"


def _actual_forecast(col: str) -> str:
    c = str(col).lower()
    if any(k in c for k in ("forecast", "projection", "predicted", "scenario", "future")):
        return "FORECAST"
    if any(k in c for k in ("actual", "observed", "realized", "real")):
        return "ACTUAL"
    return "UNKNOWN"


def _gross_net(col: str) -> str:
    c = str(col).lower()
    if "gross" in c:
        return "GROSS"
    if "net" in c:
        return "NET"
    return "UNKNOWN"


def variable_semantics_table(ds: Dataset) -> pd.DataFrame:
    """Per-variable semantic description with evidence and confidence."""
    if not ds.is_tabular or ds.df is None:
        return pd.DataFrame()
    rows = []
    purpose = ds.purpose.get("purpose")
    has_time = ds.temporal.get("coverage") is not None
    freq = ds.temporal["coverage"]["frequency"] if has_time else "UNKNOWN"
    geo = "entity-level" if ds.entity_columns else "UNKNOWN"

    for col in ds.df.columns:
        role = "UNKNOWN"
        role_conf = "UNKNOWN"
        if ds.semantic_roles is not None:
            m = ds.semantic_roles[ds.semantic_roles["column"] == str(col)]
            if len(m):
                role = str(m.iloc[0]["role"])
                role_conf = str(m.iloc[0]["confidence"])
        unit = "UNIT_UNKNOWN"
        if ds.unit_catalog is not None and len(ds.unit_catalog):
            um = ds.unit_catalog[ds.unit_catalog["variable"] == str(col)]
            if len(um):
                unit = str(um.iloc[0]["unit"])
        alerts = []
        c = str(col).lower()
        for token, others, msg in SEMANTIC_ALERTS:
            if token in c:
                alerts.append(msg)
        rows.append({
            "dataset": ds.display_name,
            "variable": str(col),
            "likely_meaning": "UNKNOWN — insufficient evidence in name",
            "role": role,
            "role_confidence": role_conf,
            "unit": unit,
            "frequency": freq if role == "VALUE" or "FEATURE" in role or role == "TARGET" else "N/A",
            "geographic_scope": geo,
            "measurement_type": _measurement_type(str(col)),
            "actual_or_forecast": _actual_forecast(str(col)),
            "gross_or_net": _gross_net(str(col)),
            "semantic_alerts": "; ".join(alerts) if alerts else "",
            "definition_evidence": f"column name '{col}' (no external documentation)",
            "confidence": "LOW",
        })
    return pd.DataFrame(rows)


def run_semantics_audit(ctx: AuditContext) -> None:
    purpose_rows = []
    var_sem = []
    for ds in ctx.datasets:
        if not ds.readable:
            ds.purpose = {"purpose": "UNKNOWN", "confidence": "UNKNOWN",
                          "evidence": "dataset not readable", "alternatives": []}
            continue
        ds.purpose = infer_purpose(ds)
        purpose_rows.append({
            "dataset": ds.display_name,
            "purpose": ds.purpose["purpose"],
            "confidence": ds.purpose["confidence"],
            "evidence": ds.purpose["evidence"],
            "alternatives": ", ".join(ds.purpose.get("alternatives", [])),
        })
        if ds.is_tabular:
            vs = variable_semantics_table(ds)
            ds.variable_semantics = vs
            if len(vs):
                var_sem.append(vs)
    ctx.add_table("_purpose.csv", pd.DataFrame(purpose_rows))
    if var_sem:
        ctx.add_table("_variable_semantics.csv", pd.concat(var_sem, ignore_index=True))
