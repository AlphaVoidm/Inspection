"""Geographic / entity detection and country-normalization analysis
(spec sections 10, 11)."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .core import AuditContext, Dataset
from .static_data import AGGREGATE_ENTITIES
from .utils import example_values
from .variables import _iso3_lookup, _iso2_lookup, _norm, country_match_frac

ENTITY_ROLES = {"COUNTRY", "COUNTRY_CODE", "ISO2", "ISO3", "REGION", "CONTINENT",
                "LOCATION", "STATION", "IDENTIFIER"}


def detect_entity_columns(ds: Dataset) -> List[str]:
    """Columns whose inferred role is entity-like."""
    if not ds.is_tabular or ds.df is None or ds.semantic_roles is None:
        return []
    out = []
    for _, r in ds.semantic_roles.iterrows():
        if r.get("role") in ENTITY_ROLES:
            out.append(str(r["column"]))
    return out


def _aggregate_flag(value: str) -> bool:
    return _norm(value) in AGGREGATE_ENTITIES


def analyze_entity_column(series: pd.Series, col: str, ds: Dataset) -> dict:
    s = series.astype(str).str.strip()
    non_null = s[s != "nan"]
    nunique = int(non_null.nunique())
    ctry_frac, ctry_seen = country_match_frac(series)

    # map each distinct value to a canonical code where possible
    mappings = []
    seen_canon = {}
    for v in sorted(non_null.unique()):
        if _aggregate_flag(v):
            mappings.append({"original": v, "canonical": None, "type": "Aggregate",
                             "confidence": "HIGH"})
            continue
        iso3 = _iso3_lookup(v)
        if iso3:
            mappings.append({"original": v, "canonical": iso3, "type": "Country",
                             "confidence": "HIGH"})
            seen_canon.setdefault(iso3, []).append(v)
        else:
            mappings.append({"original": v, "canonical": None, "type": "AMBIGUOUS_ENTITY",
                             "confidence": "LOW"})

    # duplicate representations: canonical code with >1 spelling
    dup_reprs = {k: v for k, v in seen_canon.items() if len(v) > 1}

    return {
        "column": str(col),
        "n_unique": nunique,
        "n_missing": int(series.isna().sum()),
        "examples": example_values(series, 12),
        "country_match_frac": round(ctry_frac, 3),
        "aggregates_detected": [m["original"] for m in mappings if m["type"] == "Aggregate"][:50],
        "duplicate_representations": dup_reprs,
        "mapping": mappings,
    }


def run_entity_audit(ctx: AuditContext) -> None:
    global_rows = []
    for ds in ctx.tabular_datasets:
        ds.entity_columns = detect_entity_columns(ds)
        if not ds.entity_columns:
            ds.entity_report = None
            ds.country_mapping = None
            continue
        reports = []
        mapping_rows = []
        for col in ds.entity_columns:
            rep = analyze_entity_column(ds.df[col], col, ds)
            reports.append(rep)
            for m in rep["mapping"]:
                mapping_rows.append({
                    "dataset": ds.display_name,
                    "column": col,
                    "original_value": m["original"],
                    "proposed_canonical": m["canonical"],
                    "type": m["type"],
                    "confidence": m["confidence"],
                })
            global_rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "entity_column": col,
                "n_unique": rep["n_unique"],
                "n_missing": rep["n_missing"],
                "country_match_frac": rep["country_match_frac"],
                "examples": rep["examples"],
                "n_aggregates": len(rep["aggregates_detected"]),
                "n_duplicate_representations": len(rep["duplicate_representations"]),
            })
        ds.entity_report = reports
        ds.country_mapping = pd.DataFrame(mapping_rows) if mapping_rows else None
    ctx.add_table("05_entity_coverage.csv", pd.DataFrame(global_rows))


def country_mapping_table(ctx: AuditContext) -> pd.DataFrame:
    """Flattened proposed normalization table across all datasets."""
    frames = [d.country_mapping for d in ctx.tabular_datasets if d.country_mapping is not None]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["dataset", "column", "original_value",
                                 "proposed_canonical", "type", "confidence"])
