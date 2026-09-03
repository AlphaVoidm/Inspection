"""Data-quality scoring (spec sections 24, 25) and usability (26)."""
from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd

from .core import AuditContext, Dataset

BANDS = [
    (34, "Excellent"), (28, "Strong"), (20, "Usable with caveats"),
    (12, "Weak"), (0, "Poor"),
]


def classify_score(total: float) -> str:
    for lo, label in BANDS:
        if total >= lo:
            return label
    return "Poor"


def _completeness(ds: Dataset) -> int:
    m = ds.missingness
    if not m:
        return 0
    pct = m.get("total_missing_pct", 100.0)
    score = 5 * (1 - pct / 100.0)
    return max(0, min(5, round(score)))


def _consistency(ds: Dataset) -> int:
    score = 5
    dup = ds.duplicates
    if dup:
        pct = dup.get("exact_duplicate_pct", 0.0)
        if pct > 5:
            score -= 3
        elif pct > 0.1:
            score -= 1
        if sum(1 for k in dup.get("keys", []) if k.get("conflicting_keys")):
            score -= 2
    return max(0, min(5, score))


def _temporal(ds: Dataset) -> int:
    cov = ds.temporal.get("coverage")
    if not cov:
        return 0
    freq = cov.get("frequency")
    score = 0
    # credit for a usable temporal resolution even when coverage % is unknown
    if freq in ("MONTHLY", "DAILY", "HOURLY"):
        score = 3
    elif freq in ("QUARTERLY", "WEEKLY"):
        score = 2
    elif freq not in (None, "UNKNOWN", "IRREGULAR"):
        score = 1
    pct = cov.get("coverage_pct")
    if pct is not None:
        if pct >= 99:
            score = max(score, 5)
        elif pct >= 95:
            score = max(score, 4)
        elif pct >= 85:
            score = max(score, 3)
        elif pct >= 70:
            score = max(score, 2)
        elif pct >= 50:
            score = max(score, 1)
    if cov.get("n_duplicate_timestamps", 0) > 0 and not ds.entity_columns:
        score = max(0, score - 1)
    return score


def _geographic(ds: Dataset) -> int:
    if not ds.entity_columns:
        return 0
    score = 2
    for rep in (ds.entity_report or []):
        frac = rep.get("country_match_frac", 0.0)
        if frac >= 0.9:
            score = 5
        elif frac >= 0.5:
            score = max(score, 4)
        if rep.get("aggregates_detected"):
            score = max(1, score - 1)
    return score


def _definition(ds: Dataset) -> int:
    p = ds.purpose.get("purpose")
    c = ds.purpose.get("confidence")
    if p == "UNKNOWN":
        return 0
    if c == "HIGH":
        return 4
    if c == "MEDIUM":
        return 3
    return 2


def _unit_clarity(ds: Dataset) -> int:
    if ds.unit_catalog is None or ds.unit_catalog.empty:
        return 0
    known = (ds.unit_catalog["unit"] != "UNIT_UNKNOWN").mean()
    return max(0, min(5, round(5 * known)))


def _metadata(ds: Dataset) -> int:
    n = len(ds.metadata) if ds.metadata else 0
    if ds.is_scientific:
        n += len(ds.scientific_info.get("attrs", {}))
    if n > 10:
        return 5
    if n > 5:
        return 4
    if n > 2:
        return 3
    if n >= 1:
        return 2
    return 0


def _reproducibility(ds: Dataset) -> int:
    score = 1  # file size + modified date recorded
    if ds.sha256:
        score += 1
    src = identify_source(ds)
    if src["classification"] != "UNKNOWN":
        score += 3
    return max(0, min(5, score))


def score_dataset(ds: Dataset) -> Dict:
    dims = {
        "completeness": _completeness(ds),
        "consistency": _consistency(ds),
        "temporal_quality": _temporal(ds),
        "geographic_quality": _geographic(ds),
        "definition_clarity": _definition(ds),
        "unit_clarity": _unit_clarity(ds),
        "metadata_quality": _metadata(ds),
        "reproducibility": _reproducibility(ds),
    }
    total = sum(dims.values())
    return {"dims": dims, "total": total, "classification": classify_score(total)}


def identify_source(ds: Dataset) -> Dict:
    """Look for provenance signals in metadata/attributes (never fabricated)."""
    hay = {}
    if ds.metadata:
        hay.update(ds.metadata)
    if ds.is_scientific:
        hay.update(ds.scientific_info.get("attrs", {}))
    blob = json.dumps(hay, default=str).lower()
    found = {}
    for key in ("url", "http", "www.", "license", "licence", "source", "institution",
                "provider", "publisher", "doi", "version", "title", "history",
                "references", "contact", "download"):
        if key in blob:
            # extract the actual value for key-ish matches
            for k, v in hay.items():
                if key in str(k).lower() and v not in (None, ""):
                    found[k] = str(v)[:200]
    if found:
        classification = "KNOWN" if any(k in ("source", "url", "doi", "institution", "title") for k in found) else "LIKELY"
    else:
        classification = "UNKNOWN"
    return {"classification": classification, "signals": found,
            "note": "source metadata inferred from file attributes only — verify externally"}


def run_quality_audit(ctx: AuditContext) -> None:
    rows = []
    for ds in ctx.readable_datasets:
        sc = score_dataset(ds)
        ds.quality_scores = sc
        src = identify_source(ds)
        ds.quality_scores["source"] = src
        row = {"dataset": ds.display_name, "file_id": ds.file_id, **sc["dims"],
               "total": sc["total"], "classification": sc["classification"],
               "source_classification": src["classification"]}
        rows.append(row)
    ctx.add_table("13_quality_scores.csv", pd.DataFrame(rows))


def usability_for(ds: Dataset) -> Dict:
    """Classify usability (spec section 26)."""
    sc = ds.quality_scores or {}
    total = sc.get("total", 0)
    dims = sc.get("dims", {})
    reason = []
    if total >= 34 and dims.get("unit_clarity", 0) >= 4 and dims.get("definition_clarity", 0) >= 3:
        label = "DIRECTLY USABLE"
        reason.append("high quality score with clear units and definition")
    elif total >= 20:
        label = "USABLE AFTER HARMONIZATION"
        reason.append("usable but requires harmonization/verification before analysis")
    elif total >= 12:
        label = "REQUIRES EXTERNAL INFORMATION"
        reason.append("weak quality — needs external documentation/validation")
    elif total >= 0:
        label = "INSUFFICIENT EVIDENCE"
        reason.append("too little evidence to establish usability")
    if dims.get("unit_clarity", 0) <= 1:
        reason.append("units unclear (UNIT_UNKNOWN)")
    if dims.get("temporal_quality", 0) == 0:
        reason.append("no usable temporal dimension detected")
    return {"label": label, "reason": "; ".join(reason) or "UNKNOWN"}
