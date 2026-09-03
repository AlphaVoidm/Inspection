"""Time-series / panel readiness, overall readiness, decision matrix and
recommendations (spec sections 27, 28, 29, 43, 47, 48, 57)."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .core import AuditContext, Dataset
from .utils import is_numeric_dtype


# ---------------------------------------------------------------------------
# Time-series readiness (section 27)
# ---------------------------------------------------------------------------

def ts_readiness_for(ds: Dataset) -> Dict:
    cov = ds.temporal.get("coverage")
    if not cov:
        return {"dataset": ds.display_name, "classification": "NOT APPLICABLE",
                "reason": "no temporal variable detected", "score": 0}
    score = 0
    reasons = []
    freq = cov.get("frequency")
    if freq in ("MONTHLY", "DAILY", "HOURLY"):
        score += 2
    elif freq in ("QUARTERLY", "WEEKLY"):
        score += 1
    else:
        reasons.append(f"frequency {freq} is irregular/unknown")

    coverage = cov.get("coverage_pct")
    if coverage is not None:
        if coverage >= 99:
            score += 2
        elif coverage >= 95:
            score += 1
        else:
            reasons.append(f"coverage {coverage}%")
    dup_ts = cov.get("n_duplicate_timestamps", 0)
    # duplicate timestamps are normal for panel data (one per entity); only
    # penalize when there is no entity dimension.
    if dup_ts > 0 and not ds.entity_columns:
        reasons.append(f"{dup_ts} duplicate timestamps")
    n = cov.get("n_observations", 0)
    if n >= 120:
        score += 1
    elif n < 24:
        reasons.append(f"only {n} observations")

    classification = "HIGH" if score >= 5 else ("MEDIUM" if score >= 3 else "LOW")
    return {"dataset": ds.display_name, "classification": classification,
            "reason": "; ".join(reasons) or "regular, complete, sufficient history",
            "score": score, "frequency": freq, "coverage_pct": coverage,
            "n_observations": n, "start": str(cov.get("start"))[:19], "end": str(cov.get("end"))[:19]}


# ---------------------------------------------------------------------------
# Panel readiness (section 28)
# ---------------------------------------------------------------------------

def panel_readiness_for(ds: Dataset) -> Dict:
    if not ds.entity_columns:
        return {"dataset": ds.display_name, "classification": "NOT APPLICABLE",
                "reason": "no entity column detected", "n_entities": 0, "balanced": None}
    ec = ds.entity_columns[0]
    if ec not in ds.df.columns:
        return {"dataset": ds.display_name, "classification": "NOT APPLICABLE",
                "reason": "entity column not present", "n_entities": 0, "balanced": None}
    n_entities = int(ds.df[ec].nunique())
    counts = ds.df.groupby(ec, dropna=True).size()
    balanced = bool(counts.nunique(dropna=True) <= 1)
    by_entity = ds.temporal.get("by_entity")
    if by_entity is None:
        return {"dataset": ds.display_name, "classification": "MEDIUM",
                "reason": "entities detected but no temporal coverage per entity",
                "n_entities": n_entities, "balanced": balanced,
                "min_history": int(counts.min()), "median_history": int(counts.median()),
                "max_history": int(counts.max())}
    entity_table = by_entity[by_entity["entity_column"] == str(ec)].head(60)
    classification = "HIGH" if (n_entities >= 3 and not balanced is False and
                                by_entity["coverage_pct"].median() >= 95) else "MEDIUM"
    return {
        "dataset": ds.display_name, "classification": classification,
        "reason": f"{n_entities} entities; {'balanced' if balanced else 'unbalanced'} panel",
        "n_entities": n_entities, "balanced": balanced,
        "min_history": int(counts.min()), "median_history": int(counts.median()),
        "max_history": int(counts.max()), "entity_table": entity_table,
    }


def run_readiness_audit(ctx: AuditContext) -> None:
    ts_rows, panel_rows = [], []
    for ds in ctx.readable_datasets:
        if ds.is_tabular:
            ds.ts_readiness = ts_readiness_for(ds)
            ds.panel_readiness = panel_readiness_for(ds)
        else:
            ds.ts_readiness = {"dataset": ds.display_name, "classification": "NOT APPLICABLE",
                               "reason": "non-tabular dataset", "score": 0,
                               "frequency": None, "coverage_pct": None,
                               "n_observations": None, "start": None, "end": None}
            ds.panel_readiness = {"dataset": ds.display_name, "classification": "NOT APPLICABLE",
                                  "reason": "non-tabular dataset", "n_entities": 0, "balanced": None}
        ts_rows.append({**ds.ts_readiness, "file_id": ds.file_id})
        panel_rows.append({"file_id": ds.file_id, "dataset": ds.display_name,
                           "classification": ds.panel_readiness["classification"],
                           "n_entities": ds.panel_readiness.get("n_entities"),
                           "balanced": ds.panel_readiness.get("balanced"),
                           "min_history": ds.panel_readiness.get("min_history"),
                           "median_history": ds.panel_readiness.get("median_history"),
                           "max_history": ds.panel_readiness.get("max_history"),
                           "reason": ds.panel_readiness.get("reason")})
    ctx.add_table("_ts_readiness.csv", pd.DataFrame(ts_rows))
    ctx.add_table("_panel_readiness.csv", pd.DataFrame(panel_rows))


# ---------------------------------------------------------------------------
# Overall readiness + recommendations (sections 25, 26, 43, 47, 48)
# ---------------------------------------------------------------------------

def overall_readiness(ctx: AuditContext) -> str:
    q = ctx.get_table("13_quality_scores.csv")
    if q is None or q.empty:
        return "INSUFFICIENT EVIDENCE"
    n_strong = int((q["total"] >= 28).sum())
    n_usable = int((q["total"] >= 20).sum())
    n = len(q)
    if n == 0:
        return "INSUFFICIENT EVIDENCE"
    if n_usable == n and n_strong >= 1:
        return "READY"
    if n_usable >= 1:
        return "PARTIALLY READY"
    return "NOT READY"


def build_recommendations(ctx: AuditContext) -> pd.DataFrame:
    recs = []
    q = ctx.get_table("13_quality_scores.csv")
    if q is not None and not q.empty:
        q = q.sort_values("total", ascending=False)
        best = q.iloc[0]
        recs.append({
            "category": "Strongest dataset",
            "recommendation": f"{best['dataset']} has the highest quality score ({best['total']}/40, {best['classification']}).",
            "evidence": f"quality={best['total']}, completeness={best['completeness']}, temporal={best['temporal_quality']}, units={best['unit_clarity']}.",
            "confidence": "HIGH",
        })
        worst = q.iloc[-1]
        if worst["total"] < 20:
            recs.append({
                "category": "Weakest dataset",
                "recommendation": f"Investigate {worst['dataset']} before use (score {worst['total']}/40).",
                "evidence": f"quality={worst['total']}, classification={worst['classification']}.",
                "confidence": "HIGH",
            })
    # temporal gaps
    tc = ctx.get_table("04_temporal_coverage.csv")
    if tc is not None and not tc.empty:
        gap = tc[tc["n_gaps"] > 0].sort_values("n_gaps", ascending=False)
        if len(gap):
            r = gap.iloc[0]
            recs.append({
                "category": "Temporal gap",
                "recommendation": f"Review temporal gaps in {r['dataset']} (column {r['column']}).",
                "evidence": f"{r['n_gaps']} missing periods, longest gap {r['longest_gap']}, coverage {r['coverage_pct']}%.",
                "confidence": "HIGH",
            })
    # conflicts
    cf = ctx.get_table("12_conflict_report.csv")
    if cf is not None and not cf.empty:
        n_conf = int((cf["scope"] == "CROSS-DATASET").sum())
        if n_conf:
            recs.append({
                "category": "Conflict",
                "recommendation": "Resolve cross-dataset disagreements before modeling.",
                "evidence": f"{n_conf} cross-dataset conflict(s) detected — first verify whether both sources measure the same thing.",
                "confidence": "MEDIUM",
            })
    # leakage
    lk = ctx.get_table("15_leakage_flags.csv")
    if lk is not None and not lk.empty:
        recs.append({
            "category": "Leakage",
            "recommendation": "Exclude or quarantine leakage-flagged variables from feature engineering.",
            "evidence": f"{len(lk)} variable(s) flagged by name for future/derived information.",
            "confidence": "MEDIUM",
        })
    if not recs:
        recs.append({
            "category": "General",
            "recommendation": "No strong automated flags — proceed with manual review of variable definitions.",
            "evidence": "UNKNOWN — insufficient evidence in the inspected data.",
            "confidence": "UNKNOWN",
        })
    return pd.DataFrame(recs)


def build_decision_matrix(ctx: AuditContext) -> pd.DataFrame:
    proj = ctx.config.resolve_project()
    if not (proj["target"] or proj["candidate_features"]):
        return pd.DataFrame()
    rows = []
    # target availability
    for t in proj["target"]:
        rows.append(_decision_row(ctx, t, kind="Target"))
    if proj["required_frequency"]:
        rows.append(_decision_row(ctx, None, kind="frequency"))
    for f in proj["candidate_features"]:
        rows.append(_decision_row(ctx, f, kind="Feature"))
    return pd.DataFrame(rows)


def _decision_row(ctx: AuditContext, term, kind: str) -> dict:
    if kind == "frequency":
        return {"requirement": "Monthly frequency", "available": "UNKNOWN",
                "coverage": None, "quality": None, "problem": None,
                "confidence": "UNKNOWN",
                "recommendation": "verify per-dataset temporal frequency in the coverage report"}
    from .features import _feature_match
    best = None
    for ds in ctx.tabular_datasets:
        col = _feature_match(term, [str(c) for c in ds.df.columns])
        if col is None:
            continue
        freq = ds.temporal["coverage"]["frequency"] if ds.temporal.get("coverage") else None
        start = str(ds.temporal["coverage"]["start"])[:19] if ds.temporal.get("coverage") else None
        end = str(ds.temporal["coverage"]["end"])[:19] if ds.temporal.get("coverage") else None
        q = ds.quality_scores.get("total") if ds.quality_scores else None
        cand = {"dataset": ds.display_name, "col": col, "freq": freq,
                "start": start, "end": end, "quality": q}
        if best is None or (q is not None and (best["quality"] is None or q > best["quality"])):
            best = cand
    if best is not None:
        return {
            "requirement": term, "available": "YES",
            "coverage": f"{best['start']} → {best['end']}",
            "quality": best["quality"], "problem": None,
            "confidence": "HIGH",
            "recommendation": f"available as '{best['col']}' in {best['dataset']} "
                              f"({best['freq'] or 'unknown'} frequency)",
        }
    return {"requirement": term, "available": "NO", "coverage": None, "quality": None,
            "problem": "no matching variable", "confidence": "MEDIUM",
            "recommendation": "locate an external source or derive from available data"}


# ---------------------------------------------------------------------------
# HGT-QF readiness (sections 46, 57)
# ---------------------------------------------------------------------------

def hgtqf_readiness(ctx: AuditContext) -> Dict:
    proj = ctx.config.resolve_project()
    out: Dict = {"enabled": bool(proj["target"])}
    q = ctx.get_table("13_quality_scores.csv")
    feats = ctx.get_table("14_feature_availability.csv")
    tc = ctx.get_table("04_temporal_coverage.csv")

    out["targets"] = []
    for t in proj["target"]:
        m = feats[feats["feature"] == t] if feats is not None and not feats.empty else pd.DataFrame()
        if len(m):
            r = m.iloc[0]
            out["targets"].append({
                "target": t, "dataset": r["dataset"], "frequency": r["frequency"],
                "start": r["start"], "end": r["end"], "entities": r["entities"],
                "missing_pct": r["missing_pct"], "quality": r["quality"],
            })
    # maximum defensible history
    max_history = 0
    best_freq = None
    if tc is not None and not tc.empty:
        for _, r in tc.iterrows():
            n = r.get("n_observations") or 0
            if n > max_history:
                max_history = int(n)
                best_freq = r.get("frequency")
    out["max_observations"] = max_history
    out["best_frequency"] = best_freq
    out["n_countries"] = None
    out["median_coverage_pct"] = None
    # count countries with sufficient history (country-like entity columns only)
    by_ent = []
    for ds in ctx.tabular_datasets:
        if ds.temporal.get("by_entity") is None or ds.semantic_roles is None:
            continue
        country_cols = set(
            str(r["column"]) for _, r in ds.semantic_roles.iterrows()
            if r.get("role") in ("COUNTRY", "COUNTRY_CODE", "ISO2", "ISO3"))
        et = ds.temporal["by_entity"]
        sub = et[et["entity_column"].astype(str).isin(country_cols)]
        if not sub.empty:
            by_ent.append(sub)
    if by_ent:
        ent = pd.concat(by_ent, ignore_index=True)
        if not ent.empty:
            out["n_countries"] = int((ent["observations"] >= 120).sum())
            out["median_coverage_pct"] = round(float(ent["coverage_pct"].median()), 2)
    return out
