"""Conflict analysis (spec sections 23, 20)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .core import AuditContext, Dataset


def _freq(ds: Dataset) -> str:
    cov = ds.temporal.get("coverage")
    return cov.get("frequency", "UNKNOWN") if cov else "UNKNOWN"


def _measurement_type(ds: Dataset, col: str) -> str:
    if ds.variable_semantics is not None and len(ds.variable_semantics):
        m = ds.variable_semantics[ds.variable_semantics["variable"] == col]
        if len(m):
            return str(m.iloc[0]["measurement_type"])
    return "UNKNOWN"


def classify_conflict_cause(ov, dsA: Optional[Dataset], dsB: Optional[Dataset],
                            colA: str, colB: str) -> str:
    """Heuristically classify the likely cause of a disagreement."""
    if dsA is None or dsB is None:
        return "UNKNOWN"
    fa, fb = _freq(dsA), _freq(dsB)
    if fa != "UNKNOWN" and fb != "UNKNOWN" and fa != fb:
        return "FREQUENCY_MISMATCH"
    mt_a = _measurement_type(dsA, colA)
    mt_b = _measurement_type(dsB, colB)
    if "FORECAST" in mt_a or "FORECAST" in mt_b:
        return "ACTUAL_VS_FORECAST"
    if ("ESTIMATED" in mt_a) != ("ESTIMATED" in mt_b):
        return "ESTIMATE_VS_OBSERVATION"
    if ("MODELED" in mt_a) != ("MODELED" in mt_b):
        return "METHODOLOGY"
    return "DEFINITION_MISMATCH"


def run_conflict_analysis(ctx: AuditContext) -> None:
    rows = []
    # 1) cross-dataset disagreements
    cmp = ctx.get_table("11_cross_dataset_comparison.csv")
    if cmp is not None and not cmp.empty:
        valid = cmp[cmp["valid"] == True]  # noqa: E712
        for _, r in valid.iterrows():
            pct = r.get("mean_pct_diff")
            agree = r.get("agreement_rate")
            if pct is None:
                continue
            disagreement = (agree is not None and agree < 80) or abs(pct) > 5
            if not disagreement:
                continue
            dsA = next((d for d in ctx.datasets if d.display_name == r["dataset_a"]), None)
            dsB = next((d for d in ctx.datasets if d.display_name == r["dataset_b"]), None)
            cause = classify_conflict_cause(r, dsA, dsB, r["variable_a"], r["variable_b"])
            rows.append({
                "scope": "CROSS-DATASET",
                "dataset_a": r["dataset_a"], "variable_a": r["variable_a"],
                "dataset_b": r["dataset_b"], "variable_b": r["variable_b"],
                "n_overlap": r["n_overlap"],
                "mean_pct_diff": r.get("mean_pct_diff"),
                "agreement_rate": r.get("agreement_rate"),
                "correlation": r.get("correlation"),
                "possible_cause": cause,
                "note": "Datasets disagree over overlapping observations. First question: are these really measuring the same thing?",
            })
    # 2) invalid comparisons
    if cmp is not None and not cmp.empty:
        invalid = cmp[cmp["valid"] == False]  # noqa: E712
        for _, r in invalid.iterrows():
            rows.append({
                "scope": "CROSS-DATASET",
                "dataset_a": r["dataset_a"], "variable_a": r["variable_a"],
                "dataset_b": r["dataset_b"], "variable_b": r["variable_b"],
                "n_overlap": 0, "mean_pct_diff": None, "agreement_rate": None,
                "correlation": None, "possible_cause": "COMPARISON_NOT_VALID",
                "note": r["reason"],
            })
    # 3) within-dataset conflicting duplicates
    for ds in ctx.tabular_datasets:
        for k in ds.duplicates.get("keys", []):
            if k.get("conflicting_keys"):
                for ex in k.get("conflict_examples", [])[:5]:
                    rows.append({
                        "scope": "WITHIN-DATASET",
                        "dataset_a": ds.display_name, "variable_a": str(ex.get("key", "")),
                        "dataset_b": "", "variable_b": ", ".join(ex.get("value_columns", [])),
                        "n_overlap": ex.get("n_rows"),
                        "mean_pct_diff": None, "agreement_rate": None,
                        "correlation": None,
                        "possible_cause": "DUPLICATE_CONFLICT",
                        "note": f"Same logical key '{k.get('key', k.get('name', '?'))}' maps to different values.",
                    })
    # 4) semantic mismatch alerts
    for ds in ctx.tabular_datasets:
        if ds.variable_semantics is None or not len(ds.variable_semantics):
            continue
        for _, s in ds.variable_semantics.iterrows():
            if s.get("semantic_alerts"):
                rows.append({
                    "scope": "SEMANTIC",
                    "dataset_a": ds.display_name, "variable_a": s["variable"],
                    "dataset_b": "", "variable_b": "",
                    "n_overlap": None, "mean_pct_diff": None, "agreement_rate": None,
                    "correlation": None, "possible_cause": "DEFINITION_MISMATCH",
                    "note": s["semantic_alerts"],
                })
    ctx.add_table("12_conflict_report.csv", pd.DataFrame(rows))
