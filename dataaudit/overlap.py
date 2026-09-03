"""Cross-dataset overlap detection (spec section 21)."""
from __future__ import annotations

import re
from typing import List

import pandas as pd

from .core import AuditContext, Dataset


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _tokens(name: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(name).lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dataset_vars(ds: Dataset) -> List[str]:
    if ds.is_tabular and ds.df is not None:
        return [str(c) for c in ds.df.columns]
    if ds.is_scientific:
        return [str(v) for v in ds.scientific_info.get("data_vars", [])]
    return []


def _unit_for(ds: Dataset, var: str) -> str:
    if ds.unit_catalog is not None and len(ds.unit_catalog):
        m = ds.unit_catalog[ds.unit_catalog["variable"] == var]
        if len(m):
            u = str(m.iloc[0]["unit"])
            return "" if u == "UNIT_UNKNOWN" else u
    return ""


def _role_for(ds: Dataset, var: str) -> str:
    if ds.semantic_roles is not None:
        m = ds.semantic_roles[ds.semantic_roles["column"] == var]
        if len(m):
            return str(m.iloc[0]["role"])
    return ""


def run_overlap_detection(ctx: AuditContext) -> None:
    dsets = [d for d in ctx.datasets if d.readable]
    rows = []
    for i in range(len(dsets)):
        for j in range(i + 1, len(dsets)):
            a, b = dsets[i], dsets[j]
            va, vb = _dataset_vars(a), _dataset_vars(b)
            na = {_norm(v): v for v in va}
            nb = {_norm(v): v for v in vb}
            # exact normalized-name matches
            common = set(na) & set(nb)
            for key in sorted(common):
                ua, ub = _unit_for(a, na[key]), _unit_for(b, nb[key])
                units_compatible = (ua == "" or ub == "" or ua == ub)
                confidence = "HIGH" if (ua and ub and ua == ub) else "MEDIUM"
                rows.append({
                    "variable_a": na[key], "dataset_a": a.display_name,
                    "variable_b": nb[key], "dataset_b": b.display_name,
                    "possible_overlap": "YES",
                    "match_kind": "name_exact",
                    "unit_a": ua, "unit_b": ub,
                    "confidence": confidence,
                    "note": "" if units_compatible else "unit mismatch — verify definitions",
                })
            # near matches (token similarity)
            for va_ in va:
                na_ = _norm(va_)
                for vb_ in vb:
                    nb_ = _norm(vb_)
                    if na_ == nb_ or na_ in common or nb_ in common:
                        continue
                    if na_ and nb_ and (na_ in nb_ or nb_ in na_):
                        # one name contains the other (e.g. gdp vs gdp_per_capita)
                        rows.append({
                            "variable_a": va_, "dataset_a": a.display_name,
                            "variable_b": vb_, "dataset_b": b.display_name,
                            "possible_overlap": "POSSIBLE (semantic mismatch risk)",
                            "match_kind": "name_contains",
                            "unit_a": _unit_for(a, va_), "unit_b": _unit_for(b, vb_),
                            "confidence": "LOW",
                            "note": "names overlap but are NOT identical — do not assume equivalence",
                        })
    df = pd.DataFrame(rows)
    ctx.add_table("10_overlap_analysis.csv", df)
