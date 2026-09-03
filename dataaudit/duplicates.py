"""Duplicate audit: exact duplicates, key duplicates, conflicting duplicates
(spec section 14)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset
from .temporal import detect_temporal_columns, infer_frequency, _period_key
from .utils import is_numeric_dtype


def _candidate_keys(ds: Dataset) -> List[dict]:
    """Propose logical key combinations (entity + time)."""
    keys = []
    entity_cols = [c for c in ds.entity_columns if ds.df[c].nunique() > 1]
    time_cols = [t["column"] for t in detect_temporal_columns(ds)]
    for ec in entity_cols[:2]:
        for tc in time_cols[:2]:
            keys.append({"name": f"{ec} + {tc}", "cols": [ec, tc], "kind": "entity+time"})
    # country + year if separate year column exists
    for ec in entity_cols[:2]:
        for c in ds.df.columns:
            if str(c).lower().strip() in ("year", "yr", "jahr", "anno", "annee"):
                keys.append({"name": f"{ec} + {c}", "cols": [ec, str(c)], "kind": "entity+year"})
                break
    # all-id columns
    for c in ds.df.columns:
        if str(c).lower().strip() in ("id", "code", "key", "uuid", "identifier", "station_id", "plant_id"):
            keys.append({"name": f"{c}", "cols": [str(c)], "kind": "identifier"})
            break
    return keys


def run_duplicate_audit(ctx: AuditContext) -> None:
    global_rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        dup_mask = df.duplicated()
        n_exact = int(dup_mask.sum())
        exact_sample = df[dup_mask].head(5).astype(str).to_dict("records") if n_exact else []

        key_rows = []
        for key in _candidate_keys(ds):
            cols = [c for c in key["cols"] if c in df.columns]
            if len(cols) < 1:
                continue
            if len(cols) == 1:
                kdf = df[cols[0]].astype(str)
            else:
                kdf = df[cols].astype(str).agg(" | ".join, axis=1)
            n_dup_keys = int(kdf.duplicated().sum())
            n_conflict = None
            conflict_examples = []
            if n_dup_keys > 0:
                # conflicting: same key, differing values in other columns
                grp = df.groupby(kdf, dropna=True)
                conflict_count = 0
                for _, sub in grp:
                    if len(sub) < 2:
                        continue
                    others = [c for c in df.columns if c not in cols]
                    if others:
                        variations = sub[others].nunique()
                        if variations.max() > 1:
                            conflict_count += 1
                            if len(conflict_examples) < 3:
                                conflict_examples.append({
                                    "key": sub[cols].iloc[0].to_dict(),
                                    "n_rows": len(sub),
                                    "value_columns": [c for c in others if variations[c] > 1][:8],
                                    "sample": sub.head(3).astype(str).to_dict("records"),
                                })
                n_conflict = conflict_count
            key_rows.append({
                "dataset": ds.display_name,
                "file_id": ds.file_id,
                "key": key["name"],
                "key_kind": key["kind"],
                "duplicate_keys": n_dup_keys,
                "conflicting_keys": n_conflict,
                "conflict_examples": conflict_examples if n_conflict else [],
            })

        ds.duplicates = {
            "n_exact_duplicates": n_exact,
            "exact_duplicate_pct": round(100.0 * n_exact / len(df), 4) if len(df) else 0.0,
            "exact_sample": exact_sample,
            "keys": key_rows,
        }
        global_rows.append({
            "dataset": ds.display_name,
            "file_id": ds.file_id,
            "exact_duplicate_rows": n_exact,
            "exact_duplicate_pct": ds.duplicates["exact_duplicate_pct"],
            "duplicate_keys_found": sum(1 for k in key_rows if k["duplicate_keys"] > 0),
            "conflicting_keys_found": sum(1 for k in key_rows if k.get("conflicting_keys")),
        })
    ctx.add_table("07_duplicate_report.csv", pd.DataFrame(global_rows))
