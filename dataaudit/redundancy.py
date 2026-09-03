"""Correlation / redundancy analysis (spec section 36)."""
from __future__ import annotations

import pandas as pd

from .core import AuditContext, Dataset
from .utils import is_numeric_dtype, safe_correlation

MAX_VARS = 40


def run_redundancy_analysis(ctx: AuditContext) -> None:
    rows = []
    for ds in ctx.tabular_datasets:
        df = ds.df
        num_cols = [c for c in df.columns if is_numeric_dtype(df[c].dtype)][:MAX_VARS]
        if len(num_cols) < 2:
            continue
        corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(min_periods=10)
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = corr.iloc[i, j]
                if pd.isna(r) or abs(r) < 0.85:
                    continue
                a, b = num_cols[i], num_cols[j]
                # conceptual redundancy heuristic: normalized-name overlap
                na = {_norm(a)} if False else _toks(a)
                nb = _toks(b)
                overlap = len(na & nb) / max(1, len(na | nb))
                if overlap > 0.5:
                    kind = "POSSIBLE REDUNDANCY (conceptual + statistical)"
                else:
                    kind = "HIGH CORRELATION (statistical; may be complementary)"
                rows.append({
                    "dataset": ds.display_name,
                    "file_id": ds.file_id,
                    "variable_a": str(a),
                    "variable_b": str(b),
                    "correlation": round(float(r), 4),
                    "classification": kind,
                    "note": "Correlation does NOT imply equivalence — verify definitions before treating as redundant.",
                })
    ctx.add_table("_redundancy.csv", pd.DataFrame(rows))


def _toks(name: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9]+", str(name).lower()))


def _norm(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
