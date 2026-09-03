"""Export audit artifacts: CSV tables, Markdown reports, figures
(spec sections 40, 42, 53)."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Dataset, DEFAULT_EXPORTS
from .entities import country_mapping_table
from .quality import BANDS, classify_score
from .utils import fmt_bytes, markdown_table, now_iso, truncate_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Directory scaffolding
# ---------------------------------------------------------------------------

def _ensure_dirs(ctx: AuditContext) -> dict:
    out = ctx.config.output_dir
    dirs = {
        "root": out,
        "reports": os.path.join(out, "reports"),
        "fig_missing": os.path.join(out, "figures", "missingness"),
        "fig_coverage": os.path.join(out, "figures", "coverage"),
        "fig_dist": os.path.join(out, "figures", "distributions"),
        "fig_ts": os.path.join(out, "figures", "time_series"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

EXPORT_ALIASES = {
    "18_entity_temporal.csv": "_entity_temporal.csv",
    "19_ts_readiness.csv": "_ts_readiness.csv",
    "20_panel_readiness.csv": "_panel_readiness.csv",
    "21_decision_matrix.csv": "_decision_matrix.csv",
    "22_dataset_purpose.csv": "_purpose.csv",
    "23_variable_semantics.csv": "_variable_semantics.csv",
    "24_redundancy.csv": "_redundancy.csv",
    "26_annual_monthly_flags.csv": "_annual_monthly_flags.csv",
}


def _safe_to_csv(df: Optional[pd.DataFrame], path: str) -> None:
    """Write a DataFrame to CSV, always producing a readable file (even when
    the table has zero rows/columns)."""
    if df is None:
        return
    if df.empty and df.columns.empty:
        with open(path, "w", encoding="utf-8") as f:
            f.write("note\n(no records)\n")
    else:
        df.to_csv(path, index=False)


def export_csv(ctx: AuditContext) -> List[str]:
    written = []
    # numbered primary tables
    for name in DEFAULT_EXPORTS:
        df = ctx.get_table(name)
        if df is None:
            continue
        target = os.path.join(ctx.config.output_dir, name)
        _safe_to_csv(df, target)
        written.append(target)
    # country normalization table
    cmap = country_mapping_table(ctx)
    if cmap is not None and not cmap.empty:
        target = os.path.join(ctx.config.output_dir, "25_country_normalization.csv")
        cmap.to_csv(target, index=False)
        written.append(target)
    # aliased internal tables
    for out_name, key in EXPORT_ALIASES.items():
        df = ctx.get_table(key)
        if df is not None:
            target = os.path.join(ctx.config.output_dir, out_name)
            _safe_to_csv(df, target)
            written.append(target)
    return written


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------

def _head(df: Optional[pd.DataFrame], n: int = 50) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(n)


def build_reports(ctx: AuditContext) -> dict:
    reports = {}

    # ---- DATASET_SUMMARY.md ----
    inv = ctx.get_table("01_file_inventory.csv")
    prof = ctx.get_table("02_dataset_profiles.csv")
    purpose = ctx.get_table("_purpose.csv")
    lines = ["# DATASET SUMMARY", "",
             f"Audit timestamp: {ctx.config.audit_timestamp}",
             f"Input paths: {len(ctx.input_paths)}",
             f"Files discovered: {len(ctx.discovered_files)}",
             f"Readable datasets: {len(ctx.readable_datasets)}", ""]
    if inv is not None and not inv.empty:
        lines += ["## FILE INVENTORY", "", markdown_table(_head(inv, 100)), ""]
    if purpose is not None and not purpose.empty:
        lines += ["## DATASET PURPOSE INFERENCE", "",
                  "Purpose is inferred from filenames, column names and metadata "
                  "keywords — never from the filename alone, and never certain.",
                  "", markdown_table(purpose), ""]
    reports["DATASET_SUMMARY.md"] = "\n".join(lines)

    # ---- VARIABLE_CATALOG.md ----
    vcat = ctx.get_table("03_variable_catalog.csv")
    lines = ["# VARIABLE CATALOG", "",
             "Automated role inference with confidence labels. "
             "Inferred roles are NOT certain.", ""]
    if vcat is not None and not vcat.empty:
        lines += [markdown_table(_head(vcat, 300)), ""]
    reports["VARIABLE_CATALOG.md"] = "\n".join(lines)

    # ---- COVERAGE_REPORT.md ----
    tc = ctx.get_table("04_temporal_coverage.csv")
    ec = ctx.get_table("05_entity_coverage.csv")
    et = ctx.get_table("_entity_temporal.csv")
    lines = ["# COVERAGE REPORT", ""]
    if tc is not None and not tc.empty:
        lines += ["## TEMPORAL COVERAGE", "", markdown_table(_head(tc, 100)), ""]
    if ec is not None and not ec.empty:
        lines += ["## ENTITY COVERAGE", "", markdown_table(_head(ec, 100)), ""]
    if et is not None and not et.empty:
        lines += ["## ENTITY × TIME COVERAGE", "", markdown_table(_head(et, 120)), ""]
    reports["COVERAGE_REPORT.md"] = "\n".join(lines)

    # ---- QUALITY_REPORT.md ----
    q = ctx.get_table("13_quality_scores.csv")
    lines = ["# QUALITY REPORT", "", "Scoring rubric (8 dimensions, 0–5 each, total /40):", "",
             "- Completeness /5", "- Consistency /5", "- Temporal quality /5",
             "- Geographic quality /5", "- Definition clarity /5", "- Unit clarity /5",
             "- Metadata quality /5", "- Reproducibility /5", "",
             "| Band | Label |", "| --- | --- |"]
    for lo, label in BANDS:
        lines.append(f"| {lo}–{lo + 6 if lo > 0 else 11} | {label} |")
    lines.append("")
    if q is not None and not q.empty:
        lines += [markdown_table(_head(q, 100)), ""]
    lines += ["**The score does NOT automatically determine whether a dataset should be used.**",
              "Scientific suitability is evaluated separately.", ""]
    reports["QUALITY_REPORT.md"] = "\n".join(lines)

    # ---- OVERLAP_REPORT.md ----
    ov = ctx.get_table("10_overlap_analysis.csv")
    cmp_ = ctx.get_table("11_cross_dataset_comparison.csv")
    lines = ["# OVERLAP REPORT", "",
             "Name matches alone never imply identical variables.", ""]
    if ov is not None and not ov.empty:
        lines += ["## DETECTED OVERLAPS", "", markdown_table(_head(ov, 100)), ""]
    if cmp_ is not None and not cmp_.empty:
        lines += ["## CROSS-DATASET COMPARISON", "", markdown_table(_head(cmp_, 100)), ""]
    reports["OVERLAP_REPORT.md"] = "\n".join(lines)

    # ---- CONFLICT_REPORT.md ----
    cf = ctx.get_table("12_conflict_report.csv")
    lines = ["# CONFLICT REPORT", "",
             "Disagreement does NOT mean one source is wrong. "
             "First question: are these actually measuring the same thing?", ""]
    if cf is not None and not cf.empty:
        lines += [markdown_table(_head(cf, 200)), ""]
    reports["CONFLICT_REPORT.md"] = "\n".join(lines)

    # ---- RECOMMENDATIONS.md ----
    rec = ctx.get_table("17_recommendation_table.csv")
    dm = ctx.get_table("_decision_matrix.csv")
    lines = ["# RECOMMENDATIONS", "", "Evidence-based; no recommendation is asserted "
             "without observed evidence.", ""]
    if rec is not None and not rec.empty:
        lines += [markdown_table(rec), ""]
    if dm is not None and not dm.empty:
        lines += ["## FINAL DECISION MATRIX", "", markdown_table(dm), ""]
    reports["RECOMMENDATIONS.md"] = "\n".join(lines)

    # ---- DATA_AUDIT_REPORT.md (main) ----
    reports["DATA_AUDIT_REPORT.md"] = build_main_report(ctx)

    return reports


def build_main_report(ctx: AuditContext) -> str:
    L = []
    hr = "=" * 60
    L.append(hr)
    L.append("GENERAL DATA AUDIT")
    L.append(hr)
    L.append("")
    inv = ctx.get_table("01_file_inventory.csv")
    q = ctx.get_table("13_quality_scores.csv")
    tc = ctx.get_table("04_temporal_coverage.csv")
    freqs = []
    date_range = "N/A"
    if tc is not None and not tc.empty:
        freqs = [str(f) for f in tc["frequency"].dropna().unique()]
        starts = [s for s in tc["start"].dropna() if str(s) not in ("NaT", "None", "nan")]
        ends = [e for e in tc["end"].dropna() if str(e) not in ("NaT", "None", "nan")]
        if starts and ends:
            date_range = f"{min(starts)} → {max(ends)}"

    n_files = len(ctx.discovered_files)
    n_read = len(ctx.readable_datasets)
    n_bad = sum(1 for r in ctx.discovered_files if r["status"] in ("ERROR", "UNSUPPORTED", "NOT FOUND"))
    vcat = ctx.get_table("03_variable_catalog.csv")
    total_vars = len(vcat) if vcat is not None else 0
    ec = ctx.get_table("05_entity_coverage.csv")
    total_entities = None
    if ec is not None and not ec.empty:
        total_entities = int(ec["n_unique"].sum())

    L += [f"Files inspected: {n_files}",
          f"Datasets identified: {n_read}",
          f"Readable: {n_read}",
          f"Unreadable/error: {n_bad}",
          f"Total variables: {total_vars}",
          f"Total entity values (sum across datasets): {total_entities if total_entities is not None else 'N/A'}",
          f"Date range: {date_range}",
          f"Detected frequencies: {', '.join(freqs) if freqs else 'NONE'}",
          ""]
    overall = ctx.get_table("_overall_readiness.csv")
    if overall is not None and not overall.empty:
        L += [f"Overall data readiness: {overall.iloc[0]['classification']}", ""]

    L += ["KEY FINDINGS", "-" * 60]
    findings = _key_findings(ctx)
    for i, f in enumerate(findings, 1):
        L.append(f"{i}. {f}")
    L.append("")

    # per-dataset summary
    L.append("DATASET DETAIL")
    L.append("-" * 60)
    for d in ctx.readable_datasets:
        L.append(f"\n### {d.display_name}")
        L.append(f"- type={d.file_type}, status={d.status}, mode={d.inspection_mode}")
        if d.is_tabular:
            L.append(f"- shape: {d.row_count_display()} rows × {d.n_cols} columns")
        p = d.purpose or {}
        L.append(f"- inferred purpose: {p.get('purpose')} ({p.get('confidence')})")
        sc = d.quality_scores or {}
        L.append(f"- quality: {sc.get('total')}/40 ({sc.get('classification')})")
        us = d.usability or {}
        if us:
            L.append(f"- usability: {us.get('label')} — {us.get('reason')}")
        tsr = d.ts_readiness or {}
        L.append(f"- time-series readiness: {tsr.get('classification')} ({tsr.get('reason')})")
        pr = d.panel_readiness or {}
        if pr.get("n_entities"):
            L.append(f"- panel readiness: {pr.get('classification')} — {pr.get('n_entities')} entities, {'balanced' if pr.get('balanced') else 'unbalanced'}")
        if d.errors:
            L.append(f"- ERRORS: {len(d.errors)}")
    L.append("")

    # reproducibility
    L.append("REPRODUCIBILITY")
    L.append("-" * 60)
    env = ctx.env
    L.append(f"- audit_timestamp: {ctx.config.audit_timestamp}")
    L.append(f"- python: {env.get('python_version')}")
    libs = ", ".join(f"{k}={v}" for k, v in env.items() if k not in ("python_version", "python_executable"))
    L.append(f"- libraries: {libs}")
    L.append(f"- output_dir: {ctx.config.output_dir}")
    L.append(f"- config: data_paths={ctx.config.data_paths}, recursive={ctx.config.recursive}")
    L.append("")
    L.append("CONFIDENCE LEGEND")
    L.append("-" * 60)
    L.append("HIGH CONFIDENCE = directly supported by data/metadata.")
    L.append("MEDIUM CONFIDENCE = strong inference, some uncertainty remains.")
    L.append("LOW CONFIDENCE = weak inference.")
    L.append("UNKNOWN = cannot determine from available information.")
    L.append("")

    # HGT-QF decision if enabled
    hgt = ctx.get_table("_hgtqf.csv")
    if hgt is not None and not hgt.empty:
        row = hgt.iloc[0]
        if row.get("enabled"):
            L.append(build_hgtqf_section(ctx))
    return "\n".join(L)


def _key_findings(ctx: AuditContext) -> List[str]:
    out = []
    q = ctx.get_table("13_quality_scores.csv")
    if q is not None and not q.empty:
        q = q.sort_values("total", ascending=False)
        out.append(f"Strongest dataset: {q.iloc[0]['dataset']} ({q.iloc[0]['total']}/40).")
        if q.iloc[-1]['total'] < 20:
            out.append(f"Weakest dataset: {q.iloc[-1]['dataset']} ({q.iloc[-1]['total']}/40).")
    m = ctx.get_table("06_missingness.csv")
    if m is not None and not m.empty:
        worst = m.sort_values("total_missing_pct", ascending=False).iloc[0]
        out.append(f"Severe missingness: {worst['dataset']} ({worst['total_missing_pct']}% cells missing).")
    tc = ctx.get_table("04_temporal_coverage.csv")
    if tc is not None and not tc.empty:
        gaps = tc[tc["n_gaps"] > 0]
        if len(gaps):
            r = gaps.sort_values("n_gaps", ascending=False).iloc[0]
            out.append(f"Major temporal gap: {r['dataset']} has {r['n_gaps']} missing periods (longest {r['longest_gap']}).")
    ov = ctx.get_table("10_overlap_analysis.csv")
    if ov is not None and not ov.empty:
        out.append(f"Potential overlaps: {len(ov[ov['match_kind']=='name_exact'])} exact-name overlaps, "
                   f"{len(ov[ov['match_kind']=='name_contains'])} name-similarity flags.")
    cf = ctx.get_table("12_conflict_report.csv")
    if cf is not None and not cf.empty:
        out.append(f"Potential conflicts: {len(cf)} conflict record(s).")
    lk = ctx.get_table("15_leakage_flags.csv")
    if lk is not None and not lk.empty:
        out.append(f"Potential leakage: {len(lk)} variable(s) flagged.")
    if not out:
        out.append("No strong automated findings — manual review required.")
    out.append("Important unknowns: variable definitions/units for columns marked UNIT_UNKNOWN or confidence UNKNOWN.")
    return out


def build_hgtqf_section(ctx: AuditContext) -> str:
    from .readiness import hgtqf_readiness
    info = hgtqf_readiness(ctx)
    L = []
    L.append("")
    L.append("=" * 60)
    L.append("HGT-QF DATA READINESS DECISION")
    L.append("=" * 60)
    targets = info.get("targets", [])
    if targets:
        best = max(targets, key=lambda t: t.get("quality") or 0)
        L.append("1. CAN THE PROJECT BE BUILT WITH CURRENT DATA?")
        L.append("   " + ("PARTIALLY" if best.get("quality", 0) >= 20 else "NO"))
        L.append(f"2. BEST CANDIDATE TARGET: {best.get('target')} in {best.get('dataset')}")
        L.append(f"3. TARGET COVERAGE: {best.get('start')} → {best.get('end')} "
                 f"({best.get('frequency')}, {best.get('entities')} entities, "
                 f"{best.get('missing_pct')}% missing)")
    L.append(f"8. NUMBER OF COUNTRIES WITH SUFFICIENT HISTORY: {info.get('n_countries')}")
    L.append(f"9. MAXIMUM DEFENSIBLE HISTORICAL PERIOD: {info.get('max_observations')} observations")
    L.append(f"10. REALISTIC FREQUENCY: {info.get('best_frequency')}")
    L.append("11–17. Horizon/LOCO/zero-shot/scenario readiness: UNKNOWN — insufficient evidence "
             "in the inspected data unless explicit project data is provided.")
    L.append("18. BIGGEST DATA PROBLEM: see KEY FINDINGS / conflict report.")
    L.append("19. MOST IMPORTANT MISSING DATA: see feature availability matrix (available=NO).")
    L.append("20. MOST IMPORTANT NEXT ACTION: verify variable definitions and units before any modeling.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def build_figures(ctx: AuditContext, dirs: dict) -> List[str]:
    written = []
    if not ctx.config.write_figures:
        return written
    for d in ctx.tabular_datasets:
        df = d.df
        if df is None or df.empty:
            continue
        try:
            # missingness bar
            miss = d.missingness.get("per_variable") if d.missingness else None
            if miss is not None and not miss.empty and miss["missing_pct"].max() > 0:
                top = miss.sort_values("missing_pct", ascending=False).head(30)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(top["column"].astype(str)[::-1], top["missing_pct"][::-1])
                ax.set_xlabel("missing %")
                ax.set_title(f"Missingness — {d.display_name}")
                fig.tight_layout()
                p = os.path.join(dirs["fig_missing"], f"{d.file_id}_missingness.png")
                fig.savefig(p, dpi=100)
                plt.close(fig)
                written.append(p)
            # coverage matrix
            cm = d.missingness.get("coverage_matrix") if d.missingness else None
            if cm is not None and not cm.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(cm.notna().astype(int), aspect="auto", cmap="Greys", interpolation="nearest")
                ax.set_title(f"Coverage matrix — {d.display_name}")
                ax.set_xlabel("period"); ax.set_ylabel("entity")
                fig.tight_layout()
                p = os.path.join(dirs["fig_coverage"], f"{d.file_id}_coverage.png")
                fig.savefig(p, dpi=100)
                plt.close(fig)
                written.append(p)
            # distribution of top numeric columns
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)][:4]
            if num_cols:
                fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 2.5 * len(num_cols)))
                if len(num_cols) == 1:
                    axes = [axes]
                for ax, c in zip(axes, num_cols):
                    s = pd.to_numeric(df[c], errors="coerce").dropna()
                    ax.hist(s, bins=50, color="steelblue")
                    ax.set_title(c)
                fig.tight_layout()
                p = os.path.join(dirs["fig_dist"], f"{d.file_id}_distributions.png")
                fig.savefig(p, dpi=100)
                plt.close(fig)
                written.append(p)
            # time series plot of primary value
            cov = d.temporal.get("coverage") if d.temporal else None
            if cov:
                tcol = cov.get("column")
                val = None
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c].dtype) and str(c) != tcol:
                        val = c
                        break
                if val and tcol in df.columns:
                    t = pd.to_datetime(df[tcol], errors="coerce")
                    v = pd.to_numeric(df[val], errors="coerce")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(t, v, lw=0.7, color="steelblue")
                    ax.set_title(f"{d.display_name}: {val}")
                    fig.tight_layout()
                    p = os.path.join(dirs["fig_ts"], f"{d.file_id}_{val[:20]}.png")
                    fig.savefig(p, dpi=100)
                    plt.close(fig)
                    written.append(p)
        except Exception:
            continue
    return written


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_all(ctx: AuditContext) -> dict:
    dirs = _ensure_dirs(ctx)
    result = {"csv": [], "reports": [], "figures": []}
    # usability (depends on quality) — compute now for report
    from .quality import usability_for
    for d in ctx.readable_datasets:
        d.usability = usability_for(d)

    if ctx.config.write_csv:
        result["csv"] = export_csv(ctx)
    if ctx.config.write_markdown:
        reports = build_reports(ctx)
        for name, content in reports.items():
            target = os.path.join(dirs["reports"], name)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            result["reports"].append(target)
    result["figures"] = build_figures(ctx, dirs)

    # reproducibility log
    log_path = os.path.join(dirs["root"], "00_audit_log.txt")
    lines = [f"audit_timestamp={ctx.config.audit_timestamp}",
             f"python={ctx.env.get('python_version')}",
             f"input_paths={ctx.config.data_paths}",
             f"recursive={ctx.config.recursive}",
             f"output_dir={ctx.config.output_dir}"]
    for d in ctx.datasets:
        lines.append(f"FILE {d.file_id}: {d.rel_path} size={fmt_bytes(d.size_bytes)} "
                     f"sha256={d.sha256} mode={d.inspection_mode} status={d.status}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    result["csv"].append(log_path)
    return result
