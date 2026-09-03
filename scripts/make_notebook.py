"""Generate data_audit.ipynb from a structured cell list.

Run with:  python scripts/make_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# GENERAL-PURPOSE MULTI-SOURCE DATA INSPECTION & AUDIT NOTEBOOK

This notebook **inspects, profiles, compares, validates, documents and diagnoses**
your datasets — it does **not** clean, merge, impute, or modify the raw data.

**You only need to edit the CONFIGURATION cell below** (data paths). Everything
else runs automatically.

The audit answers: *what do I actually have, what does it mean, how good is it,
how complete is it, what overlaps, what conflicts, what is missing, and what is
usable?*""")

code(r'''# ============================================================
# 01. CONFIGURATION — EDIT THIS SECTION ONLY
# ============================================================
# Point DATA_PATHS at your files and/or directories. Directories are
# searched recursively when RECURSIVE=True.

DATA_PATHS = [
    # r"path/to/dataset_1.csv",
    # r"path/to/dataset_2.xlsx",
    # r"path/to/dataset_3.parquet",
    # r"path/to/dataset_directory/",
]

OUTPUT_DIR = "data_audit_output"
RECURSIVE = True

# ------------------------------------------------------------------
# OPTIONAL project-specific requirements (general-purpose if None).
# When provided, the audit adds a decision matrix and (for
# electricity-demand style targets) an HGT-QF readiness assessment.
# ------------------------------------------------------------------
PROJECT_CONFIG = None
# PROJECT_CONFIG = {
#     "target": ["electricity demand"],
#     "required_frequency": "monthly",
#     "required_entity_level": "country",
#     "candidate_features": ["temperature", "GDP", "population", "renewable share"],
#     "forecast_horizons": [12, 36, 60],
#     "context_windows": [24, 60, 120],
#     "quantile_forecasting": True,
#     "cross_country_modeling": True,
#     "loco_readiness": True,
#     "zero_shot_country": True,
#     "scenario_readiness": True,
# }

# ------------------------------------------------------------------
# Memory / sampling controls (large files are sampled, not fully loaded).
# ------------------------------------------------------------------
MAX_ROWS_FULL_LOAD = 2_000_000   # rows beyond which we sample
SAMPLE_ROWS = 100_000            # sample size for large files

# ------------------------------------------------------------------
# Report toggles
# ------------------------------------------------------------------
WRITE_FIGURES = True
WRITE_MARKDOWN = True
WRITE_CSV = True''')

md("""## 02. Imports & audit initialization

Loads the `dataaudit` toolkit (a local package in this repository) and
initializes the audit context.""")

code(r'''from dataaudit import init_audit, run_all, run_stage
from dataaudit.core import Config
from dataaudit.utils import display, markdown_table
import pandas as pd

config = Config(
    data_paths=DATA_PATHS,
    output_dir=OUTPUT_DIR,
    recursive=RECURSIVE,
    max_rows_full_load=MAX_ROWS_FULL_LOAD,
    sample_rows=SAMPLE_ROWS,
    project_config=PROJECT_CONFIG,
    write_figures=WRITE_FIGURES,
    write_markdown=WRITE_MARKDOWN,
    write_csv=WRITE_CSV,
)
ctx = init_audit(config)
print("Audit context initialized.")
print(f"Audit timestamp: {ctx.config.audit_timestamp}")''')

md("""## 03–04. File Discovery & Inventory

Discovers files under `DATA_PATHS`, classifies their type, checks readability,
expands archives, and builds the FILE INVENTORY.""")

code(r'''run_stage(ctx, "discover")    # 03. File Discovery
run_stage(ctx, "inventory")   # 04. File Inventory
display(markdown_table(ctx.get_table("01_file_inventory.csv")))''')

md("""## 05. Safe Loading

Loads each file with the appropriate loader. Large files are sampled
(`INSPECTION MODE: SAMPLE/CHUNKED`); corrupted or unsupported files are
reported and skipped — they never abort the audit.""")

code(r'''run_stage(ctx, "load")      # 05. Safe Loading
display(markdown_table(ctx.get_table("01_file_inventory.csv")))
errs = [(d.display_name, d.status, e.get("error_message", "")[:200])
        for d in ctx.datasets for e in d.errors]
display(pd.DataFrame(errs, columns=["file", "status", "error"])
        if errs else "No load errors.")''')

md("""## 06. Dataset Profiling

Structure, previews, per-column descriptive statistics (min/max/mean/median/
std/quantiles for numeric columns only).""")

code(r'''run_stage(ctx, "profile")   # 06. Dataset Profiling
display(markdown_table(ctx.get_table("02_dataset_profiles.csv")))''')

md("""## 07. Variable Catalog & Column Semantics

Every column is assigned an inferred role (DATE/DATETIME/COUNTRY/ISO3/VALUE/…)
**with a confidence label**. Inferred roles are never treated as certain.""")

code(r'''run_stage(ctx, "variables")  # 07. Variable Catalog
vcat = ctx.get_table("03_variable_catalog.csv")
display(markdown_table(vcat.head(80)))''')

md("""## 08–09. Date/Time Audit & Temporal Coverage

Detects temporal variables, infers frequency from actual spacing, and computes
coverage, expected vs observed periods, gaps and duplicate timestamps.""")

code(r'''run_stage(ctx, "temporal")   # 08/09. Date/Time + Temporal Coverage
display(markdown_table(ctx.get_table("04_temporal_coverage.csv")))''')

md("""## 10–11. Geographic / Entity Audit

Detects entity columns, lists unique entities, flags aggregate entities
(World/Africa/EU/OECD/…), and builds a **proposed** country-normalization
mapping (never applied to the raw data).""")

code(r'''run_stage(ctx, "entities")   # 10/11. Geographic/Entity Audit
display(markdown_table(ctx.get_table("05_entity_coverage.csv")))''')

md("""## 11. Missing Data

Overall + per-variable + per-entity + per-time missingness, plus a heuristic
missingness-pattern classification (RANDOM-LOOKING / TIME-BASED / VARIABLE-BASED /
BLOCK MISSINGNESS).""")

code(r'''run_stage(ctx, "missingness")  # 11. Missing Data
display(markdown_table(ctx.get_table("06_missingness.csv")))''')

md("""## 12. Duplicate Detection

Exact duplicates, key duplicates (entity+time, country+year, …) and
**conflicting duplicates** (same logical key, different values). Nothing is
auto-dropped.""")

code(r'''run_stage(ctx, "duplicates")  # 12. Duplicate Detection
display(markdown_table(ctx.get_table("07_duplicate_report.csv")))''')

md("""## 13. Units & Metadata

Unit detection from column names / metadata / NetCDF attributes, plus metadata
extraction (Excel sheets, JSON structure, Parquet schema, NetCDF attributes).""")

code(r'''run_stage(ctx, "units_metadata")  # 13. Units & Metadata
display(markdown_table(ctx.get_table("08_unit_catalog.csv").head(80)))
display(markdown_table(ctx.get_table("09_metadata_report.csv")))''')

md("""## 14. Dataset Semantics

Infers what each dataset appears to contain (with evidence + confidence) and
builds per-variable semantics, flagging known conceptual traps
(DEMAND ≠ GENERATION, CAPACITY ≠ PRODUCTION, GDP ≠ GDP per capita, …).""")

code(r'''run_stage(ctx, "semantics")  # 14. Dataset Semantics
display(markdown_table(ctx.get_table("_purpose.csv")))''')

md("""## 15. Overlap Detection

Finds variables that may exist across datasets (name/unit/dtype/semantic
signals). Name matches alone never imply equivalence.""")

code(r'''run_stage(ctx, "overlap")   # 15. Overlap Detection
display(markdown_table(ctx.get_table("10_overlap_analysis.csv")))''')

md("""## 16. Cross-Dataset Comparison

For compatible variables with compatible keys, compares overlapping
observations (correlation, MAE, RMSE, mean/median diff, agreement rate).
Reports `COMPARISON NOT VALID` when definitions/units are incompatible.""")

code(r'''run_stage(ctx, "compare")   # 16. Cross-Dataset Comparison
display(markdown_table(ctx.get_table("11_cross_dataset_comparison.csv")))''')

md("""## 17. Conflict Analysis

Classifies likely causes of disagreement (UNIT_MISMATCH, FREQUENCY_MISMATCH,
DEFINITION_MISMATCH, ACTUAL_VS_FORECAST, DUPLICATE_CONFLICT, …).""")

code(r'''run_stage(ctx, "conflicts")  # 17. Conflict Analysis
display(markdown_table(ctx.get_table("12_conflict_report.csv").head(60)))''')

md("""## 18. Data Quality

Transparent 8-dimension scoring (each 0–5, total /40). The score never decides
suitability by itself — scientific suitability is assessed separately.""")

code(r'''run_stage(ctx, "quality")   # 18. Data Quality
display(markdown_table(ctx.get_table("13_quality_scores.csv")))''')

md("""## 19–20. Time-Series & Panel Readiness""")

code(r'''run_stage(ctx, "ts_readiness")     # 19. Time-Series Readiness
display(markdown_table(ctx.get_table("_ts_readiness.csv")))
run_stage(ctx, "panel_readiness")  # 20. Panel Readiness
display(markdown_table(ctx.get_table("_panel_readiness.csv")))''')

md("""## 21–22. Feature Availability & Leakage Audit""")

code(r'''run_stage(ctx, "features")  # 21. Feature Availability
display(markdown_table(ctx.get_table("14_feature_availability.csv")))
run_stage(ctx, "leakage")   # 22. Leakage Audit
display(markdown_table(ctx.get_table("15_leakage_flags.csv")))''')

md("""## 23. Anomaly Detection

Statistical outliers (IQR/z-score) and structural-break signals (level shift,
variance shift, new missingness regime, coverage change). Outliers are never
auto-deleted.""")

code(r'''run_stage(ctx, "anomaly")   # 23. Anomaly Detection
display(markdown_table(ctx.get_table("16_anomaly_report.csv").head(60)))''')

md("""## 24. Redundancy Analysis

Correlations among numeric variables; distinguishes conceptual vs statistical
redundancy. Highly correlated variables are never auto-removed.""")

code(r'''run_stage(ctx, "redundancy")  # 24. Redundancy Analysis
display(markdown_table(ctx.get_table("_redundancy.csv").head(60)))''')

md("""## 25–26. Overall Readiness & Recommended Next Actions""")

code(r'''run_stage(ctx, "readiness")        # 25. Overall Data Readiness
display(markdown_table(ctx.get_table("_overall_readiness.csv")))
run_stage(ctx, "recommendations")  # 26. Recommended Next Actions
display(markdown_table(ctx.get_table("17_recommendation_table.csv")))
dm = ctx.get_table("_decision_matrix.csv")
if dm is not None and not dm.empty:
    display(markdown_table(dm))''')

md("""## 27. Export Reports

Writes CSV audit tables, Markdown reports (including the human-readable
`DATA_AUDIT_REPORT.md`) and figures into `OUTPUT_DIR`. The source data is never
modified.""")

code(r'''run_stage(ctx, "export")    # 27. Export Reports
print("Export complete.")
print(f"CSV tables and reports written under: {ctx.config.output_dir}")
print("Main report:  data_audit_output/reports/DATA_AUDIT_REPORT.md")''')

md("""---
### Run everything at once (optional)

Instead of the staged cells above, you can run the whole pipeline in a single
call on a fresh context:""")

code(r'''# ctx2 = init_audit(config)
# run_all(ctx2)
# print("Full audit complete.")''')

nb["cells"] = cells

out = "data_audit.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out} with {len(cells)} cells")
