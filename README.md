# General-Purpose Multi-Source Data Inspection & Audit

A **forensic data investigation** toolkit: it observes, measures, flags,
compares, explains and recommends — but it **never** silently cleans, merges,
imputes, interpolates, normalizes, or modifies the source data.

> **What do I actually have, what does it mean, how good is it, how complete is
> it, what overlaps, what conflicts, what is missing, and what is usable?**
> That is the only thing this toolkit answers.

---

## Quick start

The primary interface is the notebook **`data_audit.ipynb`**:

1. Open the notebook.
2. Edit the **CONFIGURATION** cell (the `DATA_PATHS` list) — point it at your
   files and/or directories. That's it.
3. Run all cells.

Every audit stage then runs automatically. Results are printed inline and
written to `data_audit_output/`.

### Headless / script usage

```python
from dataaudit import init_audit, run_all
from dataaudit.core import Config

config = Config(
    data_paths=["path/to/data.csv", "path/to/data_dir/"],
    output_dir="data_audit_output",
    recursive=True,
)
ctx = init_audit(config)
run_all(ctx)                      # runs every stage, exports everything
```

You can also run stages one at a time (`run_stage(ctx, "load")`, …) — the
notebook does exactly this so you can inspect intermediate results.

---

## What it supports

| Format | Handling |
| --- | --- |
| CSV, TSV, TXT | `pandas` — sampled when large (row/byte limits) |
| XLSX / XLS / XLSM | every sheet inspected; hidden sheets flagged |
| Parquet / Feather | `pyarrow` schema + metadata + sampling |
| JSON / JSONL | structure analysis + normalization |
| NetCDF / HDF5 | `xarray`/`h5py` metadata-first, lazy (no full grid load) |
| ZIP / GZ / BZ2 | members extracted to a temp dir and audited individually |
| anything else | reported as `UNSUPPORTED` — never crashes the audit |

Large files are inspected in **SAMPLE / CHUNKED** mode and every statistic is
labelled `EXACT`, `SAMPLED`, `APPROXIMATE` or `METADATA-ONLY`.

---

## Audit stages

```
discover → inventory → load → profile → variables → temporal → entities
→ missingness → duplicates → units_metadata → semantics → overlap → compare
→ conflicts → quality → ts_readiness → panel_readiness → features → leakage
→ anomaly → redundancy → readiness → recommendations → export
```

## Output structure

```
data_audit_output/
├── 00_audit_log.txt                # reproducibility log (hashes, env, config)
├── 01_file_inventory.csv
├── 02_dataset_profiles.csv
├── 03_variable_catalog.csv
├── 04_temporal_coverage.csv
├── 05_entity_coverage.csv
├── 06_missingness.csv
├── 07_duplicate_report.csv
├── 08_unit_catalog.csv
├── 09_metadata_report.csv
├── 10_overlap_analysis.csv
├── 11_cross_dataset_comparison.csv
├── 12_conflict_report.csv
├── 13_quality_scores.csv
├── 14_feature_availability.csv
├── 15_leakage_flags.csv
├── 16_anomaly_report.csv
├── 17_recommendation_table.csv
├── 18_entity_temporal.csv
├── 19_ts_readiness.csv
├── 20_panel_readiness.csv
├── 21_decision_matrix.csv          # only when PROJECT_CONFIG provided
├── 22_dataset_purpose.csv
├── 23_variable_semantics.csv
├── 24_redundancy.csv
├── 25_country_normalization.csv    # PROPOSED mapping — never applied
├── 26_annual_monthly_flags.csv
├── reports/
│   ├── DATA_AUDIT_REPORT.md        # the human-readable final report
│   ├── DATASET_SUMMARY.md
│   ├── VARIABLE_CATALOG.md
│   ├── COVERAGE_REPORT.md
│   ├── QUALITY_REPORT.md
│   ├── OVERLAP_REPORT.md
│   ├── CONFLICT_REPORT.md
│   └── RECOMMENDATIONS.md
└── figures/                        # missingness / coverage / distributions / time_series
```

## Quality framework

Eight dimensions, each 0–5 (total **/40**): completeness, consistency, temporal
quality, geographic quality, definition clarity, unit clarity, metadata quality,
reproducibility.

```
34–40  Excellent
28–33  Strong
20–27  Usable with caveats
12–19  Weak
 0–11  Poor
```

The score documents quality — it does **not** decide whether a dataset should be
used. Scientific suitability is evaluated separately (time-series / panel /
forecast-horizon readiness).

## Confidence labels

Every inference carries one of: `HIGH CONFIDENCE`, `MEDIUM CONFIDENCE`,
`LOW CONFIDENCE`, `UNKNOWN`. Nothing inferred (date column, country column, unit,
frequency, variable meaning) is ever presented as certain.

## Project-specific / HGT-QF mode

Set `PROJECT_CONFIG` in the configuration cell (e.g. target = "electricity
demand", required frequency = monthly, candidate features, …). The audit then
additionally produces a **decision matrix** and an **HGT-QF readiness decision**
(12/36/60-month horizons, 24/60/120-month context, LOCO / zero-shot / scenario
readiness). This is entirely optional — the notebook stays general-purpose
without it.

---

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_notebook.py     # regenerate data_audit.ipynb
.venv/bin/python tests/test_audit.py          # end-to-end smoke test
```

Package layout (`dataaudit/`):

| Module | Responsibility |
| --- | --- |
| `core.py` | `Config`, `Dataset`, `AuditContext`, stage order |
| `inventory.py` | file discovery, classification, inventory table |
| `loaders.py` | safe, memory-aware loaders + sampling |
| `profiling.py` | per-column descriptive stats |
| `variables.py` | column semantic-role inference |
| `temporal.py` | date/time detection, frequency, coverage |
| `entities.py` | entity detection, country normalization analysis |
| `missingness.py` | missing-data audit + pattern classification |
| `duplicates.py` | exact / key / conflicting duplicate audit |
| `units_metadata.py` | unit detection + metadata extraction |
| `semantics.py` | purpose inference + variable semantics |
| `numerical.py` | numeric-variable audit |
| `overlap.py` | cross-dataset overlap detection |
| `compare.py` | cross-dataset numeric comparison |
| `conflicts.py` | conflict-cause classification |
| `quality.py` | quality scores, source identification, usability |
| `anomaly.py` | outliers + structural-break signals |
| `redundancy.py` | correlation / redundancy analysis |
| `features.py` | feature availability + leakage audit |
| `readiness.py` | TS/panel readiness, decision matrix, HGT-QF |
| `runner.py` | stage orchestration + error isolation |
| `export.py` | CSV / Markdown / figure export |

## Design guarantees

- Never overwrites, deletes, renames, rescales, imputes, or merges source data.
- Never auto-selects a "best" dataset without evidence.
- A corrupt or unsupported file never aborts the audit.
- All statistics carry an exactness label (`EXACT / SAMPLED / APPROXIMATE /
  METADATA-ONLY`).
- Every recommendation cites the observed evidence that supports it.
