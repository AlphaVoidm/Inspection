# Dataset Inspection

A single, self-contained Jupyter notebook for auditing a dataset **before** any cleaning or modelling:

> *What exactly is in this dataset, how reliable and usable is it, what problems does it have,
> and is it worth keeping for further analysis?*

| File | Purpose |
| --- | --- |
| `dataset_inspection.ipynb` | The audit notebook — the thing you run |
| `examples/sample_energy_panel.csv` | Small demo panel with deliberate defects, so the notebook has something to find |
| `examples/make_sample_data.py` | Script that regenerates the demo file |
| `requirements.txt` | Minimal dependencies |

## Usage

```bash
pip install -r requirements.txt
jupyter lab dataset_inspection.ipynb
```

Set `DATA_PATH` in the configuration cell (section 1) and run all cells top to bottom:

```python
DATA_PATH = "examples/sample_energy_panel.csv"
```

Supported formats: `.csv`, `.tsv`, `.txt`, `.xlsx`/`.xls`, `.parquet`, `.feather`, `.json`/`.jsonl`,
and `.nc`/NetCDF (opened with `xarray`; a derived flat table is used for the tabular checks while the
original `xarray.Dataset` stays available as `ds`).

## What it checks

1. Imports & configuration
2. Load dataset (file type, shape, head/tail, NetCDF summary)
3. Dataset structure (dtypes, memory, per-column overview, NetCDF dims/coords/attrs)
4. Important variables (keyword scan — name-based hint only)
5. Missing data audit (per column, overall, severity bands, bar chart)
6. Duplicate records (full rows and repeated identifier combinations)
7. Constant / low-variation columns
8. Cardinality (categories, high-cardinality and ID-like columns)
9. Data type validation (numbers as text, dates as text, mixed types, placeholders)
10. Numeric statistics (describe + range, CV, zeros, negatives, infinities)
11. Categorical statistics (top values, bar charts for ≤20 categories)
12. Temporal coverage (range, inferred frequency, missing periods, gaps)
13. Entity coverage (records per entity, panel balance, per-entity time span)
14. Geographic coverage (country/ISO consistency, coordinate bounds)
15. Variable distributions (≤10 histograms)
16. Outlier detection (IQR fences + boxplots)
17. Correlation analysis (heatmap, |r| > 0.7 pairs)
18. Plausibility checks (mathematically invalid vs. potentially suspicious)
19. Unit & scale inspection (documented units only — never guessed)
20. Internal consistency (identifier mappings, key uniqueness, date ordering)
21. Time-series quality (ordering, gaps, jumps, level shifts, per entity)
22. Coverage summary
23. Data quality scorecard (status + evidence per dimension)
24. Final dataset audit (problems, strengths, relevance; recommendation left blank)

## Ground rules the notebook follows

- **No modification of the raw data.** Nothing is dropped, filled, interpolated, renamed, converted,
  deduplicated or aggregated. Temporary objects needed for an inspection (parsed dates, sorted copies,
  string samples) are clearly named as derived; `df` stays exactly as loaded.
- **Evidence for every conclusion.** The scorecard and final audit quote the numbers that produced them.
- **Confirmed problems are separated from potential issues.** Infinite values, out-of-range coordinates
  and duplicate rows are facts; outliers, negatives, strong correlations and uneven coverage need domain
  judgement.
- **No verdict.** The keep / reject / investigate-further recommendation is left blank on purpose.
- **Simple code.** Plain pandas cells, no classes or pipelines; sections that do not apply print a short
  skip comment instead of failing.
