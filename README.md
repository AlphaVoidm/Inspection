# Dataset Inspection

A single, self-contained Jupyter notebook for auditing a dataset **before** any cleaning or modelling:

> *What exactly is in this dataset, how reliable and usable is it, what problems does it have,
> and is it worth keeping for further analysis?*

| File | Purpose |
| --- | --- |
| `dataset_inspection.ipynb` | The audit notebook — the thing you run |
| `batch_audit.py` | Batch runner: audits every file in every site folder and saves one report per folder |
| `examples/sample_energy_panel.csv` | Small demo panel with deliberate defects, so the notebook has something to find |
| `examples/sites/` | Demo multi-site tree (two site folders) for trying batch mode |
| `examples/make_sample_data.py` | Script that regenerates the demo files |
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

## Batch mode — one report per site folder

When your data is organised as **one folder per site / source**, each holding several files:

```text
data/
  site_alpha/demand_2015_2019.csv
  site_alpha/demand_2020_2023.csv
  site_beta/weather_monthly.csv
  site_beta/stations.csv
```

run either the last cell of the notebook (section 25, it picks up `DATA_ROOT` automatically) or:

```bash
python batch_audit.py data --out reports
python batch_audit.py examples/sites --out reports    # try it on the bundled demo
python batch_audit.py data --site site_beta           # one folder only
```

The whole notebook is executed once per data file and the results are saved **per folder**:

```text
reports/
  overview.md                    all site folders side by side
  site_alpha/
    report.html                  every output of every cell, for every file in the folder
    summary.md                   file table, schema comparison, scorecard + final audit per file
    audits/demand_2015_2019.json machine-readable summary (one per file)
```

- **`report.html`** is the evidence: for each file, a header card with the key numbers, the scorecard,
  the final audit, and then the complete notebook output — every table, plot and printed line, in
  notebook order, with a table of contents at the top.
- **`summary.md`** is the quick read: one row per file (rows, columns, missing %, duplicates, entities,
  period, frequency, problem count), a **schema comparison** across the files of that folder (which
  layouts exist, which columns are missing from some files, which columns are stored with different
  dtypes in different files), the **problems that recur across files**, and then each file's scorecard
  and final audit text.
- A file whose audit hits an error does not stop the batch: the traceback is kept in `report.html` and
  the file is listed under "Files that did not run cleanly".

Reports are written locally and are excluded from git (`reports/` is in `.gitignore`).
Expect roughly 0.5–1 MB of HTML per audited file, since the plots are embedded.

Running the notebook on a single file (no `data/` folder) still saves
`reports/<filename>_audit.txt` with the final audit and scorecard.

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
25. Save the report / batch mode (text + JSON per file, one report per site folder)

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
