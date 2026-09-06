# Dataset Inspection

A single, self-contained Jupyter notebook for auditing a dataset **before** any cleaning or modelling:

> *What exactly is in this dataset, how reliable and usable is it, what problems does it have,
> and is it worth keeping for further analysis?*

| File | Purpose |
| --- | --- |
| `dataset_inspection.ipynb` | The audit notebook — the thing you run |
| `batch_audit.py` | Batch runner: audits every file of every source, one report per dataset folder |
| `examples/sample_energy_panel.csv` | Small demo panel with deliberate defects, so the notebook has something to find |
| `examples/sites/` | Demo multi-source tree for trying batch mode without real data |
| `examples/make_sample_data.py` | Script that regenerates the demo files |
| `requirements.txt` | Minimal dependencies |

## Usage

```bash
pip install -r requirements.txt
jupyter lab dataset_inspection.ipynb      # then: Run All
```

Running it top to bottom does everything: it inventories the source folders, audits one file in
detail (`DATA_PATH = "auto"` picks the first file of the inventory), and then audits **every** file and
saves one report per dataset folder. To inspect one specific file interactively, name it instead:

```python
DATA_PATH = "OWID/owid-energy-data.csv"
RUN_BATCH = False        # skip the full batch while working on a single file
```

Supported formats: `.csv`, `.tsv`, `.txt`, `.xlsx`/`.xls`, `.parquet`, `.feather`, `.json`/`.jsonl`,
and `.nc`/NetCDF (opened with `xarray`; a derived flat table is used for the tabular checks while the
original `xarray.Dataset` stays available as `ds`).

## Your data sources

The notebook is pre-configured with the source folders of this project (configuration cell, section 1):

```python
PROJECT_ROOT = "."
DATA_SOURCES = ["CDS", "EIA", "Ember", "ENTSOE", "IEA", "IRENASTAT", "OWID", "WorldBank"]
```

Section 2 walks them and builds a **file inventory** — every file, its size, and whether it will be
audited or skipped and why. Skipped by design:

| Skipped | Why |
| --- | --- |
| `dataaudit/`, `data_audit_output/`, `scripts/`, `tests/`, `__pycache__/`, `reports/`, `examples/` | code, caches, previous results |
| `*.zip` (`WDI_CSV.zip`, `58df30c3….zip`) | archives — extract them; the extracted folder is audited |
| `How to download.txt`, `download_links.csv` | documentation and link lists, not data |
| `notebook.ipynb` inside `CDS/…` | not a data format |

The inventory also flags **byte-identical files**, which is how re-downloads such as
`inventory_of_transmission_2023 (1).csv` and `IEA-MethaneEmissionsComparison-World (1).csv` surface.

Loading is adapted to what is actually in these folders:

- **Delimiter sniffing** — ENTSO-E exports are `;` separated, others `,`; the file decides.
- **EIA bulk `.txt`** (`EIA/ELEC/ELEC.txt`) is read as line-delimited JSON, capped at `JSON_LINES_ROWS`.
- **Excel workbooks** print their sheet names; only `EXCEL_SHEET` is audited (says so in the output).
- **Very large files** (> `LARGE_FILE_MB`, e.g. `WDICSV.csv`) are read partially and every report marks
  the audit **PARTIAL READ**, so a truncated audit is never mistaken for a complete one.
- The configuration cell prints a `Notebook version` line. If a run of yours dies inside
  `ds.to_dataframe()`, the copy being executed predates the fix below — pull this notebook again.
- **NetCDF** (`CDS/*.nc`) is opened lazily with `xarray`. A grid is **never** flattened whole:
  `1039 time x 721 lat x 1440 lon` is 1.08 billion rows (that is the `MemoryError: unable to allocate
  16.1 GiB` you get from `to_dataframe()`). Above `NETCDF_MAX_CELLS` the flat table is built from a
  strided sample — **spatial dimensions are thinned first so the time axis stays complete** — and text
  coordinates such as ERA5's `expver` are kept out of the table, because they expand to one
  4-character string per grid point. For the file above that means every 32nd latitude and every 64th
  longitude: 549,631 rows, all 1039 time steps, 0.05% of the grid. `ds` still holds the full dataset,
  so the metadata/dimension report covers everything, and every report is stamped **PARTIAL AUDIT**
  with the exact sampling used.

## Batch mode — one report per dataset folder

Run the notebook top to bottom (section 25 launches it automatically when the source folders exist), or:

```bash
python batch_audit.py                              # every source found next to the notebook
python batch_audit.py --sources ENTSOE OWID        # only these sources
python batch_audit.py --sources IRENASTAT --limit 2   # quick smoke test, 2 files per folder
python batch_audit.py --skip-existing                # resume: skip folders already reported
python batch_audit.py --max-file-mb 500              # skip anything bigger than 500 MB
python batch_audit.py examples/sites               # the bundled demo tree
```

The whole notebook runs once per file, and results are grouped exactly like your folders:

```text
reports/
  overview.md                                  all sources side by side
  00_file_inventory.csv                        every file found, audited or skipped, and why
  00_duplicate_files.csv                       byte-identical downloads
  ENTSOE/index.md                              all ENTSO-E dataset folders side by side
  ENTSOE/MonthlyDomesticValues/
    report.html                                every cell output for all 10 files in that folder
    summary.md                                 file table, schema comparison, audit per file
    audits/monthly_domestic_values_2019.json   machine-readable summary
  OWID/report.html, OWID/summary.md            files sitting directly in a source folder
```

- **`report.html`** — the evidence: per file a header card, the scorecard, the final audit, then the
  complete notebook output (every table, plot and printed line), with a table of contents.
- **`summary.md`** — the quick read: one row per file (rows, cols, missing %, duplicates, entities,
  period, frequency, problems), the **schema comparison across the files of that folder** (which
  layouts exist, which columns are missing from some years, which columns changed dtype between
  downloads), **problems that recur across files**, and each file's scorecard and final audit.
- **`index.md` / `overview.md`** — dataset folders per source, and sources side by side.
- **Nothing stops the batch.** Files are audited one folder at a time, each in its own kernel process,
  so memory is released after every file. A cell that raises keeps its traceback in `report.html`
  ("Files that did not run cleanly"); a file that cannot be loaded at all — out of memory, dead
  kernel, unreadable — is recorded under "Files that could not be audited at all" and the run
  continues. Use `--skip-existing` to resume an interrupted run.

Runtime is a few seconds per file plus load time (≈90 s for 70 small files), and reports run roughly
0.5–0.7 MB of HTML per file. Set `RUN_BATCH = False` in the configuration cell to skip the batch while
working interactively. Reports are written locally and gitignored.

## What it checks

1. Imports & configuration (sources, skip rules, large-file limits)
2. Dataset inventory + load (file inventory, duplicate files, then the selected file)
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
25. Save the report / batch mode (text + JSON per file, one report per dataset folder)

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
