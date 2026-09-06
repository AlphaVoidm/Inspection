"""Audit every data file of every source and save one report per dataset folder.

Layout it expects (the folders of this project):

    Inspection/
      ENTSOE/MonthlyDomesticValues/monthly_domestic_values_2019.csv
      ENTSOE/MonthlyDomesticValues/monthly_domestic_values_2020.csv
      OWID/owid-energy-data.csv
      CDS/58df30c3.../data_stream-moda_....nc

What it writes:

    reports/
      overview.md                  all sources side by side
      00_file_inventory.csv        every file found, audited or skipped, and why
      00_duplicate_files.csv       byte-identical downloads
      ENTSOE/index.md              all ENTSO-E datasets side by side
      ENTSOE/MonthlyDomesticValues/report.html    every cell output, for every file in the folder
      ENTSOE/MonthlyDomesticValues/summary.md     file table, schema comparison, audit per file
      ENTSOE/MonthlyDomesticValues/audits/*.json  machine-readable summary per file
      OWID/report.html, OWID/summary.md           files sitting directly in the source folder

Usage:
    python batch_audit.py                                  # all sources found next to this script
    python batch_audit.py --sources ENTSOE OWID            # only these
    python batch_audit.py --root . --out reports
    python batch_audit.py data                             # any folder of sub-folders (demo layout)
"""

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

DATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".parquet", ".feather",
                   ".json", ".jsonl", ".nc", ".nc4", ".cdf", ".netcdf"}
SKIP_DIRS = {"dataaudit", "data_audit_output", "reports", "scripts", "tests", "examples",
             "__pycache__", ".git", ".ipynb_checkpoints", ".venv", "venv", ".idea", "figures"}
SKIP_FILE_PATTERNS = ["how to download", "download_links", "readme"]

PAGE_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1180px; padding: 24px 32px 80px; color: #1c1c1c; line-height: 1.5; }
h1 { border-bottom: 3px solid #4f81bd; padding-bottom: 6px; margin-top: 48px; }
h2 { color: #24486b; margin-top: 28px; }
pre { background: #f6f7f9; border: 1px solid #e2e5e9; border-radius: 4px;
      padding: 10px 12px; overflow-x: auto; font-size: 12.5px; white-space: pre; }
table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
table td, table th { border: 1px solid #d8dce1; padding: 3px 8px; text-align: right; }
table th { background: #eef2f7; }
img { max-width: 100%; height: auto; }
.toc { background: #f6f7f9; border: 1px solid #e2e5e9; border-radius: 6px; padding: 12px 20px; }
.filecard { background: #eef2f7; border-left: 5px solid #4f81bd; padding: 10px 16px; margin: 12px 0; }
.status-Problem { color: #b02318; font-weight: 600; }
.status-Warning { color: #b06a00; font-weight: 600; }
.status-Good { color: #1d6b32; font-weight: 600; }
.meta { color: #5a6470; font-size: 13px; }
.error { background: #fdeceb; border-left: 5px solid #b02318; padding: 10px 16px; }
.partial { background: #fff6e0; border-left: 5px solid #b06a00; padding: 10px 16px; }
"""


# --------------------------------------------------------------------------- discovery
def inventory_records(root: Path, sources):
    """Walk the source folders and describe every file: audit it, or skip it and say why."""
    records = []
    for source in sources:
        source_dir = root / source
        if not source_dir.is_dir():
            print(f"! source folder not found, skipped: {source_dir}")
            continue
        for dirpath, dirnames, filenames in os.walk(source_dir):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                extension = file_path.suffix.lower()
                size_mb = file_path.stat().st_size / 1024 ** 2
                dataset = str(Path(dirpath).relative_to(source_dir))
                if any(pattern in filename.lower() for pattern in SKIP_FILE_PATTERNS):
                    status = "skipped: documentation / link list"
                elif extension == ".zip":
                    status = "skipped: archive (extract it - the extracted folder is audited)"
                elif extension not in DATA_EXTENSIONS:
                    status = f"skipped: not a data format ({extension or 'no extension'})"
                elif size_mb == 0:
                    status = "skipped: empty file"
                else:
                    status = "audit"
                records.append({"source": source, "dataset": dataset, "file": filename,
                                "ext": extension, "size_mb": round(size_mb, 3),
                                "status": status, "path": str(file_path)})
    return records


def duplicate_groups(records):
    """Byte-identical files (first 4 MB + size), i.e. re-downloaded copies."""
    digests = {}
    for record in records:
        if record["status"] != "audit":
            continue
        hasher = hashlib.md5()
        with open(record["path"], "rb") as handle:
            hasher.update(handle.read(4 * 1024 ** 2))
        digests.setdefault((hasher.hexdigest(), record["size_mb"]), []).append(record["path"])
    return [paths for paths in digests.values() if len(paths) > 1]


def safe_name(text: str) -> str:
    """Folder name that is valid on every OS and still readable."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text)).strip(" .")
    return (cleaned[:60] or "_")


# --------------------------------------------------------------------------- execution
def run_notebook(notebook_path: Path, data_file: Path, summary_path: Path, timeout: int, dpi: int):
    """Execute the audit notebook for one data file. Returns (executed notebook, error count)."""
    nb = nbformat.read(notebook_path, as_version=4)
    overrides = "\n".join([
        "# Parameters injected by batch_audit.py",
        f"DATA_PATH = {str(data_file.resolve())!r}",
        f"SUMMARY_JSON_PATH = {str(summary_path.resolve())!r}",
        "SAVE_TEXT_REPORT = False",
        "BATCH_CHILD = True",
        "RUN_BATCH = False",
        f"plt.rcParams['figure.dpi'] = {dpi}",     # smaller embedded plots keep the report light
        "print('Batch parameters applied for:', DATA_PATH)",
    ])
    first_code = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code")
    nb.cells.insert(first_code + 1, nbformat.v4.new_code_cell(overrides))

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                            extra_arguments=["--log-level=ERROR"],
                            resources={"metadata": {"path": str(notebook_path.parent)}})
    client.execute()
    errors = sum(1 for cell in nb.cells for out in cell.get("outputs", [])
                 if out.get("output_type") == "error")
    return nb, errors


# --------------------------------------------------------------------------- rendering
def markdown_table(rows):
    if not rows:
        return "_none_\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]).replace("|", "/") for h in headers) + " |")
    return "\n".join(out) + "\n"


def html_table(rows):
    if not rows:
        return "<p><em>none</em></p>"
    headers = list(rows[0].keys())
    cells = ["<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"]
    for row in rows:
        cells.append("<tr>" + "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers) + "</tr>")
    return "<table>" + "".join(cells) + "</table>"


def file_rows(summaries):
    rows = []
    for s in summaries:
        period = "-"
        if s.get("time_min"):
            period = f"{str(s['time_min'])[:10]} → {str(s['time_max'])[:10]}"
        rows.append({
            "File": s["file"],
            "Rows": f"{s['rows']:,}" + (" (partial)" if s.get("partial_read") else ""),
            "Cols": s["columns"],
            "Missing %": f"{s['missing_overall_pct']:.2f}",
            "Dup rows": f"{s['duplicate_rows']:,}",
            "Entities": s["n_entities"] if s["n_entities"] is not None else "-",
            "Period": period,
            "Frequency": (s.get("frequency") or "-"),
            "Problems": len(s.get("problems", [])),
        })
    return rows


def schema_comparison(summaries):
    """How the files of one dataset folder differ in structure."""
    groups = {}
    for s in summaries:
        groups.setdefault(tuple(s["column_names"]), []).append(s["file"])

    lines = []
    if len(groups) == 1:
        columns = next(iter(groups))
        lines.append(f"All {len(summaries)} file(s) share the same {len(columns)} columns - they can be "
                     "stacked or compared without renaming.")
    else:
        lines.append(f"The {len(summaries)} file(s) use {len(groups)} different column layouts:")
        for i, (columns, files) in enumerate(groups.items(), start=1):
            lines.append(f"\n**Layout {i}** ({len(columns)} columns) — {', '.join(files)}")
            lines.append("`" + "`, `".join(map(str, columns)) + "`")
        everywhere = set.intersection(*(set(c) for c in groups))
        anywhere = set().union(*(set(c) for c in groups))
        differing = sorted(anywhere - everywhere)
        lines.append(f"\nColumns present in every file: {len(everywhere)}")
        lines.append("Columns missing from at least one file: "
                     + (", ".join(f"`{c}`" for c in differing[:40]) if differing else "none"))

    dtype_map = {}
    for s in summaries:
        for column, dtype in s["dtypes"].items():
            dtype_map.setdefault(column, {}).setdefault(str(dtype), []).append(s["file"])
    clashes = {c: d for c, d in dtype_map.items() if len(d) > 1}
    if clashes:
        lines.append("\n**Same column, different storage type across files "
                     "(one of the downloads parsed differently):**")
        for column, dtypes in list(clashes.items())[:15]:
            detail = "; ".join(f"`{dtype}` in {', '.join(files)}" for dtype, files in dtypes.items())
            lines.append(f"- `{column}`: {detail}")
    return "\n".join(lines) + "\n"


def write_dataset_report(title, results, out_dir: Path):
    """report.html (all cell outputs) + summary.md (quick read) for one dataset folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    summaries = [r["summary"] for r in results if r["summary"]]

    toc = "".join(
        f'<li><a href="#file-{i}">{html.escape(r["file"].name)}</a>'
        f'{" <em>(execution errors)</em>" if r["errors"] else ""}</li>'
        for i, r in enumerate(results))
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>Data audit - {html.escape(title)}</title><style>{PAGE_CSS}</style></head><body>",
             f"<h1>Data quality audit — {html.escape(title)}</h1>",
             f"<p class='meta'>Generated {generated} &middot; {len(results)} file(s) &middot; "
             "every output of every notebook cell is included below.</p>",
             "<h2>Files in this dataset folder</h2>", html_table(file_rows(summaries)),
             "<h2>Contents</h2>", f"<div class='toc'><ol>{toc}</ol></div>"]

    for i, result in enumerate(results):
        summary = result["summary"]
        parts.append(f"<h1 id='file-{i}'>{i + 1}. {html.escape(result['file'].name)}</h1>")
        parts.append(f"<p class='meta'>{html.escape(str(result['file']))}</p>")
        if result["errors"]:
            parts.append(f"<div class='error'><strong>{result['errors']} cell(s) raised an error</strong>"
                         " during execution — the tracebacks are shown inline below.</div>")
        if summary:
            if summary.get("partial_read"):
                reason = summary.get("partial_reason") or f"only {summary['rows']:,} rows were read"
                parts.append("<div class='partial'><strong>Partial audit</strong> — "
                             f"{html.escape(str(reason))}. Every number below describes that "
                             "sample, not the whole file.</div>")
            scorecard = "".join(
                f"<tr><td style='text-align:left'>{html.escape(str(row['quality_dimension']))}</td>"
                f"<td style='text-align:left'><span class='status-{html.escape(str(row['status']))}'>"
                f"{html.escape(str(row['status']))}</span></td>"
                f"<td style='text-align:left'>{html.escape(str(row['evidence']))}</td></tr>"
                for row in summary.get("scorecard", []))
            parts.append("<div class='filecard'>"
                         f"<strong>{summary['rows']:,} rows &times; {summary['columns']} columns</strong>"
                         f" &middot; missing {summary['missing_overall_pct']:.2f}%"
                         f" &middot; {summary['duplicate_rows']:,} duplicate rows"
                         f" &middot; entities: "
                         f"{summary['n_entities'] if summary['n_entities'] is not None else 'n/a'}"
                         f" &middot; period: "
                         f"{str(summary['time_min'])[:10] if summary['time_min'] else 'n/a'} → "
                         f"{str(summary['time_max'])[:10] if summary['time_max'] else 'n/a'}</div>")
            parts.append("<h2>Scorecard</h2><table><tr><th>Quality dimension</th><th>Status</th>"
                         f"<th>Evidence</th></tr>{scorecard}</table>")
            parts.append("<h2>Final audit</h2><pre>"
                         + html.escape(summary.get("final_audit_text", "")) + "</pre>")
        parts.append("<h2>Full notebook output</h2>")
        parts.append(result["body"])
    parts.append("</body></html>")
    (out_dir / "report.html").write_text("".join(parts), encoding="utf-8")

    md = [f"# Data quality audit — {title}", "",
          f"_Generated {generated} from {len(results)} file(s). "
          "Full evidence (every cell output, tables and plots): `report.html`._", "",
          "## Files", "", markdown_table(file_rows(summaries))]

    crashed = [r for r in results if r.get("failure")]
    if crashed:
        md += ["", "## Files that could not be audited at all", ""]
        md += [f"- `{r['file'].name}`: {r['failure']}" for r in crashed]

    failed = [r for r in results if r["errors"] and not r.get("failure")]
    if failed:
        md += ["", "## Files that did not run cleanly", ""]
        md += [f"- `{r['file'].name}`: {r['errors']} cell(s) raised an error "
               "(traceback in `report.html`)" for r in failed]

    partial = [s for s in summaries if s.get("partial_read")]
    if partial:
        md += ["", "## Partially audited files", "",
               "These were too large to load completely, so their audit describes a sample only:"]
        md += [f"- `{s['file']}` ({s['file_size_mb']} MB): "
               f"{s.get('partial_reason') or str(s['rows']) + ' rows read'}" for s in partial]

    if summaries:
        md += ["", "## Schema comparison across the files of this folder", "",
               schema_comparison(summaries)]
        recurring = {}
        for s in summaries:
            for problem in s.get("problems", []):
                recurring.setdefault(problem.split(":")[0][:70], []).append(s["file"])
        recurring = {k: v for k, v in recurring.items() if len(v) > 1}
        if recurring:
            md += ["## Problems that recur across files", ""]
            md += [f"- {key} — in {len(files)} file(s): {', '.join(sorted(set(files)))}"
                   for key, files in list(recurring.items())[:20]] + [""]

    for s in summaries:
        md += [f"## {s['file']}", "",
               f"- Path: `{s['path']}`",
               f"- Loaded with: `{s.get('loader', '')}`"
               + ("  **(partial read)**" if s.get("partial_read") else ""),
               f"- Size: {s['file_size_mb']} MB | {s['rows']:,} rows x {s['columns']} columns "
               f"({s['n_numeric']} numeric, {s['n_categorical']} categorical, {s['n_datetime']} datetime)",
               (f"- Entity column: `{s['entity_column']}`" if s["entity_column"]
                else "- Entity column: none detected"),
               (f"- Time column: `{s['time_column']}` ({s['frequency']})" if s["time_column"]
                else "- Time column: none detected"),
               "", "### Scorecard", "",
               markdown_table([{"Quality dimension": r["quality_dimension"], "Status": r["status"],
                                "Evidence": str(r["evidence"])} for r in s.get("scorecard", [])]),
               "### Final audit", "", "```text", s.get("final_audit_text", ""), "```", ""]

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def write_source_index(source, dataset_results, out_dir: Path, skipped, duplicates):
    """index.md listing every dataset folder of one source."""
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for dataset, (results, rel_dir) in dataset_results.items():
        summaries = [r["summary"] for r in results if r["summary"]]
        link = f"[{rel_dir}/summary.md]({rel_dir}/summary.md)" if rel_dir != "." else "[summary.md](summary.md)"
        rows.append({
            "Dataset folder": dataset if dataset != "." else "(files in the source root)",
            "Files": len(results),
            "Rows": f"{sum(s['rows'] for s in summaries):,}",
            "Missing % (max)": f"{max((s['missing_overall_pct'] for s in summaries), default=0):.2f}",
            "Dup rows": f"{sum(s['duplicate_rows'] for s in summaries):,}",
            "Problems": sum(len(s.get("problems", [])) for s in summaries),
            "Failed cells": sum(r["errors"] for r in results),
            "Report": link,
        })
    md = [f"# {source} — audit index", "", f"_Generated {generated}._", "",
          "## Dataset folders", "", markdown_table(rows)]
    if duplicates:
        md += ["## Byte-identical files inside this source", "",
               "Re-downloaded copies; they are reported, never deleted.", ""]
        for paths in duplicates:
            md += [f"- {len(paths)} copies:"] + [f"    - `{p}`" for p in paths]
        md += [""]
    if skipped:
        md += ["## Files that were not audited", "", markdown_table(skipped)]
    (out_dir / "index.md").write_text("\n".join(md), encoding="utf-8")


def write_overview(source_summaries, records, duplicates, out_root: Path):
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for source, info in source_summaries.items():
        rows.append({
            "Source": f"[{source}]({source}/index.md)",
            "Dataset folders": info["datasets"],
            "Files audited": info["files"],
            "Rows": f"{info['rows']:,}",
            "Missing % (max)": f"{info['max_missing']:.2f}",
            "Duplicate rows": f"{info['duplicate_rows']:,}",
            "Problems": info["problems"],
            "Failed cells": info["errors"],
        })
    audited = sum(1 for r in records if r["status"] == "audit")
    md = ["# Dataset audit overview", "",
          f"_Generated {generated} — {audited} file(s) audited across {len(source_summaries)} source(s)._",
          "", markdown_table(rows), "",
          "Each source has an `index.md` listing its dataset folders. Each dataset folder has a "
          "`summary.md` (quick read) and a `report.html` (every output of every notebook cell, per file).",
          "", "## Files that were not audited", ""]
    skipped_rows = [{"Source": r["source"], "File": r["file"], "Size MB": r["size_mb"],
                     "Reason": r["status"]} for r in records if r["status"] != "audit"]
    md += [markdown_table(skipped_rows[:80])]
    if len(skipped_rows) > 80:
        md += [f"_... {len(skipped_rows) - 80} more, see `00_file_inventory.csv`._", ""]
    if duplicates:
        md += ["", "## Byte-identical files across the whole collection", ""]
        for paths in duplicates:
            md += [f"- {len(paths)} copies:"] + [f"    - `{p}`" for p in paths]
    (out_root / "overview.md").write_text("\n".join(md), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".", help="project root (default: .)")
    parser.add_argument("--root", dest="root_opt", help="project root (same as the positional argument)")
    parser.add_argument("--sources", nargs="+",
                        help="source folders to audit (default: every folder under the root "
                             "that is not a code/output folder)")
    parser.add_argument("--out", default="reports", help="output folder (default: reports)")
    parser.add_argument("--notebook", default="dataset_inspection.ipynb", help="audit notebook to run")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    parser.add_argument("--dpi", type=int, default=70, help="plot resolution inside the reports")
    parser.add_argument("--limit", type=int, help="audit at most this many files per dataset folder")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip dataset folders that already have a report.html "
                             "(resume an interrupted run)")
    parser.add_argument("--max-file-mb", type=float,
                        help="skip files larger than this many MB instead of auditing them")
    args = parser.parse_args()

    root = Path(args.root_opt or args.root).resolve()
    out_root = Path(args.out)
    notebook_path = Path(args.notebook)
    if not root.is_dir():
        sys.exit(f"Root folder not found: {root}")
    if not notebook_path.is_file():
        sys.exit(f"Notebook not found: {notebook_path}")

    sources = args.sources or sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS
        and not p.name.startswith("."))

    records = inventory_records(root, sources)
    if args.max_file_mb:
        for record in records:
            if record["status"] == "audit" and record["size_mb"] > args.max_file_mb:
                record["status"] = f"skipped: larger than --max-file-mb ({args.max_file_mb} MB)"
    if not records:
        sys.exit(f"No files found under {root} for sources: {', '.join(sources)}")

    out_root.mkdir(parents=True, exist_ok=True)
    import csv as csv_module
    with open(out_root / "00_file_inventory.csv", "w", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    duplicates = duplicate_groups(records)
    if duplicates:
        with open(out_root / "00_duplicate_files.csv", "w", newline="") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(["copies", "paths"])
            for paths in duplicates:
                writer.writerow([len(paths), " | ".join(paths)])

    to_audit = [r for r in records if r["status"] == "audit"]
    print(f"{len(records)} file(s) found, {len(to_audit)} to audit, "
          f"{len(records) - len(to_audit)} skipped, {len(duplicates)} duplicate group(s).")

    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    source_summaries = {}
    for source in sources:
        source_files = [r for r in to_audit if r["source"] == source]
        if not source_files:
            continue
        print(f"\n=== {source}: {len(source_files)} file(s) ===")
        datasets = {}
        for record in source_files:
            datasets.setdefault(record["dataset"], []).append(record)

        dataset_results = {}
        for folder_no, (dataset, dataset_records) in enumerate(datasets.items(), start=1):
            if args.limit:
                dataset_records = dataset_records[:args.limit]
            rel_dir = "." if dataset == "." else "/".join(safe_name(p) for p in Path(dataset).parts)
            out_dir = out_root / source if rel_dir == "." else out_root / source / rel_dir
            audits_dir = out_dir / "audits"
            audits_dir.mkdir(parents=True, exist_ok=True)
            label = source if dataset == "." else f"{source} / {dataset}"
            # One folder at a time, and one file at a time inside it: each file is audited in its
            # own kernel process, so its memory is released before the next file is opened.
            print(f"  [folder {folder_no}/{len(datasets)}] {label} "
                  f"- {len(dataset_records)} file(s)")

            if args.skip_existing and (out_dir / "report.html").exists():
                print("    already reported - skipped (--skip-existing)")
                continue

            results = []
            for record in dataset_records:
                data_file = Path(record["path"])
                print(f"    - {data_file.name} ({record['size_mb']} MB) ... ", end="", flush=True)
                summary_path = audits_dir / f"{safe_name(data_file.stem)}.json"
                try:
                    nb, errors = run_notebook(notebook_path, data_file, summary_path,
                                              args.timeout, args.dpi)
                    body, _ = exporter.from_notebook_node(nb)
                except Exception as exc:      # deliberately broad: keep the batch running
                    # A file that cannot be audited at all (out of memory, dead kernel, unreadable)
                    # is recorded as a failure and the batch moves on to the next one.
                    message = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED - {message[:120]}")
                    results.append({"file": data_file, "errors": 1, "summary": None,
                                    "body": f"<div class='error'><strong>This file could not be "
                                            f"audited.</strong><pre>{html.escape(message)}</pre></div>",
                                    "failure": message})
                    continue
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
                results.append({"file": data_file, "errors": errors, "summary": summary, "body": body})
                print("done" if not errors else f"done ({errors} cell error(s))")

            if not results:
                continue

            write_dataset_report(label, results, out_dir)
            dataset_results[dataset] = (results, rel_dir)
            print(f"    -> {out_dir / 'report.html'}")

        skipped = [{"File": r["file"], "Dataset": r["dataset"], "Size MB": r["size_mb"],
                    "Reason": r["status"]}
                   for r in records if r["source"] == source and r["status"] != "audit"]
        source_duplicates = [paths for paths in duplicates
                             if all(f"{os.sep}{source}{os.sep}" in p for p in paths)]
        write_source_index(source, dataset_results, out_root / source, skipped, source_duplicates)

        all_summaries = [r["summary"] for results, _ in dataset_results.values()
                         for r in results if r["summary"]]
        source_summaries[source] = {
            "datasets": len(dataset_results),
            "files": sum(len(results) for results, _ in dataset_results.values()),
            "rows": sum(s["rows"] for s in all_summaries),
            "max_missing": max((s["missing_overall_pct"] for s in all_summaries), default=0),
            "duplicate_rows": sum(s["duplicate_rows"] for s in all_summaries),
            "problems": sum(len(s.get("problems", [])) for s in all_summaries),
            "errors": sum(r["errors"] for results, _ in dataset_results.values() for r in results),
        }

    write_overview(source_summaries, records, duplicates, out_root)
    print(f"\nOverview: {out_root / 'overview.md'}")


if __name__ == "__main__":
    main()
