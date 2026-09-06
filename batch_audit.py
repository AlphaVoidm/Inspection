"""Run the audit notebook on every data file in every site folder and save one report per folder.

Expected layout (one sub-folder per site / data source):

    data/
      site_alpha/demand_2015_2019.csv
      site_alpha/demand_2020_2023.csv
      site_beta/weather.csv

Produces, for each site folder:

    reports/site_alpha/report.html    every output of every notebook cell, for every file
    reports/site_alpha/summary.md     site overview, schema comparison, per-file audit text
    reports/site_alpha/audits/*.json  machine-readable summary per file
    reports/overview.md               all sites side by side

Usage:
    python batch_audit.py data
    python batch_audit.py data --out reports --notebook dataset_inspection.ipynb
"""

import argparse
import datetime as dt
import html
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".parquet",
                        ".feather", ".json", ".jsonl", ".nc", ".nc4", ".cdf", ".netcdf"}

PAGE_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1180px; padding: 24px 32px 80px; color: #1c1c1c; line-height: 1.5; }
h1 { border-bottom: 3px solid #4f81bd; padding-bottom: 6px; margin-top: 48px; }
h2 { color: #24486b; margin-top: 28px; }
h3 { color: #24486b; }
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
"""


def find_sites(root: Path):
    """Return {site name: [data files]}. Sub-folders are sites; loose files land in '_root'."""
    sites = {}
    loose = sorted(f for f in root.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
    if loose:
        sites[root.name] = loose
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.rglob("*")
                       if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
        if files:
            sites[folder.name] = files
    return sites


def run_notebook(notebook_path: Path, data_file: Path, summary_path: Path, timeout: int):
    """Execute the audit notebook for one data file. Returns (executed notebook, error count)."""
    nb = nbformat.read(notebook_path, as_version=4)

    overrides = "\n".join([
        "# Parameters injected by batch_audit.py",
        f"DATA_PATH = {str(data_file.resolve())!r}",
        f"SUMMARY_JSON_PATH = {str(summary_path.resolve())!r}",
        "SAVE_TEXT_REPORT = False",
        "BATCH_CHILD = True",
        "print('Batch parameters applied for:', DATA_PATH)",
    ])
    # Insert straight after the configuration cell (the first code cell).
    first_code = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code")
    nb.cells.insert(first_code + 1, nbformat.v4.new_code_cell(overrides))

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                            extra_arguments=["--log-level=ERROR"],   # quieter kernel startup
                            resources={"metadata": {"path": str(notebook_path.parent)}})
    client.execute()

    errors = sum(1 for cell in nb.cells for out in cell.get("outputs", [])
                 if out.get("output_type") == "error")
    return nb, errors


def notebook_body_html(nb, exporter: HTMLExporter) -> str:
    body, _ = exporter.from_notebook_node(nb)
    return body


def status_span(status: str) -> str:
    return f'<span class="status-{html.escape(str(status))}">{html.escape(str(status))}</span>'


def file_overview_rows(summaries):
    """Rows for the per-site overview table."""
    rows = []
    for s in summaries:
        period = "-"
        if s.get("time_min"):
            period = f"{str(s['time_min'])[:10]} → {str(s['time_max'])[:10]}"
        rows.append({
            "File": s["file"],
            "Rows": f"{s['rows']:,}",
            "Cols": s["columns"],
            "Missing %": f"{s['missing_overall_pct']:.2f}",
            "Dup rows": f"{s['duplicate_rows']:,}",
            "Entities": s["n_entities"] if s["n_entities"] is not None else "-",
            "Period": period,
            "Frequency": s.get("frequency") or "-",
            "Problems": len(s.get("problems", [])),
        })
    return rows


def markdown_table(rows):
    if not rows:
        return "_no files_\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


def html_table(rows):
    if not rows:
        return "<p><em>no files</em></p>"
    headers = list(rows[0].keys())
    cells = ["<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cells.append("<tr>" + "".join(f"<td>{html.escape(str(r[h]))}</td>" for h in headers) + "</tr>")
    return "<table>" + "".join(cells) + "</table>"


def schema_comparison(summaries):
    """Group the files of a site by their exact column set and list the differences."""
    groups = {}
    for s in summaries:
        groups.setdefault(tuple(s["column_names"]), []).append(s["file"])

    lines = []
    if len(groups) == 1:
        only = next(iter(groups))
        lines.append(f"All {len(summaries)} file(s) share the same {len(only)} columns - "
                     "they can be compared or stacked without renaming.")
    else:
        lines.append(f"The {len(summaries)} file(s) use {len(groups)} different column layouts:")
        for i, (cols, files) in enumerate(groups.items(), start=1):
            lines.append(f"\n**Layout {i}** ({len(cols)} columns) - {', '.join(files)}")
            lines.append("`" + "`, `".join(map(str, cols)) + "`")
        everywhere = set.intersection(*(set(c) for c in groups)) if groups else set()
        anywhere = set().union(*(set(c) for c in groups)) if groups else set()
        differing = sorted(anywhere - everywhere)
        lines.append(f"\nColumns present in every file: {len(everywhere)}")
        lines.append(f"Columns missing from at least one file: "
                     + (", ".join(f"`{c}`" for c in differing) if differing else "none"))

    # Same column name stored with different dtypes across files.
    dtype_map = {}
    for s in summaries:
        for col, dtype in s["dtypes"].items():
            dtype_map.setdefault(col, {}).setdefault(str(dtype), []).append(s["file"])
    clashes = {c: d for c, d in dtype_map.items() if len(d) > 1}
    if clashes:
        lines.append("\n**Same column, different storage type across files (possible parsing problem):**")
        for col, dtypes in list(clashes.items())[:15]:
            detail = "; ".join(f"{dt_} in {', '.join(files)}" for dt_, files in dtypes.items())
            lines.append(f"- `{col}`: {detail}")
    return "\n".join(lines) + "\n"


def write_site_report(site: str, results, out_dir: Path, exporter: HTMLExporter):
    """Write report.html (all cell outputs) and summary.md (quick read) for one site folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    summaries = [r["summary"] for r in results if r["summary"]]

    # ---------------------------------------------------------------- report.html
    toc = "".join(
        f'<li><a href="#file-{i}">{html.escape(r["file"].name)}</a>'
        f'{" <em>(execution errors)</em>" if r["errors"] else ""}</li>'
        for i, r in enumerate(results))

    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>Data audit - {html.escape(site)}</title><style>{PAGE_CSS}</style></head><body>",
             f"<h1>Data quality audit - {html.escape(site)}</h1>",
             f"<p class='meta'>Generated {generated} &middot; {len(results)} file(s) &middot; "
             f"every output of every notebook cell is included below.</p>",
             "<h2>Files in this folder</h2>", html_table(file_overview_rows(summaries)),
             "<h2>Contents</h2>", f"<div class='toc'><ol>{toc}</ol></div>"]

    for i, r in enumerate(results):
        s = r["summary"]
        parts.append(f"<h1 id='file-{i}'>{i + 1}. {html.escape(r['file'].name)}</h1>")
        parts.append(f"<p class='meta'>{html.escape(str(r['file']))}</p>")
        if r["errors"]:
            parts.append(f"<div class='error'><strong>{r['errors']} cell(s) raised an error</strong> "
                         "during execution - the tracebacks are shown inline below.</div>")
        if s:
            score = "".join(
                f"<tr><td style='text-align:left'>{html.escape(str(row['quality_dimension']))}</td>"
                f"<td style='text-align:left'>{status_span(row['status'])}</td>"
                f"<td style='text-align:left'>{html.escape(str(row['evidence']))}</td></tr>"
                for row in s.get("scorecard", []))
            parts.append("<div class='filecard'>"
                         f"<strong>{s['rows']:,} rows &times; {s['columns']} columns</strong> &middot; "
                         f"missing {s['missing_overall_pct']:.2f}% &middot; "
                         f"{s['duplicate_rows']:,} duplicate rows &middot; "
                         f"entities: {s['n_entities'] if s['n_entities'] is not None else 'n/a'} &middot; "
                         f"period: {str(s['time_min'])[:10] if s['time_min'] else 'n/a'} → "
                         f"{str(s['time_max'])[:10] if s['time_max'] else 'n/a'}</div>")
            parts.append("<h2>Scorecard</h2><table><tr><th>Quality dimension</th><th>Status</th>"
                         f"<th>Evidence</th></tr>{score}</table>")
            parts.append("<h2>Final audit</h2><pre>"
                         + html.escape(s.get("final_audit_text", "")) + "</pre>")
        parts.append("<h2>Full notebook output</h2>")
        parts.append(r["body"])

    parts.append("</body></html>")
    (out_dir / "report.html").write_text("".join(parts), encoding="utf-8")

    # ---------------------------------------------------------------- summary.md
    md = [f"# Data quality audit - {site}", "",
          f"_Generated {generated} from {len(results)} file(s). "
          f"Full evidence (every cell output, tables and plots): `report.html`._", "",
          "## Files", "", markdown_table(file_overview_rows(summaries))]

    failed = [r for r in results if r["errors"]]
    if failed:
        md += ["", "## Files that did not run cleanly", ""]
        md += [f"- `{r['file'].name}`: {r['errors']} cell(s) raised an error "
               "(see the traceback in `report.html`)" for r in failed]

    if summaries:
        md += ["", "## Schema comparison across the files of this folder", "", schema_comparison(summaries)]

        all_problems = {}
        for s in summaries:
            for p in s.get("problems", []):
                all_problems.setdefault(p.split(":")[0][:60], []).append(s["file"])
        recurring = {k: v for k, v in all_problems.items() if len(v) > 1}
        if recurring:
            md += ["## Problems that recur across files", ""]
            md += [f"- {k} — in {len(v)} file(s): {', '.join(sorted(set(v)))}"
                   for k, v in list(recurring.items())[:20]]
            md += [""]

    for s in summaries:
        md += [f"## {s['file']}", "",
               f"- Path: `{s['path']}`",
               f"- Size: {s['file_size_mb']} MB | {s['rows']:,} rows x {s['columns']} columns "
               f"({s['n_numeric']} numeric, {s['n_categorical']} categorical, {s['n_datetime']} datetime)",
               f"- Entity column: `{s['entity_column']}`" if s["entity_column"] else "- Entity column: none detected",
               f"- Time column: `{s['time_column']}` ({s['frequency']})" if s["time_column"] else "- Time column: none detected",
               "", "### Scorecard", "",
               markdown_table([{"Quality dimension": r["quality_dimension"],
                                "Status": r["status"],
                                "Evidence": str(r["evidence"]).replace("|", "/")}
                               for r in s.get("scorecard", [])]),
               "### Final audit", "", "```text", s.get("final_audit_text", ""), "```", ""]

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return out_dir / "report.html", out_dir / "summary.md"


def write_overview(site_results, out_root: Path):
    """One markdown file comparing all site folders."""
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for site, results in site_results.items():
        summaries = [r["summary"] for r in results if r["summary"]]
        rows.append({
            "Site folder": site,
            "Files": len(results),
            "Rows (total)": f"{sum(s['rows'] for s in summaries):,}",
            "Missing % (max)": f"{max((s['missing_overall_pct'] for s in summaries), default=0):.2f}",
            "Duplicate rows": f"{sum(s['duplicate_rows'] for s in summaries):,}",
            "Problems": sum(len(s.get("problems", [])) for s in summaries),
            "Failed cells": sum(r["errors"] for r in results),
            "Report": f"[{site}/summary.md]({site}/summary.md)",
        })
    md = ["# Dataset audit overview", "",
          f"_Generated {generated} for {len(site_results)} site folder(s)._", "",
          markdown_table(rows), "",
          "Each site folder has its own `report.html` (every output of every notebook cell, per file) "
          "and `summary.md` (overview, schema comparison and final audit per file).", ""]
    (out_root / "overview.md").write_text("\n".join(md), encoding="utf-8")
    return out_root / "overview.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", help="folder containing one sub-folder per site")
    parser.add_argument("--out", default="reports", help="output folder (default: reports)")
    parser.add_argument("--notebook", default="dataset_inspection.ipynb",
                        help="audit notebook to execute (default: dataset_inspection.ipynb)")
    parser.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    parser.add_argument("--site", action="append",
                        help="only process this site folder (repeatable)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    notebook_path = Path(args.notebook)

    if not data_root.is_dir():
        sys.exit(f"Data folder not found: {data_root}")
    if not notebook_path.is_file():
        sys.exit(f"Notebook not found: {notebook_path}")

    sites = find_sites(data_root)
    if args.site:
        sites = {k: v for k, v in sites.items() if k in set(args.site)}
    if not sites:
        sys.exit(f"No supported data files found under {data_root}")

    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    out_root.mkdir(parents=True, exist_ok=True)
    site_results = {}

    for site, files in sites.items():
        print(f"\n=== {site}: {len(files)} file(s) ===")
        site_dir = out_root / site
        audits_dir = site_dir / "audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for data_file in files:
            print(f"  - {data_file.name} ... ", end="", flush=True)
            summary_path = audits_dir / f"{data_file.stem}.json"
            nb, errors = run_notebook(notebook_path, data_file, summary_path, args.timeout)
            summary = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
            results.append({"file": data_file, "errors": errors, "summary": summary,
                            "body": notebook_body_html(nb, exporter)})
            print("done" if not errors else f"done ({errors} cell error(s))")

        report, summary_md = write_site_report(site, results, site_dir, exporter)
        site_results[site] = results
        print(f"  -> {report}")
        print(f"  -> {summary_md}")

    overview = write_overview(site_results, out_root)
    print(f"\nOverview: {overview}")


if __name__ == "__main__":
    main()
