"""File discovery and file inventory (spec sections 3, 4)."""
from __future__ import annotations

import datetime as _dt
import os
import zipfile
from typing import Dict, List, Optional

import pandas as pd

from .core import AuditContext, Config
from .utils import fmt_bytes, hash_file

# ---------------------------------------------------------------------------
# Extension -> (file_type label, dataset kind, loader name)
# ---------------------------------------------------------------------------
FORMATS = {
    ".csv": ("CSV", "tabular", "pandas.read_csv"),
    ".tsv": ("TSV", "tabular", "pandas.read_csv(sep='\\t')"),
    ".txt": ("TXT", "text", "text/pandas.read_csv"),
    ".xlsx": ("XLSX", "tabular", "pandas.read_excel(openpyxl)"),
    ".xls": ("XLS", "tabular", "pandas.read_excel(xlrd)"),
    ".xlsm": ("XLSM", "tabular", "pandas.read_excel(openpyxl)"),
    ".parquet": ("Parquet", "tabular", "pandas.read_parquet(pyarrow)"),
    ".pq": ("Parquet", "tabular", "pandas.read_parquet(pyarrow)"),
    ".feather": ("Feather", "tabular", "pandas.read_feather(pyarrow)"),
    ".json": ("JSON", "semi-structured", "pandas.read_json / json"),
    ".jsonl": ("JSONL", "semi-structured", "pandas.read_json(lines=True)"),
    ".ndjson": ("JSONL", "semi-structured", "pandas.read_json(lines=True)"),
    ".nc": ("NetCDF", "scientific", "xarray.open_dataset"),
    ".nc4": ("NetCDF", "scientific", "xarray.open_dataset"),
    ".cdf": ("NetCDF", "scientific", "xarray.open_dataset"),
    ".h5": ("HDF5", "scientific", "xarray/pandas.read_hdf"),
    ".hdf5": ("HDF5", "scientific", "xarray/pandas.read_hdf"),
    ".hdf": ("HDF5", "scientific", "xarray/pandas.read_hdf"),
    ".zip": ("ZIP", "archive", "zipfile + recursive load"),
    ".gz": ("GZIP", "archive", "gzip"),
    ".bz2": ("BZIP2", "archive", "bz2"),
}

ARCHIVE_EXTS = {".zip", ".gz", ".bz2"}

# ZIP member extensions we can inspect directly
ZIP_SUPPORTED = {
    ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl",
    ".nc", ".h5", ".hdf5", ".feather",
}


def classify_path(path: str) -> tuple:
    """Return (file_type, kind, loader) for a path; UNKNOWN if unsupported."""
    ext = os.path.splitext(path)[1].lower()
    if ext in FORMATS:
        return FORMATS[ext]
    return ("UNSUPPORTED", "unsupported", "")


def _resolve_input_paths(config: Config) -> List[str]:
    """Expand DATA_PATHS entries into a concrete list of files.

    - A file path is taken as-is (regardless of extension, so a known file
      with an odd extension can still be audited).
    - A directory is walked (recursively if config.recursive), collecting
      files with recognized extensions plus any explicitly unsupported file
      so it can be reported as UNSUPPORTED rather than silently ignored.
    """
    resolved: List[str] = []
    for raw in config.data_paths:
        p = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isfile(p):
            resolved.append(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv", "node_modules")]
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    if config.recursive or os.path.dirname(fp) == p:
                        resolved.append(fp)
                if not config.recursive:
                    break
        else:
            resolved.append(p)  # keep it so we can report "NOT FOUND"
    # de-duplicate while preserving order
    seen = set()
    out = []
    for p in resolved:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _archive_members(path: str) -> List[dict]:
    """List supported members inside a ZIP archive."""
    members = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = os.path.splitext(info.filename)[1].lower()
                if ext in ZIP_SUPPORTED:
                    members.append({
                        "name": info.filename,
                        "size": info.file_size,
                        "ext": ext,
                    })
    except Exception:
        pass
    return members


def discover_files(config: Config) -> tuple:
    """Discover files and return (input_paths, discovered_records)."""
    input_paths = _resolve_input_paths(config)
    records = []

    for idx, path in enumerate(input_paths):
        rec = {
            "index": idx + 1,
            "filename": os.path.basename(path),
            "full_path": path,
            "relative_path": os.path.relpath(path, os.getcwd()),
            "extension": os.path.splitext(path)[1].lower(),
            "size_bytes": None,
            "size_human": "",
            "modified": None,
            "file_type": "UNSUPPORTED",
            "kind": "unsupported",
            "load_method": "",
            "readable": False,
            "exists": False,
            "rows": None,
            "columns": None,
            "status": "UNKNOWN",
        }

        if not os.path.exists(path):
            rec["status"] = "NOT FOUND"
            records.append(rec)
            continue

        rec["exists"] = True
        try:
            st = os.stat(path)
            rec["size_bytes"] = st.st_size
            rec["size_human"] = fmt_bytes(st.st_size)
            rec["modified"] = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            pass

        ftype, kind, loader = classify_path(path)
        rec["file_type"] = ftype
        rec["kind"] = kind
        rec["load_method"] = loader

        if ftype == "UNSUPPORTED":
            rec["status"] = "UNSUPPORTED"
            rec["readable"] = False
        else:
            rec["status"] = "PENDING"
            # readability: can we open for reading? (archives checked separately)
            try:
                with open(path, "rb"):
                    pass
                rec["readable"] = True
            except OSError:
                rec["readable"] = False
                rec["status"] = "UNREADABLE"

        rec["archive_members"] = _archive_members(path) if ftype == "ZIP" else []
        records.append(rec)

    return input_paths, records


def build_inventory(ctx: AuditContext) -> pd.DataFrame:
    """Build the FILE INVENTORY table (spec section 4)."""
    records = ctx.discovered_files
    rows = []
    for r in records:
        rows.append({
            "File": r["filename"],
            "Type": r["file_type"],
            "Size": r["size_human"],
            "Size_bytes": r["size_bytes"],
            "Rows": r["rows"],
            "Columns": r["columns"],
            "Readable": "YES" if r["readable"] else "NO",
            "Status": r["status"],
            "Modified": r["modified"],
            "Relative path": r["relative_path"],
        })
    df = pd.DataFrame(rows)
    return df


def inventory_summary(ctx: AuditContext) -> dict:
    recs = ctx.discovered_files
    return {
        "n_input_paths": len(ctx.input_paths),
        "n_files": len(recs),
        "n_found": sum(1 for r in recs if r["exists"]),
        "n_not_found": sum(1 for r in recs if not r["exists"]),
        "n_readable": sum(1 for r in recs if r["readable"]),
        "n_unsupported": sum(1 for r in recs if r["file_type"] == "UNSUPPORTED"),
        "n_archives": sum(1 for r in recs if r["file_type"] == "ZIP"),
    }
