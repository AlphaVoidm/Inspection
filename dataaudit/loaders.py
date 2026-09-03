"""Safe, memory-aware loading of discovered files (spec section 5).

Never writes anywhere.  Large files are sampled rather than fully loaded, and
the inspection mode is always reported (FULL / SAMPLE / METADATA-ONLY).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from .core import AuditContext, Config, Dataset
from .utils import df_memory, error_string

# A fixed RNG seed makes sampling reproducible.
RNG = np.random.default_rng(42)


def _should_sample(config: Config, n_rows: Optional[int], size_bytes: Optional[int]) -> bool:
    if n_rows is not None and n_rows > config.max_rows_full_load:
        return True
    if size_bytes is not None and size_bytes > config.max_bytes_full_load:
        return True
    return False


def _sample_frame(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Stratified-ish random sample of rows (head/tail/middle always useful)."""
    n = len(df)
    if n <= config.sample_rows:
        return df
    n_sample = min(config.sample_rows, n)
    # keep some of the head and tail, sample the middle
    head = df.iloc[: max(1, n_sample // 10)]
    tail = df.iloc[-max(1, n_sample // 10):]
    middle = df.iloc[max(1, n_sample // 10): -max(1, n_sample // 10)]
    if len(middle) > 0:
        k = max(0, n_sample - len(head) - len(tail))
        idx = RNG.choice(len(middle), size=min(k, len(middle)), replace=False)
        mid_sample = middle.iloc[np.sort(idx)]
        out = pd.concat([head, mid_sample, tail], axis=0)
    else:
        out = df
    return out.reset_index(drop=True)


def _detect_sep_and_read_csv(path: str, config: Config, kind: str) -> tuple:
    """Read a CSV/TSV/TXT file, sampling if large. Returns (df, meta)."""
    sep = "," if kind == "CSV" else ("\t" if kind in ("TSV",) else None)

    # count lines first (cheap) for CSV/TSV/TXT
    n_lines = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            n_lines = sum(1 for _ in f)
    except Exception:
        n_lines = None

    approx_rows = max(0, (n_lines - 1)) if n_lines is not None else None

    df = None
    df_is_full = False
    df_exact_rows = False
    meta: dict = {"engine": "pandas.read_csv", "n_lines": n_lines}

    # sniff delimiter for .txt files
    if sep is None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = "".join(f.readline() for _ in range(5))
            import csv as _csv
            sep = _csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
            meta["sep"] = sep
        except Exception:
            sep = ","

    try:
        if _should_sample(config, approx_rows, os.path.getsize(path)):
            # sample lines uniformly
            n_rows_sample = config.sample_rows
            skip = sorted(RNG.choice(max(1, approx_rows), size=max(0, approx_rows - n_rows_sample), replace=False)) \
                if approx_rows and approx_rows > n_rows_sample else []
            if not skip:
                df = pd.read_csv(path, sep=sep, low_memory=False)
                df_is_full = True
            else:
                df = pd.read_csv(path, sep=sep, low_memory=False, skiprows=skip)
                df_is_full = False
            df_exact_rows = df_is_full
        else:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            df_is_full = True
            df_exact_rows = True
    except pd.errors.ParserError as exc:
        # fall back to a more permissive read (e.g. ragged lines)
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False, on_bad_lines="skip", engine="python")
            meta["warning"] = f"parser fallback used (on_bad_lines=skip): {exc}"
            df_is_full = True
            df_exact_rows = True
        except Exception as exc2:
            raise exc2
    return df, meta, df_is_full, df_exact_rows


def _load_excel(path: str, config: Config) -> tuple:
    """Load every sheet of an Excel file (spec section 17).

    Returns (df, meta) where df is the first DATA-like sheet's frame and meta
    contains per-sheet information.
    """
    import openpyxl  # noqa: F401  (ensures engine availability)
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        xls = pd.ExcelFile(path)

    sheets = {}
    for name in xls.sheet_names:
        try:
            sdf = xls.parse(sheet_name=name, nrows=config.sample_rows + 1)
        except Exception as exc:
            sheets[name] = {"error": error_string(exc), "df": None}
            continue
        truncated = len(sdf) > config.sample_rows
        if truncated:
            sdf = sdf.iloc[: config.sample_rows]
        sheets[name] = {
            "df": sdf,
            "n_rows_loaded": len(sdf),
            "truncated": truncated,
            "shape": sdf.shape,
        }

    meta = {
        "engine": "pandas.ExcelFile(openpyxl)",
        "sheet_names": xls.sheet_names,
        "n_sheets": len(xls.sheet_names),
        "sheets": sheets,
    }

    # choose a primary sheet: prefer one that looks like data (has headers,
    # more rows/cols), fall back to the first sheet.
    primary_df = None
    primary_name = None
    best_score = -1
    for name, info in sheets.items():
        df = info.get("df")
        if df is None or df.empty:
            continue
        score = (len(df.columns) * 2) + (len(df) > 1) * 5
        if df.columns[0] is not None and str(df.columns[0]) != "0":
            score += 1
        if score > best_score:
            best_score = score
            primary_df = df
            primary_name = name
    if primary_df is None and sheets:
        primary_name = xls.sheet_names[0]
        primary_df = sheets[primary_name].get("df")
    meta["primary_sheet"] = primary_name
    return primary_df, meta, False, False


def _load_parquet(path: str, config: Config, kind: str) -> tuple:
    if kind == "Feather":
        df = pd.read_feather(path)
        return df, {"engine": "pandas.read_feather"}, True, True
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    meta = {
        "engine": "pyarrow.parquet",
        "num_rows": n_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "num_columns": pf.metadata.num_columns,
        "schema": pf.schema_arrow.to_string() if hasattr(pf.schema_arrow, "to_string") else str(pf.schema_arrow),
    }
    if _should_sample(config, n_rows, os.path.getsize(path)):
        df = pf.read().to_pandas()
        df = _sample_frame(df, config)
        return df, meta, False, False
    df = pf.read().to_pandas()
    return df, meta, True, True


def _load_json(path: str, config: Config, kind: str) -> tuple:
    if kind == "JSONL":
        n_lines = None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                n_lines = sum(1 for _ in f)
        except Exception:
            pass
        if _should_sample(config, n_lines, os.path.getsize(path)):
            df = pd.read_json(path, lines=True, nrows=config.sample_rows)
            return df, {"engine": "pandas.read_json(lines=True)", "n_lines": n_lines, "sampled": True}, False, False
        df = pd.read_json(path, lines=True)
        return df, {"engine": "pandas.read_json(lines=True)", "n_lines": n_lines}, True, True

    # .json (records / object). Read raw to inspect structure.
    meta: dict = {"engine": "json + pandas", "structure": None}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = json.load(f)
    def _describe(obj, depth=0):
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return {"type": "object", "keys": keys[:200], "n_keys": len(keys),
                    "sample": {k: _describe(obj[k], depth + 1) for k in keys[:3]} if depth < 2 else None}
        if isinstance(obj, list):
            return {"type": "array", "length": len(obj), "sample": _describe(obj[0], depth + 1) if obj and depth < 2 else None}
        return {"type": type(obj).__name__, "value": str(obj)[:50]}
    meta["structure"] = _describe(raw)

    try:
        df = pd.json_normalize(raw, max_level=1)
    except Exception:
        df = pd.DataFrame(raw)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if _should_sample(config, len(df), os.path.getsize(path)):
        df = _sample_frame(df, config)
        return df, meta, False, False
    return df, meta, True, True


def _load_txt(path: str, config: Config) -> tuple:
    """Plain text: read as lines; also try a tabular parse as a fallback."""
    meta: dict = {"engine": "text"}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception as exc:
        raise exc
    meta["n_lines"] = len(lines)
    if len(lines) == 0:
        df = pd.DataFrame({"line": []})
        return df, meta, True, True
    # try tabular parse (single-column data or delimited text)
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        if df.shape[1] >= 1 and len(df) > 1:
            meta["engine"] = "pandas.read_csv(sep=sniffed)"
            return df, meta, True, True
    except Exception:
        pass
    df = pd.DataFrame({"line": lines[: config.sample_rows]})
    return df, meta, (len(lines) <= config.sample_rows), (len(lines) <= config.sample_rows)


def _load_netcdf(path: str, config: Config) -> tuple:
    """Load NetCDF/HDF5 metadata + optionally data via xarray (lazy)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        info = {
            "engine": "xarray.open_dataset",
            "dims": dict(ds.sizes),
            "coords": list(ds.coords),
            "data_vars": list(ds.data_vars),
            "attrs": dict(ds.attrs),
        }
        var_info = {}
        for name, var in ds.variables.items():
            vd = {
                "dims": list(var.dims),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "attrs": {k: str(v)[:200] for k, v in var.attrs.items()},
            }
            try:
                vd["units"] = var.attrs.get("units")
                vd["long_name"] = var.attrs.get("long_name", var.attrs.get("standard_name"))
            except Exception:
                pass
            var_info[name] = vd
        info["variables"] = var_info
        # temporal coverage if a time coordinate exists
        try:
            for coord in ("time", "Time", "TIME"):
                if coord in ds.coords:
                    t = ds[coord]
                    if np.issubdtype(t.dtype, np.datetime64):
                        info["time_start"] = str(t.values.min())
                        info["time_end"] = str(t.values.max())
                        info["time_n"] = int(t.size)
                        info["time_dtype"] = str(t.dtype)
                    break
        except Exception:
            pass
        # spatial coverage
        for coord in ("lat", "latitude", "y"):
            if coord in ds.coords:
                try:
                    v = ds[coord].values
                    info["lat_min"], info["lat_max"] = float(np.nanmin(v)), float(np.nanmax(v))
                except Exception:
                    pass
                break
        for coord in ("lon", "longitude", "x"):
            if coord in ds.coords:
                try:
                    v = ds[coord].values
                    info["lon_min"], info["lon_max"] = float(np.nanmin(v)), float(np.nanmax(v))
                except Exception:
                    pass
                break
        # aggregate the primary variable to a small sample table for preview
        df = None
        try:
            primary = None
            for v in ds.data_vars:
                if v not in ("time_bnds", "lat_bnds", "lon_bnds"):
                    primary = v
                    break
            if primary:
                da = ds[primary]
                if "time" in da.dims and da.sizes.get("time", 0) > 1:
                    df = da.isel(time=slice(0, min(10, da.sizes["time"]))).to_pandas().reset_index()
                elif da.ndim <= 2:
                    df = da.isel({d: slice(0, min(100, da.sizes[d])) for d in da.dims}).to_pandas()
        except Exception:
            pass
        return ds, info, df


def load_dataset(ds: Dataset, config: Config) -> None:
    """Populate a Dataset's data holder (df / xr) and inspection mode."""
    ds.errors = []
    try:
        if ds.file_type in ("CSV", "TSV"):
            df, meta, full, exact = _detect_sep_and_read_csv(ds.path, config, ds.file_type)
        elif ds.file_type == "TXT":
            df, meta, full, exact = _load_txt(ds.path, config)
        elif ds.file_type in ("XLSX", "XLS", "XLSM"):
            df, meta, full, exact = _load_excel(ds.path, config)
        elif ds.file_type in ("Parquet", "Feather"):
            df, meta, full, exact = _load_parquet(ds.path, config, ds.file_type)
        elif ds.file_type in ("JSON", "JSONL"):
            df, meta, full, exact = _load_json(ds.path, config, ds.file_type)
        elif ds.file_type == "NetCDF":
            xr_ds, info, df = _load_netcdf(ds.path, config)
            ds.xr = xr_ds
            ds.scientific_info = info
            ds.metadata = info
            ds.df = df
            ds.inspection_mode = "METADATA-ONLY" if df is None else "SAMPLE"
            ds.readable = True
            ds.status = "OK"
            ds.n_rows = None
            ds.n_cols = len(info.get("data_vars", []))
            return
        elif ds.file_type == "HDF5":
            # try pandas.read_hdf (PyTables may be absent); else xarray/h5py
            df, meta, full, exact = _load_hdf5(ds.path, config)
        else:
            ds.readable = False
            ds.status = "UNSUPPORTED"
            ds.inspection_mode = "NOT INSPECTED"
            return

        ds.df = df
        ds.metadata.update(meta)
        ds.df_is_full = full
        ds.df_exact_rows = exact
        if df is None:
            ds.inspection_mode = "METADATA-ONLY"
            ds.readable = True
            ds.status = "OK"
            return
        if exact:
            ds.n_rows = len(df)
        else:
            ds.n_rows = len(df)
        ds.n_cols = df.shape[1]
        ds.columns = [str(c) for c in df.columns]
        ds.dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
        ds.memory_bytes = df_memory(df)
        ds.inspection_mode = "FULL" if (full and exact) else "SAMPLE"
        ds.readable = True
        ds.status = "OK"
    except Exception as exc:
        ds.errors.append({
            "file": ds.path, "stage": "load", "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        })
        ds.readable = False
        ds.status = "ERROR"
        ds.inspection_mode = "NOT INSPECTED"


def _load_hdf5(path: str, config: Config) -> tuple:
    """HDF5: prefer pandas.read_hdf for tabular; else h5py metadata."""
    meta: dict = {"engine": "pandas.read_hdf"}
    df = None
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            meta["keys"] = keys
            if keys:
                df = store[keys[0]]
                meta["key_used"] = keys[0]
                return df, meta, True, True
    except Exception:
        pass
    # h5py metadata path
    try:
        import h5py
        with h5py.File(path, "r") as f:
            def _walk(name, obj):
                out = {"type": type(obj).__name__}
                if isinstance(obj, h5py.Dataset):
                    out.update({"shape": list(obj.shape), "dtype": str(obj.dtype)})
                return out
            structure = {}
            f.visititems(lambda n, o: structure.setdefault(n, _walk(n, o)))
            meta = {"engine": "h5py", "structure": structure}
    except Exception:
        pass
    return df, meta, False, False
