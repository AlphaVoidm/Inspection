"""End-to-end smoke test for the data audit pipeline.

Run with:  .venv/bin/python tests/test_audit.py

Generates a synthetic multi-format fixture set in a temp dir and runs the
entire pipeline, asserting that the audit (a) never crashes on any file and
(b) produces the expected artifacts.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataaudit import init_audit, run_all  # noqa: E402
from dataaudit.core import Config  # noqa: E402


def make_fixtures(root: str) -> None:
    rng = np.random.default_rng(7)
    countries = ["Egypt", "United States of America", "Germany", "South Africa"]
    iso3 = ["EGY", "USA", "DEU", "ZAF"]
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")

    # 1) monthly demand CSV with missingness, duplicates and a conflict
    rows = []
    for c, code in zip(countries, iso3):
        base = rng.normal(loc=1000 + 100 * len(c), scale=50, size=len(dates))
        for i, d in enumerate(dates):
            rows.append({
                "country": c, "iso3": code, "date": d.strftime("%Y-%m-%d"),
                "demand_mw": round(float(base[i]), 2),
                "temperature_c": round(float(rng.normal(20, 5)), 2),
                "population": int(1_000_000 * len(c)),
            })
    demand = pd.DataFrame(rows)
    # inject missingness (one country's temperature for a block of months)
    mask = (demand["iso3"] == "EGY") & (demand["date"] >= "2018-01-01") & (demand["date"] <= "2018-06-01")
    demand.loc[mask, "temperature_c"] = np.nan
    # inject an exact duplicate row
    demand = pd.concat([demand, demand.iloc[[0]]], ignore_index=True)
    # inject a conflicting duplicate (same country+date, different demand)
    conflict = demand.iloc[1].copy()
    conflict["demand_mw"] = conflict["demand_mw"] * 2
    demand = pd.concat([demand, pd.DataFrame([conflict])], ignore_index=True)
    demand.to_csv(os.path.join(root, "demand.csv"), index=False)

    # 2) multi-sheet Excel generation (data + README sheet)
    gen = demand[["country", "iso3", "date"]].copy()
    gen["generation_mw"] = demand["demand_mw"] * rng.normal(1.0, 0.02, size=len(demand))
    gen["demand_mw"] = demand["demand_mw"] * rng.normal(1.0, 0.01, size=len(demand))
    with pd.ExcelWriter(os.path.join(root, "generation.xlsx"), engine="openpyxl") as xw:
        gen.to_excel(xw, sheet_name="generation", index=False)
        pd.DataFrame({"note": ["source: synthetic test data", "units: MW"]}).to_excel(
            xw, sheet_name="README", index=False)

    # 3) annual GDP CSV (overlaps country/iso3/population)
    gdp_rows = []
    for c, code in zip(countries, iso3):
        for y in range(2015, 2025):
            gdp_rows.append({"country": c, "iso3": code, "year": y,
                             "gdp_usd": int(rng.uniform(1e10, 5e12)),
                             "population": int(1_000_000 * len(c))})
    pd.DataFrame(gdp_rows).to_csv(os.path.join(root, "gdp.csv"), index=False)

    # 4) NetCDF gridded temperature
    try:
        import xarray as xr
        time = pd.date_range("2020-01-01", periods=24, freq="MS")
        lat = np.linspace(-30, 30, 8)
        lon = np.linspace(-20, 40, 12)
        temp = rng.normal(20, 5, size=(len(time), len(lat), len(lon)))
        ds = xr.Dataset(
            {"temperature": (("time", "lat", "lon"), temp,
                             {"units": "degC", "long_name": "surface air temperature"})},
            coords={"time": time, "lat": lat, "lon": lon},
        )
        ds.to_netcdf(os.path.join(root, "climate.nc"))
    except Exception:
        pass

    # 5) JSON records
    import json
    json_records = [{"country": c, "year": y, "value": rng.uniform(0, 100)}
                    for c in countries for y in range(2018, 2023)]
    with open(os.path.join(root, "records.json"), "w") as f:
        json.dump(json_records, f)

    # 6) corrupt parquet (garbage bytes -> loader error, must not abort)
    with open(os.path.join(root, "broken.parquet"), "wb") as f:
        f.write(b"this is not a parquet file" * 100)

    # 7) unsupported extension
    with open(os.path.join(root, "notes.xyz"), "w") as f:
        f.write("some text the audit cannot parse automatically")

    # 8) zip archive containing a CSV and a JSONL
    with zipfile.ZipFile(os.path.join(root, "bundle.zip"), "w") as zf:
        zf.writestr("inner.csv", demand.head(50).to_csv(index=False))
        zf.writestr("inner.jsonl", "\n".join(json.dumps(r) for r in json_records[:10]))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dataaudit_test_")
    try:
        make_fixtures(tmp)
        out = os.path.join(tmp, "out")
        config = Config(data_paths=[tmp], output_dir=out, recursive=True)
        ctx = init_audit(config)
        run_all(ctx)

        problems = []
        def check(cond, msg):
            if not cond:
                problems.append(msg)

        inv = ctx.get_table("01_file_inventory.csv")
        check(inv is not None and not inv.empty, "file inventory empty")
        check("UNSUPPORTED" in set(inv["Status"]) or "UNSUPPORTED" in set(inv["Type"]),
              "unsupported file not flagged")
        # error file recorded
        errs = [e for d in ctx.datasets for e in d.errors]
        check(len(errs) >= 1, "expected at least one load error (broken.parquet)")
        # temporal coverage present for the CSV
        tc = ctx.get_table("04_temporal_coverage.csv")
        check(tc is not None and not tc.empty, "no temporal coverage table")
        if tc is not None and not tc.empty:
            freq_set = set(tc["frequency"])
            check("MONTHLY" in freq_set, f"expected MONTHLY frequency, got {freq_set}")
        # entity detection
        ec = ctx.get_table("05_entity_coverage.csv")
        check(ec is not None and not ec.empty, "no entity coverage table")
        # overlap detection
        ov = ctx.get_table("10_overlap_analysis.csv")
        check(ov is not None and not ov.empty, "no overlap table")
        # quality scores
        q = ctx.get_table("13_quality_scores.csv")
        check(q is not None and not q.empty, "no quality scores")
        # exports exist
        check(os.path.exists(os.path.join(out, "reports", "DATA_AUDIT_REPORT.md")),
              "main report not written")
        check(os.path.exists(os.path.join(out, "01_file_inventory.csv")),
              "inventory CSV not written")

        print("=" * 60)
        print("AUDIT SMOKE TEST")
        print("=" * 60)
        for d in ctx.datasets:
            print(f"  {d.file_id:6s} {d.rel_path:30s} {d.file_type:8s} {d.status:12s} {d.inspection_mode}")
        print("-" * 60)
        if problems:
            print("FAILURES:")
            for p in problems:
                print("  -", p)
            return 1
        print("OK — all checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
