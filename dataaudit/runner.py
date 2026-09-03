"""Pipeline orchestration: stage registry, error isolation, archive handling."""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Dict

import pandas as pd

from .core import AuditContext, Config, Dataset, STAGE_ORDER, STAGE_TITLES
from . import inventory as inv
from . import loaders
from . import profiling
from . import variables
from . import temporal
from . import entities
from . import missingness
from . import duplicates
from . import units_metadata
from . import semantics
from . import numerical
from . import overlap
from . import compare
from . import conflicts
from . import quality
from . import readiness
from . import features
from . import anomaly
from . import redundancy
from .utils import error_record, hash_file

# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------

def _expand_archive(rec: dict, tmp_root: str) -> list:
    """Extract supported members of a ZIP/GZ/BZ2 archive into tmp_root.

    Returns a list of new discovered-file records.
    """
    import zipfile
    import gzip
    import bz2

    out = []
    path = rec["full_path"]
    ext = rec["extension"].lower()

    def new_rec(full, member_name, size):
        inner_ext = os.path.splitext(full)[1].lower()
        ftype, kind, loader = inv.classify_path(full)
        return {
            "index": None,
            "filename": os.path.basename(full),
            "full_path": full,
            "relative_path": f"{rec['relative_path']}::{member_name}",
            "extension": inner_ext,
            "size_bytes": size,
            "size_human": "",
            "modified": None,
            "file_type": ftype,
            "kind": kind,
            "load_method": loader,
            "readable": True,
            "exists": True,
            "rows": None,
            "columns": None,
            "status": "PENDING",
            "archive_members": [],
            "from_archive": rec["relative_path"],
        }

    try:
        if ext == ".zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_ext = os.path.splitext(info.filename)[1].lower()
                    if inner_ext not in inv.ZIP_SUPPORTED:
                        continue
                    target = os.path.join(tmp_root, os.path.basename(info.filename))
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    out.append(new_rec(target, info.filename, info.file_size))
        elif ext in (".gz", ".bz2"):
            inner_name = os.path.splitext(os.path.basename(path))[0]
            target = os.path.join(tmp_root, inner_name)
            opener = gzip.open if ext == ".gz" else bz2.open
            with opener(path, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(new_rec(target, inner_name, os.path.getsize(target)))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _stage_discover(ctx: AuditContext) -> None:
    input_paths, records = inv.discover_files(ctx.config)
    ctx.input_paths = input_paths

    # expand archives into temp dirs
    tmp_root = tempfile.mkdtemp(prefix="dataaudit_archive_")
    ctx.notes.append(f"Archive temp dir: {tmp_root}")
    expanded = []
    for rec in records:
        if rec["file_type"] in ("ZIP", "GZIP", "BZIP2") and rec["exists"]:
            expanded.extend(_expand_archive(rec, tmp_root))
    records.extend(expanded)
    ctx.discovered_files = records

    # create Dataset objects
    for i, rec in enumerate(records):
        d = Dataset(
            file_id=f"ds{i:03d}",
            filename=rec["filename"],
            path=rec["full_path"],
            rel_path=rec["relative_path"],
            extension=rec["extension"],
            size_bytes=rec["size_bytes"],
            modified=rec["modified"],
            sha256=hash_file(rec["full_path"]) if rec.get("exists") else None,
            file_type=rec["file_type"],
            kind=rec["kind"],
            readable=rec["readable"],
            status=rec["status"],
            load_method=rec["load_method"],
        )
        ctx.datasets.append(d)
        ctx.dataset_by_id[d.file_id] = d


def _stage_inventory(ctx: AuditContext) -> None:
    ctx.add_table("01_file_inventory.csv", inv.build_inventory(ctx))
    ctx.log(f"Discovered {len(ctx.discovered_files)} file(s); "
            f"{sum(1 for r in ctx.discovered_files if r['readable'])} readable.")


def _stage_load(ctx: AuditContext) -> None:
    for d in ctx.datasets:
        if d.file_type in ("ZIP", "GZIP", "BZIP2"):
            d.status = "ARCHIVE (expanded)"
            d.readable = False  # the container itself is not an inspectable dataset
            continue
        if d.file_type == "UNSUPPORTED":
            d.status = "UNSUPPORTED"
            continue
        if not os.path.exists(d.path):
            d.status = "NOT FOUND"
            continue
        try:
            loaders.load_dataset(d, ctx.config)
        except Exception as exc:
            d.errors.append(error_record(d.path, "load", exc))
            d.readable = False
            d.status = "ERROR"
    # refresh inventory rows/cols from loaded datasets
    by_path = {d.path: d for d in ctx.datasets}
    for rec in ctx.discovered_files:
        d = by_path.get(rec["full_path"])
        if d is None:
            continue
        rec["status"] = d.status
        if d.readable and d.is_tabular:
            rec["rows"] = d.row_count_display()
            rec["columns"] = d.n_cols
    ctx.add_table("01_file_inventory.csv", inv.build_inventory(ctx))


def _stage_profile(ctx: AuditContext) -> None:
    for d in ctx.datasets:
        if d.is_tabular:
            profiling.profile_dataset(ctx, d)
    ctx.add_table("02_dataset_profiles.csv", profiling.build_dataset_profiles_table(ctx))


def _stage_variables(ctx: AuditContext) -> None:
    ctx.add_table("03_variable_catalog.csv", variables.build_variable_catalog(ctx))
    numerical.run_numerical_audit(ctx)


def _stage_temporal(ctx: AuditContext) -> None:
    temporal.run_temporal_audit(ctx)


def _stage_entities(ctx: AuditContext) -> None:
    entities.run_entity_audit(ctx)
    for d in ctx.tabular_datasets:
        temporal.run_entity_temporal(ctx, d, d.entity_columns)
    # merge per-entity temporal coverage into one table
    by_ent = [d.temporal.get("by_entity") for d in ctx.tabular_datasets
              if d.temporal.get("by_entity") is not None]
    if by_ent:
        ctx.add_table("_entity_temporal.csv", pd.concat(by_ent, ignore_index=True))


def _stage_missingness(ctx: AuditContext) -> None:
    missingness.run_missingness_audit(ctx)


def _stage_duplicates(ctx: AuditContext) -> None:
    duplicates.run_duplicate_audit(ctx)


def _stage_units_metadata(ctx: AuditContext) -> None:
    units_metadata.run_units_metadata_audit(ctx)


def _stage_semantics(ctx: AuditContext) -> None:
    semantics.run_semantics_audit(ctx)


def _stage_overlap(ctx: AuditContext) -> None:
    overlap.run_overlap_detection(ctx)


def _stage_compare(ctx: AuditContext) -> None:
    compare.run_comparison(ctx)


def _stage_conflicts(ctx: AuditContext) -> None:
    conflicts.run_conflict_analysis(ctx)


def _stage_quality(ctx: AuditContext) -> None:
    quality.run_quality_audit(ctx)


def _stage_ts_readiness(ctx: AuditContext) -> None:
    readiness.run_readiness_audit(ctx)


def _stage_panel_readiness(ctx: AuditContext) -> None:
    # already computed in ts_readiness; refresh is idempotent
    if not ctx.completed.get("ts_readiness"):
        readiness.run_readiness_audit(ctx)


def _stage_features(ctx: AuditContext) -> None:
    features.run_feature_availability(ctx)


def _stage_leakage(ctx: AuditContext) -> None:
    features.run_leakage_audit(ctx)
    am = features.annual_monthly_flags(ctx)
    if am is not None and not am.empty:
        ctx.add_table("_annual_monthly_flags.csv", am)


def _stage_anomaly(ctx: AuditContext) -> None:
    anomaly.run_anomaly_audit(ctx)


def _stage_redundancy(ctx: AuditContext) -> None:
    redundancy.run_redundancy_analysis(ctx)


def _stage_readiness(ctx: AuditContext) -> None:
    ctx.add_table("_overall_readiness.csv",
                  pd.DataFrame([{"classification": readiness.overall_readiness(ctx)}]))
    ctx.add_table("_hgtqf.csv", pd.DataFrame([readiness.hgtqf_readiness(ctx)]))


def _stage_recommendations(ctx: AuditContext) -> None:
    recs = readiness.build_recommendations(ctx)
    ctx.add_table("17_recommendation_table.csv", recs)
    dm = readiness.build_decision_matrix(ctx)
    if dm is not None and not dm.empty:
        ctx.add_table("_decision_matrix.csv", dm)


def _stage_export(ctx: AuditContext) -> None:
    from .export import export_all
    export_all(ctx)


STAGE_FUNCS: Dict[str, Callable] = {
    "discover": _stage_discover,
    "inventory": _stage_inventory,
    "load": _stage_load,
    "profile": _stage_profile,
    "variables": _stage_variables,
    "temporal": _stage_temporal,
    "entities": _stage_entities,
    "missingness": _stage_missingness,
    "duplicates": _stage_duplicates,
    "units_metadata": _stage_units_metadata,
    "semantics": _stage_semantics,
    "overlap": _stage_overlap,
    "compare": _stage_compare,
    "conflicts": _stage_conflicts,
    "quality": _stage_quality,
    "ts_readiness": _stage_ts_readiness,
    "panel_readiness": _stage_panel_readiness,
    "features": _stage_features,
    "leakage": _stage_leakage,
    "anomaly": _stage_anomaly,
    "redundancy": _stage_redundancy,
    "readiness": _stage_readiness,
    "recommendations": _stage_recommendations,
    "export": _stage_export,
}


def init_audit(config: Config) -> AuditContext:
    ctx = AuditContext(config=config)
    return ctx


def run_stage(ctx: AuditContext, stage: str) -> None:
    if stage not in STAGE_FUNCS:
        raise KeyError(f"Unknown stage '{stage}'. Available: {list(STAGE_FUNCS)}")
    try:
        STAGE_FUNCS[stage](ctx)
        ctx.completed[stage] = True
    except Exception as exc:  # one failing stage must not kill the audit
        ctx.notes.append(f"STAGE {stage} FAILED: {exc}")
        ctx.completed[stage] = False


def run_all(ctx: AuditContext) -> None:
    for stage in STAGE_ORDER:
        run_stage(ctx, stage)
