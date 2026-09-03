"""Core data structures: configuration, dataset records, audit context."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import now_iso, python_env_summary

# ---------------------------------------------------------------------------
# The pipeline stage order.  Matches the notebook section numbering.
# ---------------------------------------------------------------------------
STAGE_ORDER = [
    "discover",          # 03. File Discovery
    "inventory",         # 04. File Inventory
    "load",              # 05. Safe Loading
    "profile",           # 06. Dataset Profiling
    "variables",         # 07. Variable Catalog (+ column semantics)
    "temporal",          # 08/09. Date/Time Audit & Temporal Coverage
    "entities",          # 10/11. Geographic/Entity Audit & normalization
    "missingness",       # 11b. Missing Data
    "duplicates",        # 12. Duplicate Detection
    "units_metadata",    # 13. Units & Metadata
    "semantics",         # 14. Dataset Semantics (purpose + variable semantics)
    "overlap",           # 15. Overlap Detection
    "compare",           # 16. Cross-Dataset Comparison
    "conflicts",         # 17. Conflict Analysis
    "quality",           # 18. Data Quality scores
    "ts_readiness",      # 19. Time-Series Readiness
    "panel_readiness",   # 20. Panel Readiness
    "features",          # 21. Feature Availability
    "leakage",           # 22. Leakage Audit
    "anomaly",           # 23. Anomaly Detection (outliers + structural breaks)
    "redundancy",        # 24. Redundancy Analysis
    "readiness",         # 25. Overall Data Readiness
    "recommendations",   # 26. Recommended Next Actions
    "export",            # 27. Export Reports
]

STAGE_TITLES = {
    "discover": "03. File Discovery",
    "inventory": "04. File Inventory",
    "load": "05. Safe Loading",
    "profile": "06. Dataset Profiling",
    "variables": "07. Variable Catalog",
    "temporal": "08/09. Date/Time Audit & Temporal Coverage",
    "entities": "10/11. Geographic/Entity Audit",
    "missingness": "11. Missing Data",
    "duplicates": "12. Duplicate Detection",
    "units_metadata": "13. Units & Metadata",
    "semantics": "14. Dataset Semantics",
    "overlap": "15. Overlap Detection",
    "compare": "16. Cross-Dataset Comparison",
    "conflicts": "17. Conflict Analysis",
    "quality": "18. Data Quality",
    "ts_readiness": "19. Time-Series Readiness",
    "panel_readiness": "20. Panel Readiness",
    "features": "21. Feature Availability",
    "leakage": "22. Leakage Audit",
    "anomaly": "23. Anomaly Detection",
    "redundancy": "24. Redundancy Analysis",
    "readiness": "25. Overall Data Readiness",
    "recommendations": "26. Recommended Next Actions",
    "export": "27. Export Reports",
}


@dataclass
class Config:
    """User-facing configuration (spec section 2)."""

    data_paths: List[str] = field(default_factory=list)
    output_dir: str = "data_audit_output"
    recursive: bool = True

    # Sampling / memory limits
    max_rows_full_load: int = 2_000_000          # rows beyond which we sample
    sample_rows: int = 100_000                   # sample size for large files
    max_bytes_full_load: int = 1_500_000_000     # ~1.5 GB safety cap
    preview_rows: int = 8
    max_entities_detail: int = 50                # cap entities listed in reports

    # Optional project-specific requirements (spec section 45/46/47).
    project_config: Optional[Dict[str, Any]] = None

    # Report toggles
    write_figures: bool = True
    write_markdown: bool = True
    write_csv: bool = True

    # Paths actually resolved during discovery (populated at runtime).
    audit_timestamp: str = field(default_factory=now_iso)

    def resolve_project(self) -> Dict[str, Any]:
        pc = self.project_config or {}
        return {
            "target": pc.get("target", []),
            "required_frequency": pc.get("required_frequency"),
            "required_entity_level": pc.get("required_entity_level"),
            "candidate_features": pc.get("candidate_features", []),
            "forecast_horizons": pc.get("forecast_horizons", [12, 36, 60]),
            "context_windows": pc.get("context_windows", [24, 60, 120]),
            "quantile_forecasting": pc.get("quantile_forecasting", False),
            "cross_country_modeling": pc.get("cross_country_modeling", False),
            "loco_readiness": pc.get("loco_readiness", False),
            "zero_shot_country": pc.get("zero_shot_country", False),
            "scenario_readiness": pc.get("scenario_readiness", False),
        }


@dataclass
class Dataset:
    """Everything the audit learns about one discovered file/dataset."""

    # ---- identity (from inventory) ----
    file_id: str = ""
    filename: str = ""
    path: str = ""                       # full path
    rel_path: str = ""
    extension: str = ""
    size_bytes: Optional[int] = None
    modified: Optional[str] = None
    sha256: Optional[str] = None

    # ---- discovery/loading status ----
    file_type: str = "UNKNOWN"           # CSV / XLSX / Parquet / NetCDF / JSON / ...
    kind: str = "UNSUPPORTED"            # tabular / semi-structured / scientific / text / archive / unsupported
    readable: bool = False
    status: str = "UNKNOWN"              # OK / ERROR / UNSUPPORTED / EMPTY ...
    load_method: str = ""
    inspection_mode: str = "NOT INSPECTED"   # FULL / SAMPLE / METADATA-ONLY / NOT INSPECTED
    errors: List[dict] = field(default_factory=list)

    # ---- tabular structure ----
    n_rows: Optional[int] = None         # exact row count when known, else sample/None
    n_cols: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)
    memory_bytes: Optional[int] = None

    # ---- data holder ----
    df: Any = None                       # pandas DataFrame (sampled for large files)
    df_is_full: bool = False
    df_exact_rows: bool = False

    # ---- scientific data holder ----
    xr: Any = None                       # xarray Dataset / DataArray
    scientific_info: Dict[str, Any] = field(default_factory=dict)

    # ---- metadata ----
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- derived results (each stage fills these) ----
    profile: Any = None                  # per-column profile DataFrame
    variable_catalog: Any = None
    semantic_roles: Any = None           # per-column role DataFrame
    temporal: Dict[str, Any] = field(default_factory=dict)
    entity_columns: List[str] = field(default_factory=list)
    entity_report: Any = None
    country_mapping: Any = None
    missingness: Dict[str, Any] = field(default_factory=dict)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    unit_catalog: Any = None
    purpose: Dict[str, Any] = field(default_factory=dict)
    variable_semantics: Any = None
    numerical_audit: Any = None
    anomaly_report: Any = None
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    usability: Dict[str, Any] = field(default_factory=dict)
    ts_readiness: Dict[str, Any] = field(default_factory=dict)
    panel_readiness: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tabular(self) -> bool:
        return self.kind == "tabular" and self.df is not None

    @property
    def is_scientific(self) -> bool:
        return self.kind == "scientific"

    @property
    def display_name(self) -> str:
        return self.rel_path or self.filename or self.path

    def row_count_display(self) -> str:
        if self.n_rows is None:
            return "N/A"
        if self.df_exact_rows:
            return f"{self.n_rows:,}"
        return f"~{self.n_rows:,} (sampled)"


@dataclass
class AuditContext:
    """Mutable context threaded through every pipeline stage."""

    config: Config
    started_at: str = field(default_factory=now_iso)

    # resolved input files (list of path strings)
    input_paths: List[str] = field(default_factory=list)
    discovered_files: List[dict] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)
    dataset_by_id: Dict[str, Dataset] = field(default_factory=dict)

    # global (cross-dataset) result tables, keyed by export basename
    tables: Dict[str, Any] = field(default_factory=dict)
    # free-form markdown notes appended to the final report
    notes: List[str] = field(default_factory=list)
    # stage completion bookkeeping
    completed: Dict[str, bool] = field(default_factory=dict)

    env: Dict[str, Any] = field(default_factory=python_env_summary)

    def log(self, message: str) -> None:
        self.notes.append(message)

    def add_table(self, key: str, df: Any) -> Any:
        self.tables[key] = df
        return df

    def get_table(self, key: str, default: Any = None) -> Any:
        return self.tables.get(key, default)

    def dataset(self, file_id: str) -> Optional[Dataset]:
        return self.dataset_by_id.get(file_id)

    @property
    def readable_datasets(self) -> List[Dataset]:
        return [d for d in self.datasets if d.readable]

    @property
    def tabular_datasets(self) -> List[Dataset]:
        return [d for d in self.datasets if d.is_tabular]

    @property
    def scientific_datasets(self) -> List[Dataset]:
        return [d for d in self.datasets if d.is_scientific]


# Default table list for the export stage (spec section 40).
DEFAULT_EXPORTS = [
    "01_file_inventory.csv",
    "02_dataset_profiles.csv",
    "03_variable_catalog.csv",
    "04_temporal_coverage.csv",
    "05_entity_coverage.csv",
    "06_missingness.csv",
    "07_duplicate_report.csv",
    "08_unit_catalog.csv",
    "09_metadata_report.csv",
    "10_overlap_analysis.csv",
    "11_cross_dataset_comparison.csv",
    "12_conflict_report.csv",
    "13_quality_scores.csv",
    "14_feature_availability.csv",
    "15_leakage_flags.csv",
    "16_anomaly_report.csv",
    "17_recommendation_table.csv",
]
