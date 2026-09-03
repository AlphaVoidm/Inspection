"""
dataaudit — a general-purpose, multi-source data inspection & audit toolkit.

The toolkit behaves like a forensic data investigator: it *observes*,
*measures*, *flags*, *compares*, *explains* and *recommends* — but it never
silently cleans, merges, imputes, or modifies the source data.

Public API (most users only need the notebook, which calls these):

    from dataaudit import init_audit, run_stage

    ctx = init_audit(config)            # create the audit context
    run_stage(ctx, "discover")          # run one pipeline stage
    run_all(ctx)                        # run the whole pipeline
    ctx.export(output_dir)              # write CSV / Markdown / figures

See README.md for the full stage list and configuration options.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .core import Config, AuditContext, Dataset, STAGE_ORDER
from .runner import run_stage, run_all, init_audit

__all__ = [
    "__version__",
    "Config",
    "AuditContext",
    "Dataset",
    "STAGE_ORDER",
    "run_stage",
    "run_all",
    "init_audit",
]
