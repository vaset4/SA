"""Oracle Retrieval 与失败归因评测包。"""

from .attribution import attribute_pair
from .contracts import (
    AttributionLabel,
    OracleAttribution,
    OraclePairRecord,
    OraclePairSpec,
    OracleSideRun,
    make_pair_spec,
)
from .report import build_oracle_report, render_oracle_report, write_oracle_report
from .runner import run_oracle_pair, run_oracle_pair_isolated
from .source import build_oracle_evidence

__all__ = [
    "AttributionLabel", "OracleAttribution", "OraclePairRecord", "OraclePairSpec",
    "OracleSideRun", "attribute_pair", "build_oracle_evidence", "build_oracle_report",
    "make_pair_spec", "render_oracle_report", "run_oracle_pair",
    "run_oracle_pair_isolated", "write_oracle_report",
]
