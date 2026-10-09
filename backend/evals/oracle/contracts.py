"""Oracle 配对实验的冻结合同与输入隔离。作者：xxx。"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from evals.contracts import snapshot, snapshot_hash, validate_execution_input

ORACLE_SCHEMA_VERSION: Literal["oracle-pair-v1"] = "oracle-pair-v1"
AttributionLabel = Literal[
    "retrieval_gap", "selection_gap", "downstream_use_candidate", "unknown",
]
MeasurementStatus = Literal["measured", "missing", "invalid_input", "not_measured"]


class FrozenModel(BaseModel):
    """Oracle 合同的不可变基础模型。"""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OraclePairSpec(FrozenModel):
    """固定同一案例的控制条件、真值身份和两侧缓存命名空间。"""

    schema_version: Literal["oracle-pair-v1"] = ORACLE_SCHEMA_VERSION
    pair_id: str
    case_id: str
    case_version: str
    input_snapshot: str
    truth_snapshot: str
    control_snapshot: str
    index_snapshot_id: str
    actual_cache_namespace: str
    oracle_cache_namespace: str

    @field_validator("input_snapshot", "truth_snapshot", "control_snapshot")
    @classmethod
    def canonical_snapshot(cls, value: str) -> str:
        """参数为 JSON 对象快照；返回规范化文本。"""
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("oracle snapshot must contain an object")
        return snapshot(payload)

    @model_validator(mode="after")
    def validate_pair(self) -> OraclePairSpec:
        """无额外参数；验证身份和缓存命名空间互不相同。"""
        if not self.pair_id.strip() or not self.case_id.strip() or not self.case_version.strip():
            raise ValueError("oracle pair identity must be non-empty")
        if not self.index_snapshot_id.strip():
            raise ValueError("oracle index snapshot identity is required")
        if not self.actual_cache_namespace.strip() or not self.oracle_cache_namespace.strip():
            raise ValueError("oracle cache namespaces are required")
        if self.actual_cache_namespace == self.oracle_cache_namespace:
            raise ValueError("actual and oracle cache namespaces must differ")
        validate_execution_input(json.loads(self.input_snapshot))
        return self


class OracleSideRun(FrozenModel):
    """冻结 Actual 或 Oracle 一侧的实际运行与阶段观察。"""

    side: Literal["actual", "oracle"]
    run_id: str
    run_mode: Literal["live", "oracle"]
    retrieval_mode: Literal["actual", "oracle"]
    input_snapshot: str
    observation_snapshot: str
    cache_namespace: str
    status: Literal["completed", "failed", "invalid_input"]
    missing_fields: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("input_snapshot", "observation_snapshot")
    @classmethod
    def validate_side_snapshot(cls, value: str) -> str:
        """参数为一侧 JSON 快照；验证对象形状并规范化。"""
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("oracle side snapshot must contain an object")
        return snapshot(payload)


class OracleAttribution(FrozenModel):
    """保存基于阶段观察的多候选归因，不宣称唯一根因。"""

    status: MeasurementStatus
    labels: tuple[AttributionLabel, ...] = ()
    reason: str
    evidence: tuple[str, ...] = ()


class OraclePairRecord(FrozenModel):
    """保存 Actual/Oracle 配对、控制校验和派生诊断。"""

    schema_version: Literal["oracle-pair-v1"] = ORACLE_SCHEMA_VERSION
    pair: OraclePairSpec
    actual: OracleSideRun
    oracle: OracleSideRun
    attribution: OracleAttribution
    control_hash: str

    @model_validator(mode="after")
    def validate_sides(self) -> OraclePairRecord:
        """无额外参数；验证两侧身份、输入和模式与配对合同一致。"""
        if self.actual.side != "actual" or self.oracle.side != "oracle":
            raise ValueError("oracle pair sides are invalid")
        if self.actual.retrieval_mode != "actual" or self.oracle.retrieval_mode != "oracle":
            raise ValueError("oracle retrieval modes are invalid")
        if self.actual.input_snapshot != self.pair.input_snapshot or self.oracle.input_snapshot != self.pair.input_snapshot:
            raise ValueError("actual and oracle input snapshots must match pair input")
        if self.actual.cache_namespace != self.pair.actual_cache_namespace or self.oracle.cache_namespace != self.pair.oracle_cache_namespace:
            raise ValueError("oracle side cache namespace mismatch")
        expected_hash = snapshot_hash(self.pair.control_snapshot)
        if self.control_hash != expected_hash:
            raise ValueError("oracle control snapshot checksum mismatch")
        return self


def make_pair_spec(
    *,
    pair_id: str,
    case_id: str,
    case_version: str,
    input_payload: Mapping[str, Any],
    truth_payload: Mapping[str, Any],
    control_payload: Mapping[str, Any],
    index_snapshot_id: str,
) -> OraclePairSpec:
    """参数为案例输入、真值和控制条件；生成两侧隔离的配对合同。"""
    return OraclePairSpec(
        pair_id=pair_id,
        case_id=case_id,
        case_version=case_version,
        input_snapshot=snapshot(dict(input_payload)),
        truth_snapshot=snapshot(dict(truth_payload)),
        control_snapshot=snapshot(dict(control_payload)),
        index_snapshot_id=index_snapshot_id,
        actual_cache_namespace=f"{pair_id}:actual",
        oracle_cache_namespace=f"{pair_id}:oracle",
    )
