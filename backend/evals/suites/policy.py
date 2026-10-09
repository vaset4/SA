"""显式冻结工作流与检索测评范围。作者：xxx。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

WORKFLOW_POLICY_VERSION = "workflow-retrieval-v2"
RETRIEVAL_GRADER_VERSION = "ragas-retrieval-v2"
RETRIEVAL_METRICS = ("context_precision", "context_recall")


class MeasurementPolicy(BaseModel):
    """新运行必须明确声明此策略；缺少策略的历史配置保持原口径。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal["workflow-retrieval-v1", "workflow-retrieval-v2"]
    ragas_metrics: tuple[Literal["context_precision"], Literal["context_recall"]]
    disabled_metrics: tuple[Literal["faithfulness"], Literal["factual_correctness"], Literal["answer_relevancy"]]
    workflow_grader_version: Literal["agent-workflow-v1", "agent-workflow-v2"]
    retrieval_grader_version: Literal["ragas-retrieval-v2"]

    @model_validator(mode="after")
    def validate_workflow_version(self) -> MeasurementPolicy:
        """无额外参数；评分政策与工作流语义版本必须配对，旧字段不得静默升级。"""
        expected = {"workflow-retrieval-v1": "agent-workflow-v1", "workflow-retrieval-v2": "agent-workflow-v2"}
        if self.workflow_grader_version != expected[self.version]:
            raise ValueError("workflow_policy_grader_version_mismatch")
        return self


def workflow_policy(*, version: Literal["workflow-retrieval-v1", "workflow-retrieval-v2"] = "workflow-retrieval-v2") -> MeasurementPolicy:
    """参数为显式策略版本；新建默认V2，历史配置可明确选择V1且保持原始序列化。"""
    return MeasurementPolicy(version=version, ragas_metrics=RETRIEVAL_METRICS,
                             disabled_metrics=("faithfulness", "factual_correctness", "answer_relevancy"),
                             workflow_grader_version="agent-workflow-v1" if version == "workflow-retrieval-v1" else "agent-workflow-v2",
                             retrieval_grader_version="ragas-retrieval-v2")
