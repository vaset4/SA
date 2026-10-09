"""Failure trace helpers for planner validation."""

from __future__ import annotations

from app.models.research_planner_contracts import (
    PlannerFailure,
    PlannerValidationResult,
)


def failure_from_result(
    result: PlannerValidationResult,
    *,
    original: PlannerValidationResult | None = None,
) -> PlannerValidationResult:
    """Attach original validation issues when a repair attempt also fails."""
    if result.failure is not None:
        return _with_original_trace(result, original)
    failure = PlannerFailure(code="planner_validation_failed", message="planner validation failed")
    return PlannerValidationResult(failure=failure)


def _with_original_trace(
    result: PlannerValidationResult,
    original: PlannerValidationResult | None,
) -> PlannerValidationResult:
    if original is None or original.failure is None or result.failure is None:
        return result
    trace = dict(result.failure.trace)
    trace["original_validation_errors"] = original.failure.trace.get("validation_errors", [])
    trace["final_validation_errors"] = result.failure.trace.get("validation_errors", [])
    failure = result.failure.model_copy(update={"trace": trace})
    return PlannerValidationResult(failure=failure)
