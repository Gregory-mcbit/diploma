from __future__ import annotations

from datetime import datetime, timezone
import uuid

from app.domain.schemas import DecisionLogEntry, FreshnessStatus, ProvenanceRecord, TraceEvent
from app.graph.state import GraphState
from app.observability.tracing import get_correlation_id, get_request_id


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_provenance_record(
    *,
    source: str,
    staleness_status: FreshnessStatus,
    confidence: float,
    details: dict | None = None,
    retrieval_id: str | None = None,
    timestamp: str | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        source=source,
        timestamp=timestamp or utc_now_iso(),
        staleness_status=staleness_status,
        confidence=confidence,
        retrieval_id=retrieval_id or str(uuid.uuid4()),
        details=details or {},
    )


def append_decision_log(
    state: GraphState,
    *,
    stage: str,
    event: str,
    message: str,
    tool_calls: list[str] | None = None,
    rule_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> list[DecisionLogEntry]:
    history = list(state.get("decision_log", []))
    request_id = state.get("request_id") or get_request_id()
    correlation_id = state.get("correlation_id") or get_correlation_id()
    history.append(
        DecisionLogEntry(
            stage=stage,
            event=event,
            message=message,
            timestamp=utc_now_iso(),
            tool_calls=tool_calls or [],
            rule_ids=rule_ids or [],
            metadata={
                **(metadata or {}),
                "request_id": request_id,
                "correlation_id": correlation_id,
            },
        )
    )
    return history


def append_trace_event(
    state: GraphState,
    *,
    stage: str,
    event: str,
    message: str,
    metadata: dict | None = None,
) -> list[TraceEvent]:
    request_id = state.get("request_id") or get_request_id()
    correlation_id = state.get("correlation_id") or get_correlation_id()
    if not request_id:
        raise RuntimeError("Trace event emission requires request_id in state or tracing context.")
    if not correlation_id:
        raise RuntimeError("Trace event emission requires correlation_id in state or tracing context.")

    history = list(state.get("trace_log", []))
    history.append(
        TraceEvent(
            request_id=request_id,
            correlation_id=correlation_id,
            stage=stage,
            event=event,
            message=message,
            timestamp=utc_now_iso(),
            metadata=metadata or {},
        )
    )
    return history
