from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.schemas import ExecutionTraceRecord
from app.observability.logger import get_logger
from app.observability.tracing import to_serializable


TRACE_STORE_DIR = Path("data/trace_store")
TRACE_STORE_FILE = TRACE_STORE_DIR / "executions.jsonl"

logger = get_logger(__name__)


def persist_execution_trace(record: ExecutionTraceRecord) -> Path:
    TRACE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_STORE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(record), ensure_ascii=True) + "\n")
    logger.info(
        "Persisted execution trace run_type=%s status=%s to %s.",
        record.run_type,
        record.status,
        TRACE_STORE_FILE,
    )
    return TRACE_STORE_FILE


def load_execution_traces() -> list[dict[str, Any]]:
    if not TRACE_STORE_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    with TRACE_STORE_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
