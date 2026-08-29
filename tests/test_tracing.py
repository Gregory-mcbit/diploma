from pathlib import Path

from app.domain.schemas import ExecutionTraceRecord
from app.graph.pipeline import run_investment_graph
from app.observability.trace_store import load_execution_traces, persist_execution_trace
from app.graph.telemetry import utc_now_iso


def test_trace_store_persists_and_loads(monkeypatch, tmp_path):
    trace_dir = tmp_path / "trace_store"
    trace_file = trace_dir / "executions.jsonl"
    monkeypatch.setattr("app.observability.trace_store.TRACE_STORE_DIR", trace_dir)
    monkeypatch.setattr("app.observability.trace_store.TRACE_STORE_FILE", trace_file)

    record = ExecutionTraceRecord(
        request_id="req-trace",
        correlation_id="corr-trace",
        run_type="investment",
        status="completed",
        started_at=utc_now_iso(),
        completed_at=utc_now_iso(),
    )

    persist_execution_trace(record)
    loaded = load_execution_traces()

    assert trace_file.exists()
    assert len(loaded) == 1
    assert loaded[0]["request_id"] == "req-trace"
    assert loaded[0]["correlation_id"] == "corr-trace"


def test_run_investment_graph_persists_execution_trace(monkeypatch, tmp_path):
    trace_dir = tmp_path / "trace_store"
    trace_file = trace_dir / "executions.jsonl"
    monkeypatch.setattr("app.observability.trace_store.TRACE_STORE_DIR", trace_dir)
    monkeypatch.setattr("app.observability.trace_store.TRACE_STORE_FILE", trace_file)

    class FakeCompiledGraph:
        def invoke(self, initial_state):
            return initial_state

    monkeypatch.setattr("app.graph.pipeline.build_investment_graph", lambda: FakeCompiledGraph())

    result = run_investment_graph(
        "Build a moderate portfolio.",
        request_id="req-trace-graph",
        correlation_id="corr-trace-graph",
    )
    loaded = load_execution_traces()

    assert result["request_id"] == "req-trace-graph"
    assert result["correlation_id"] == "corr-trace-graph"
    assert len(result["trace_log"]) >= 2
    assert trace_file.exists()
    assert loaded[-1]["request_id"] == "req-trace-graph"
    assert loaded[-1]["correlation_id"] == "corr-trace-graph"
    assert loaded[-1]["run_type"] == "investment"
    assert loaded[-1]["status"] == "completed"
