from app.graph.state import GraphState


def build_initial_state(
    user_query: str,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> GraphState:
    from app.graph.pipeline import build_initial_state as _build_initial_state

    return _build_initial_state(user_query, request_id=request_id, correlation_id=correlation_id)


def build_investment_graph():
    from app.graph.pipeline import build_investment_graph as _build_investment_graph

    return _build_investment_graph()


def build_monitoring_graph():
    from app.graph.pipeline import build_monitoring_graph as _build_monitoring_graph

    return _build_monitoring_graph()


def run_investment_graph(
    user_query: str,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> GraphState:
    from app.graph.pipeline import run_investment_graph as _run_investment_graph

    return _run_investment_graph(user_query, request_id=request_id, correlation_id=correlation_id)


def run_monitoring_graph(initial_state: GraphState) -> GraphState:
    from app.graph.pipeline import run_monitoring_graph as _run_monitoring_graph

    return _run_monitoring_graph(initial_state)


__all__ = [
    "build_initial_state",
    "build_investment_graph",
    "build_monitoring_graph",
    "run_investment_graph",
    "run_monitoring_graph",
]
