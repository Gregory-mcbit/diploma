from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator
import uuid

from pydantic import BaseModel


_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
_CORRELATION_ID: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def generate_request_id() -> str:
    return str(uuid.uuid4())


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def set_trace_context(
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> tuple[str, str, Token, Token]:
    resolved_request_id = request_id or generate_request_id()
    resolved_correlation_id = correlation_id or generate_correlation_id()
    request_token = _REQUEST_ID.set(resolved_request_id)
    correlation_token = _CORRELATION_ID.set(resolved_correlation_id)
    return resolved_request_id, resolved_correlation_id, request_token, correlation_token


def reset_trace_context(request_token: Token, correlation_token: Token) -> None:
    _REQUEST_ID.reset(request_token)
    _CORRELATION_ID.reset(correlation_token)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def get_correlation_id() -> str | None:
    return _CORRELATION_ID.get()


@contextmanager
def request_trace_context(
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> Iterator[tuple[str, str]]:
    resolved_request_id, resolved_correlation_id, request_token, correlation_token = set_trace_context(
        request_id=request_id,
        correlation_id=correlation_id,
    )
    try:
        yield resolved_request_id, resolved_correlation_id
    finally:
        reset_trace_context(request_token, correlation_token)


def to_serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    return value
