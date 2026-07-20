"""
RAG Trace — Owner: C
Tracing and observability for RAG pipeline execution.

Provides span-based tracing with:
  - Hierarchical spans (pipeline → retrieve → generate)
  - Event recording with timestamps
  - In-memory and pluggable backends
  - Trace context manager for automatic span lifecycle
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional


# ── Enums ─────────────────────────────────────────────────────────────────


class SpanStatus(str, Enum):
    """Status of a span — follows OpenTelemetry conventions."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TraceEventType(str, Enum):
    """Well-known trace event types for the RAG pipeline."""

    # Pipeline lifecycle
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    PIPELINE_ERROR = "pipeline.error"

    # Retrieval
    RETRIEVAL_START = "retrieval.start"
    RETRIEVAL_END = "retrieval.end"
    RETRIEVAL_ERROR = "retrieval.error"

    # Augmentation
    AUGMENT_START = "augment.start"
    AUGMENT_END = "augment.end"

    # Generation
    GENERATION_START = "generation.start"
    GENERATION_END = "generation.end"
    GENERATION_ERROR = "generation.error"

    # Agent-specific
    AGENT_STEP_START = "agent.step.start"
    AGENT_STEP_END = "agent.step.end"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_ROUTER_DECISION = "agent.router_decision"
    AGENT_FALLBACK = "agent.fallback"

    # Modular pipeline
    MODULE_START = "module.start"
    MODULE_END = "module.end"
    MODULE_ERROR = "module.error"


# ── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class TraceEvent:
    """A single event within a span."""

    event_type: TraceEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class TraceSpan:
    """A span representing a unit of work in the RAG pipeline.

    Spans can be nested: a pipeline span may contain retrieval and generation spans.
    """

    span_id: str = field(default_factory=lambda: f"span_{uuid.uuid4().hex[:16]}")
    parent_id: Optional[str] = None
    name: str = ""
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    events: List[TraceEvent] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Wall-clock duration of the span in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def add_event(
        self,
        event_type: TraceEventType,
        message: str = "",
        **attributes: Any,
    ) -> TraceEvent:
        """Append an event to this span."""
        evt = TraceEvent(
            event_type=event_type,
            attributes=attributes,
            message=message,
        )
        self.events.append(evt)
        return evt

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Mark the span as finished."""
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    def fail(self, error: Exception) -> None:
        """Mark the span as errored."""
        self.end_time = datetime.now(timezone.utc)
        self.status = SpanStatus.ERROR
        self.error_message = f"{type(error).__name__}: {error}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the span to a JSON-compatible dict."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": [
                {
                    "type": e.event_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "message": e.message,
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


# ── Trace ─────────────────────────────────────────────────────────────────


@dataclass
class Trace:
    """A complete trace — the root object for a single RAG query execution.

    A trace contains one or more spans arranged in a tree.
    """

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")
    query_id: str = ""
    spans: List[TraceSpan] = field(default_factory=list)

    @property
    def root_span(self) -> Optional[TraceSpan]:
        """The top-level span (no parent)."""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return None

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total wall-clock duration of the trace."""
        root = self.root_span
        return root.duration_ms if root else None

    @property
    def error_spans(self) -> List[TraceSpan]:
        """Spans that ended in error."""
        return [s for s in self.spans if s.status == SpanStatus.ERROR]

    def add_span(self, span: TraceSpan) -> None:
        """Add a span to this trace."""
        self.spans.append(span)

    def create_child_span(self, name: str, parent_id: str) -> TraceSpan:
        """Create a new span as a child of the given parent."""
        span = TraceSpan(name=name, parent_id=parent_id)
        span.attributes["trace_id"] = self.trace_id
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire trace to a JSON-compatible dict."""
        return {
            "trace_id": self.trace_id,
            "query_id": self.query_id,
            "total_duration_ms": self.total_duration_ms,
            "span_count": len(self.spans),
            "error_count": len(self.error_spans),
            "spans": [s.to_dict() for s in self.spans],
        }


# ── Trace Context Manager ─────────────────────────────────────────────────


class TraceContext:
    """Context manager that automatically manages span lifecycle.

    Usage:
        trace = Trace(query_id=q.query_id)
        with TraceContext(trace, "retrieval") as span:
            span.add_event(TraceEventType.RETRIEVAL_START)
            # ... do work ...
            span.add_event(TraceEventType.RETRIEVAL_END, chunks=str(len(chunks)))
        # span is auto-finished on exit
    """

    def __init__(
        self,
        trace: Trace,
        span_name: str,
        parent_id: Optional[str] = None,
    ) -> None:
        self.trace = trace
        self.span_name = span_name
        self.parent_id = parent_id
        self.span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        self.span = TraceSpan(
            name=self.span_name,
            parent_id=self.parent_id,
        )
        self.span.attributes["trace_id"] = self.trace.trace_id
        self.trace.add_span(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        assert self.span is not None
        if exc_type is not None:
            self.span.fail(exc_val)
        else:
            self.span.finish(SpanStatus.OK)
        return False  # Don't suppress exceptions


# ── Trace Store ───────────────────────────────────────────────────────────


class TraceStore:
    """In-memory trace store with pluggable backend support.

    Default implementation keeps traces in a bounded in-memory dict.
    Callers can inject a custom `backend` (e.g. a DB writer or OTLP exporter)
    via the `save_hook` callback.
    """

    def __init__(
        self,
        max_traces: int = 1000,
        save_hook: Optional[Callable[[Trace], None]] = None,
    ) -> None:
        self._traces: Dict[str, Trace] = {}
        self._max_traces = max_traces
        self._save_hook = save_hook

    def save(self, trace: Trace) -> None:
        """Persist a trace. Evicts oldest if over capacity."""
        if len(self._traces) >= self._max_traces:
            oldest = next(iter(self._traces))
            del self._traces[oldest]
        self._traces[trace.trace_id] = trace
        if self._save_hook:
            self._save_hook(trace)

    def get(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a trace by ID."""
        return self._traces.get(trace_id)

    def list_recent(self, limit: int = 50) -> List[Trace]:
        """Return most recent traces, newest first."""
        traces = list(self._traces.values())
        traces.sort(key=lambda t: t.root_span.start_time if t.root_span else datetime.min, reverse=True)
        return traces[:limit]

    def list_by_query(self, query_id: str) -> List[Trace]:
        """Return all traces for a given query."""
        return [t for t in self._traces.values() if t.query_id == query_id]

    def clear(self) -> None:
        """Remove all traces."""
        self._traces.clear()

    def __len__(self) -> int:
        return len(self._traces)


# ── Global store singleton ────────────────────────────────────────────────

_default_trace_store: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    """Return the global (process-level) trace store, creating it if needed."""
    global _default_trace_store
    if _default_trace_store is None:
        _default_trace_store = TraceStore()
    return _default_trace_store


def set_trace_store(store: TraceStore) -> None:
    """Replace the global trace store (e.g. for testing or custom backends)."""
    global _default_trace_store
    _default_trace_store = store
