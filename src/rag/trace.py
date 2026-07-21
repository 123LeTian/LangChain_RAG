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


# ═══════════════════════════════════════════════════════════════════════════
# TraceRecorder — Stage-level pipeline tracing (Pydantic TraceEvent)
# ═══════════════════════════════════════════════════════════════════════════


class TraceRecorder:
    """Records RAG pipeline execution as ordered stage-level :class:`TraceEvent` s.

    Each event captures:
      - ``trace_id``    — shared identifier for correlation
      - ``stage``        — :class:`TraceStage` enum value
      - ``started_at``   — UTC timestamp when the stage began
      - ``duration_ms``  — wall-clock duration computed from ``start()`` / ``end()``
      - ``input_summary``  — short human-readable input description
      - ``output_summary`` — short human-readable output description

    Designed for three use-cases:
      1. **SSE real-time output** — subscribe a callback; fired on every ``end()``.
      2. **Frontend trace timeline** — ``get_trace()`` returns ordered events.
      3. **Evaluation** — full event history for metric computation.

    Usage::

        recorder = TraceRecorder()

        # Start a trace
        recorder.start_trace("trace_001")

        # Record stages
        recorder.start("trace_001", TraceStage.RETRIEVE, "query=What is RAG?")
        # ... do work ...
        event = recorder.end("trace_001", TraceStage.RETRIEVE, "retrieved 5 chunks")

        recorder.start("trace_001", TraceStage.GENERATE, "context=5 chunks")
        # ... do work ...
        event = recorder.end("trace_001", TraceStage.GENERATE, "answer=42 chars")

        # Retrieve trace
        events = recorder.get_trace("trace_001")
        for evt in events:
            print(f"{evt.stage.value}: {evt.duration_ms:.1f}ms")
    """

    def __init__(self) -> None:
        # trace_id → ordered list of completed TraceEvent
        self._traces: Dict[str, List[Any]] = {}
        # (trace_id, stage) → (started_at_float, input_summary)
        self._active_spans: Dict[tuple, tuple[float, str]] = {}
        # Callbacks for real-time event streaming (SSE)
        self._subscribers: List[Callable[[Any], None]] = []

    # ── Trace lifecycle ──────────────────────────────────────────────────

    def start_trace(self, trace_id: str) -> None:
        """Initialise storage for a new trace.

        Args:
            trace_id: Unique identifier shared by all events in this request.
        """
        if trace_id not in self._traces:
            self._traces[trace_id] = []

    def end_trace(self, trace_id: str) -> None:
        """Mark a trace as complete.  No more events will be added.

        Any unclosed spans are closed automatically with a warning summary.
        """
        # Auto-close any unclosed spans
        for (tid, stage), (started, input_sum) in list(self._active_spans.items()):
            if tid == trace_id:
                self.end(trace_id, stage, f"[auto-closed] started with: {input_sum}")

    # ── Span lifecycle ───────────────────────────────────────────────────

    def start(
        self,
        trace_id: str,
        stage: Any,             # TraceStage
        input_summary: str = "",
    ) -> None:
        """Record the start of a pipeline stage.

        Args:
            trace_id:  Trace identifier.
            stage:     :class:`TraceStage` enum member.
            input_summary: Short description of what entered this stage.

        Raises:
            ValueError: If the same stage is already active for this trace_id.
        """
        from src.models.rag import TraceStage as _TS

        key = (trace_id, stage)
        if key in self._active_spans:
            raise ValueError(
                f"Stage '{stage.value}' is already active for trace '{trace_id}'. "
                f"Call end() before starting it again."
            )
        self.start_trace(trace_id)
        self._active_spans[key] = (time.perf_counter(), input_summary)

    def end(
        self,
        trace_id: str,
        stage: Any,             # TraceStage
        output_summary: str = "",
    ) -> Any:                   # TraceEvent (Pydantic)
        """Record the end of a pipeline stage and return the completed event.

        Computes ``duration_ms`` from the corresponding ``start()`` call.

        Args:
            trace_id:  Trace identifier.
            stage:     :class:`TraceStage` enum member.
            output_summary: Short description of what this stage produced.

        Returns:
            A completed :class:`TraceEvent` (Pydantic model from ``src.models.rag``).

        Raises:
            ValueError: If the stage was not started.
        """
        from src.models.rag import TraceEvent, TraceStage as _TS

        key = (trace_id, stage)
        if key not in self._active_spans:
            raise ValueError(
                f"Stage '{stage.value}' was not started for trace '{trace_id}'. "
                f"Call start() first."
            )

        started_float, input_summary = self._active_spans.pop(key)
        duration_ms = (time.perf_counter() - started_float) * 1000.0

        event = TraceEvent(
            trace_id=trace_id,
            stage=stage,
            started_at=datetime.fromtimestamp(started_float, tz=timezone.utc),
            duration_ms=round(duration_ms, 3),
            input_summary=input_summary,
            output_summary=output_summary,
        )
        self._traces.setdefault(trace_id, []).append(event)
        self._notify(event)
        return event

    # ── Direct event recording ───────────────────────────────────────────

    def record(self, trace_id: str, event: Any) -> None:
        """Insert a pre-built :class:`TraceEvent` directly.

        Use this when you already have a fully-formed event (e.g. from an
        external source or reconstructed trace).

        Args:
            trace_id: Trace identifier.
            event:    A :class:`TraceEvent` Pydantic instance.
        """
        self.start_trace(trace_id)
        self._traces[trace_id].append(event)
        self._notify(event)

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> List[Any]:
        """Return all events for *trace_id* in insertion order.

        Args:
            trace_id: Trace identifier.

        Returns:
            Ordered list of :class:`TraceEvent` objects (empty if unknown trace_id).
        """
        return list(self._traces.get(trace_id, []))

    def list_trace_ids(self) -> List[str]:
        """Return all trace identifiers that have at least one event."""
        return [tid for tid, events in self._traces.items() if events]

    @property
    def trace_count(self) -> int:
        """Number of traces with at least one recorded event."""
        return sum(1 for events in self._traces.values() if events)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all traces and active spans."""
        self._traces.clear()
        self._active_spans.clear()

    def clear_trace(self, trace_id: str) -> None:
        """Remove a single trace and its active spans."""
        self._traces.pop(trace_id, None)
        for (tid, stage) in list(self._active_spans):
            if tid == trace_id:
                del self._active_spans[(tid, stage)]

    # ── Real-time streaming (SSE) ────────────────────────────────────────

    def subscribe(self, callback: Callable[[Any], None]) -> None:
        """Register a callback invoked on every completed event.

        Use for SSE, WebSocket, or log-based real-time output.

        Args:
            callback: ``Callable[[TraceEvent], None]`` — receives each event.
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Any], None]) -> None:
        """Remove a previously registered subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify(self, event: Any) -> None:
        """Push *event* to all subscribers (best-effort)."""
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass  # Subscriber failure must not break pipeline


# ── Global singleton ─────────────────────────────────────────────────────

_default_recorder: Optional[TraceRecorder] = None


def get_recorder() -> TraceRecorder:
    """Return the global (process-level) TraceRecorder singleton."""
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = TraceRecorder()
    return _default_recorder


def set_recorder(recorder: TraceRecorder) -> None:
    """Replace the global TraceRecorder (e.g. for testing)."""
    global _default_recorder
    _default_recorder = recorder
