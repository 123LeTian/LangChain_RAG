"""
Tests for src/rag/trace.py — Trace, TraceSpan, TraceContext, TraceStore.
"""

import pytest

from src.rag.trace import (
    SpanStatus,
    Trace,
    TraceContext,
    TraceEvent,
    TraceEventType,
    TraceSpan,
    TraceStore,
    get_trace_store,
    set_trace_store,
)


class TestTraceSpan:
    """Tests for TraceSpan."""

    def test_span_creation(self):
        span = TraceSpan(name="test_span")
        assert span.span_id.startswith("span_")
        assert span.name == "test_span"
        assert span.status == SpanStatus.UNSET
        assert span.duration_ms is None  # Not finished yet

    def test_span_add_event(self):
        span = TraceSpan(name="test")
        evt = span.add_event(TraceEventType.RETRIEVAL_START, message="started", chunks=5)
        assert len(span.events) == 1
        assert evt.event_type == TraceEventType.RETRIEVAL_START
        assert evt.attributes["chunks"] == 5

    def test_span_finish(self):
        span = TraceSpan(name="test")
        span.finish(SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_fail(self):
        span = TraceSpan(name="test")
        span.fail(ValueError("boom"))
        assert span.status == SpanStatus.ERROR
        assert "ValueError" in span.error_message
        assert "boom" in span.error_message
        assert span.duration_ms is not None

    def test_span_to_dict(self):
        span = TraceSpan(name="test")
        span.add_event(TraceEventType.PIPELINE_START)
        span.finish()
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "ok"
        assert d["duration_ms"] is not None
        assert len(d["events"]) == 1

    def test_span_parent_child(self):
        parent = TraceSpan(name="parent")
        child = TraceSpan(name="child", parent_id=parent.span_id)
        assert child.parent_id == parent.span_id


class TestTrace:
    """Tests for Trace."""

    def test_trace_creation(self):
        trace = Trace(query_id="q123")
        assert trace.trace_id.startswith("trace_")
        assert trace.query_id == "q123"
        assert len(trace.spans) == 0

    def test_add_span(self):
        trace = Trace(query_id="q")
        span = TraceSpan(name="s")
        trace.add_span(span)
        assert len(trace.spans) == 1

    def test_create_child_span(self):
        trace = Trace(query_id="q")
        parent = TraceSpan(name="parent")
        trace.add_span(parent)
        child = trace.create_child_span("child", parent.span_id)
        assert child.parent_id == parent.span_id
        assert len(trace.spans) == 2

    def test_root_span(self):
        trace = Trace(query_id="q")
        root = TraceSpan(name="root")
        trace.add_span(root)
        trace.add_span(TraceSpan(name="child", parent_id=root.span_id))
        assert trace.root_span is root

    def test_root_span_none_when_empty(self):
        trace = Trace(query_id="q")
        assert trace.root_span is None

    def test_total_duration(self):
        trace = Trace(query_id="q")
        root = TraceSpan(name="root")
        trace.add_span(root)
        root.finish()
        assert trace.total_duration_ms is not None
        assert trace.total_duration_ms >= 0

    def test_error_spans(self):
        trace = Trace(query_id="q")
        s1 = TraceSpan(name="ok")
        s2 = TraceSpan(name="err")
        trace.add_span(s1)
        trace.add_span(s2)
        s1.finish()
        s2.fail(ValueError("x"))
        assert len(trace.error_spans) == 1
        assert trace.error_spans[0].name == "err"

    def test_to_dict(self):
        trace = Trace(query_id="q")
        root = TraceSpan(name="root")
        trace.add_span(root)
        root.finish()
        d = trace.to_dict()
        assert d["trace_id"] == trace.trace_id
        assert d["query_id"] == "q"
        assert d["span_count"] == 1
        assert d["error_count"] == 0


class TestTraceContext:
    """Tests for TraceContext manager."""

    def test_context_success(self):
        trace = Trace(query_id="q")
        with TraceContext(trace, "test_op") as span:
            span.add_event(TraceEventType.RETRIEVAL_START)
        assert span.status == SpanStatus.OK
        assert len(trace.spans) == 1

    def test_context_error(self):
        trace = Trace(query_id="q")
        with pytest.raises(ValueError):
            with TraceContext(trace, "test_op") as span:
                raise ValueError("boom")
        assert span.status == SpanStatus.ERROR
        assert "ValueError" in span.error_message


class TestTraceStore:
    """Tests for TraceStore."""

    def setup_method(self):
        self.store = TraceStore(max_traces=10)

    def test_save_and_get(self):
        trace = Trace(query_id="q1")
        self.store.save(trace)
        retrieved = self.store.get(trace.trace_id)
        assert retrieved is trace

    def test_get_missing(self):
        assert self.store.get("nonexistent") is None

    def test_list_recent(self):
        for i in range(5):
            t = Trace(query_id=f"q{i}")
            self.store.save(t)
        recent = self.store.list_recent(limit=3)
        assert len(recent) == 3

    def test_list_by_query(self):
        t1 = Trace(query_id="qa")
        t2 = Trace(query_id="qb")
        t3 = Trace(query_id="qa")
        self.store.save(t1)
        self.store.save(t2)
        self.store.save(t3)
        qa_traces = self.store.list_by_query("qa")
        assert len(qa_traces) == 2

    def test_eviction(self):
        store = TraceStore(max_traces=3)
        for i in range(5):
            store.save(Trace(query_id=f"q{i}"))
        assert len(store) <= 3

    def test_clear(self):
        self.store.save(Trace(query_id="q"))
        self.store.clear()
        assert len(self.store) == 0

    def test_save_hook(self):
        saved = []

        def hook(trace):
            saved.append(trace.trace_id)

        store = TraceStore(max_traces=10, save_hook=hook)
        t = Trace(query_id="q")
        store.save(t)
        assert len(saved) == 1
        assert saved[0] == t.trace_id


class TestGlobalTraceStore:
    """Tests for the global trace store singleton."""

    def teardown_method(self):
        set_trace_store(TraceStore())

    def test_get_default(self):
        store = get_trace_store()
        assert isinstance(store, TraceStore)

    def test_set_custom(self):
        custom = TraceStore(max_traces=5)
        set_trace_store(custom)
        assert get_trace_store() is custom
