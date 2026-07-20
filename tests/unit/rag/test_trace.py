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
    TraceRecorder,
    TraceSpan,
    TraceStore,
    get_recorder,
    get_trace_store,
    set_recorder,
    set_trace_store,
)
from src.models.rag import TraceStage


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


# ═══════════════════════════════════════════════════════════════════════════
# TraceRecorder Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTraceRecorderCreation:
    """1. Trace can be created."""

    def test_create_recorder(self):
        rec = TraceRecorder()
        assert rec.trace_count == 0
        assert rec.list_trace_ids() == []

    def test_start_trace_initialises(self):
        rec = TraceRecorder()
        rec.start_trace("t1")
        assert rec.get_trace("t1") == []

    def test_start_trace_idempotent(self):
        rec = TraceRecorder()
        rec.start_trace("t1")
        rec.start_trace("t1")  # Should not raise or overwrite


class TestTraceRecorderStartEnd:
    """Core start/end recording."""

    def test_start_end_single_stage(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE, "query=test")
        event = rec.end("t1", TraceStage.RETRIEVE, "5 chunks")

        assert event.trace_id == "t1"
        assert event.stage == TraceStage.RETRIEVE
        assert event.input_summary == "query=test"
        assert event.output_summary == "5 chunks"
        assert event.duration_ms >= 0

    def test_end_without_start_raises(self):
        rec = TraceRecorder()
        with pytest.raises(ValueError, match="was not started"):
            rec.end("t1", TraceStage.GENERATE)

    def test_double_start_raises(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE)
        with pytest.raises(ValueError, match="already active"):
            rec.start("t1", TraceStage.RETRIEVE)


class TestTraceRecorderEventOrder:
    """2. Trace events are in correct order."""

    def test_events_ordered_by_insertion(self):
        rec = TraceRecorder()
        stages = [
            TraceStage.INTENT,
            TraceStage.RETRIEVE,
            TraceStage.RERANK,
            TraceStage.GENERATE,
            TraceStage.COMPLETE,
        ]
        for s in stages:
            rec.start("t1", s, f"in:{s.value}")
            rec.end("t1", s, f"out:{s.value}")

        events = rec.get_trace("t1")
        assert len(events) == 5
        for i, s in enumerate(stages):
            assert events[i].stage == s

    def test_multiple_traces_independent(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE)
        rec.end("t1", TraceStage.RETRIEVE, "5 chunks")

        rec.start("t2", TraceStage.GENERATE)
        rec.end("t2", TraceStage.GENERATE, "answer")

        assert len(rec.get_trace("t1")) == 1
        assert len(rec.get_trace("t2")) == 1
        assert rec.get_trace("t1")[0].stage == TraceStage.RETRIEVE
        assert rec.get_trace("t2")[0].stage == TraceStage.GENERATE

    def test_same_stage_different_traces_ok(self):
        """Same stage can be active in different traces simultaneously."""
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE)
        rec.start("t2", TraceStage.RETRIEVE)  # Should not raise
        rec.end("t1", TraceStage.RETRIEVE)
        rec.end("t2", TraceStage.RETRIEVE)
        assert len(rec.get_trace("t1")) == 1
        assert len(rec.get_trace("t2")) == 1


class TestTraceRecorderTiming:
    """3. Time statistics are correct."""

    def test_duration_ms_positive(self):
        rec = TraceRecorder()
        import time
        rec.start("t1", TraceStage.GENERATE, "in")
        time.sleep(0.05)
        evt = rec.end("t1", TraceStage.GENERATE, "out")
        assert evt.duration_ms >= 40  # at least 40ms

    def test_duration_is_monotonic(self):
        """Later stages should not have zero or negative time relative to earlier."""
        rec = TraceRecorder()
        import time

        rec.start("t1", TraceStage.RETRIEVE)
        time.sleep(0.01)
        evt1 = rec.end("t1", TraceStage.RETRIEVE)

        rec.start("t1", TraceStage.GENERATE)
        time.sleep(0.02)
        evt2 = rec.end("t1", TraceStage.GENERATE)

        # Both have positive duration
        assert evt1.duration_ms > 0
        assert evt2.duration_ms > 0


class TestTraceRecorderRecord:
    """Direct event recording via record()."""

    def test_record_prebuilt_event(self):
        from src.models.rag import TraceEvent
        from datetime import datetime, timezone

        rec = TraceRecorder()
        evt = TraceEvent(
            trace_id="t1",
            stage=TraceStage.ERROR,
            started_at=datetime.now(timezone.utc),
            duration_ms=100.0,
            input_summary="query=test",
            output_summary="error=timeout",
        )
        rec.record("t1", evt)
        assert len(rec.get_trace("t1")) == 1
        assert rec.get_trace("t1")[0].stage == TraceStage.ERROR


class TestTraceRecorderManagement:
    """Lifecycle management: end_trace, clear, clear_trace."""

    def test_end_trace_auto_closes_spans(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE, "in")
        # Forget to call end() — end_trace should close it
        rec.end_trace("t1")

        events = rec.get_trace("t1")
        assert len(events) == 1
        assert "[auto-closed]" in events[0].output_summary

    def test_clear_removes_all(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE)
        rec.end("t1", TraceStage.RETRIEVE)
        rec.clear()
        assert rec.trace_count == 0
        assert rec.get_trace("t1") == []

    def test_clear_trace_single(self):
        rec = TraceRecorder()
        rec.start("t1", TraceStage.RETRIEVE)
        rec.end("t1", TraceStage.RETRIEVE)
        rec.start("t2", TraceStage.GENERATE)
        rec.end("t2", TraceStage.GENERATE)

        rec.clear_trace("t1")
        assert rec.get_trace("t1") == []
        assert len(rec.get_trace("t2")) == 1

    def test_list_trace_ids(self):
        rec = TraceRecorder()
        assert rec.list_trace_ids() == []
        rec.start("t_a", TraceStage.RETRIEVE)
        rec.end("t_a", TraceStage.RETRIEVE)
        rec.start("t_b", TraceStage.GENERATE)
        rec.end("t_b", TraceStage.GENERATE)
        ids = rec.list_trace_ids()
        assert "t_a" in ids
        assert "t_b" in ids
        assert len(ids) == 2


class TestTraceRecorderSubscribers:
    """SSE real-time subscriber support."""

    def test_subscriber_called_on_end(self):
        rec = TraceRecorder()
        received = []

        def cb(event):
            received.append(event)

        rec.subscribe(cb)
        rec.start("t1", TraceStage.RETRIEVE)
        rec.end("t1", TraceStage.RETRIEVE, "done")

        assert len(received) == 1
        assert received[0].stage == TraceStage.RETRIEVE

    def test_subscriber_called_on_record(self):
        rec = TraceRecorder()
        received = []

        def cb(event):
            received.append(event)

        rec.subscribe(cb)
        from src.models.rag import TraceEvent
        from datetime import datetime, timezone
        evt = TraceEvent(
            trace_id="t1", stage=TraceStage.COMPLETE,
            started_at=datetime.now(timezone.utc),
        )
        rec.record("t1", evt)
        assert len(received) == 1

    def test_unsubscribe_stops_calls(self):
        rec = TraceRecorder()
        received = []

        def cb(event):
            received.append(event)

        rec.subscribe(cb)
        rec.unsubscribe(cb)
        rec.start("t1", TraceStage.RETRIEVE)
        rec.end("t1", TraceStage.RETRIEVE)
        assert len(received) == 0

    def test_subscriber_exception_does_not_break_pipeline(self):
        rec = TraceRecorder()

        def bad_cb(event):
            raise RuntimeError("subscriber boom")

        rec.subscribe(bad_cb)
        # Should not raise
        rec.start("t1", TraceStage.RETRIEVE)
        event = rec.end("t1", TraceStage.RETRIEVE, "ok")
        assert event is not None


class TestGlobalRecorder:
    """Global TraceRecorder singleton."""

    def teardown_method(self):
        set_recorder(TraceRecorder())

    def test_get_default(self):
        r = get_recorder()
        assert isinstance(r, TraceRecorder)

    def test_set_custom(self):
        custom = TraceRecorder()
        set_recorder(custom)
        assert get_recorder() is custom
