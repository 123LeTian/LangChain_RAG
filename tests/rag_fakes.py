"""Exact-shape test doubles for C's not-yet-merged frozen contracts."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FakeRAGMode(str, Enum):
    NAIVE = "naive"
    ADVANCED = "advanced"


class FakeTraceStage(str, Enum):
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class FakeRAGSource:
    document_id: str
    chunk_index: int = 0
    source_path: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeRAGChunk:
    chunk_id: str
    content: str
    source: FakeRAGSource
    embedding: Optional[List[float]] = None


@dataclass
class FakeRAGCitation:
    chunk_id: str
    document_id: str
    text_snippet: str = ""


@dataclass
class FakeTraceEvent:
    trace_id: str
    stage: FakeTraceStage
    started_at: datetime
    duration_ms: float
    input_summary: str = ""
    output_summary: str = ""


@dataclass
class FakeRAGResult:
    answer: str
    citations: List[FakeRAGCitation] = field(default_factory=list)
    hits: List[FakeRAGChunk] = field(default_factory=list)
    trace: List[FakeTraceEvent] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class FakeContractFactory:
    @property
    def naive_mode(self):
        return FakeRAGMode.NAIVE

    def stage(self, name):
        return FakeTraceStage[name]

    def trace_event(self, **fields):
        return FakeTraceEvent(**fields)

    def source(self, **fields):
        return FakeRAGSource(**fields)

    def chunk(self, **fields):
        return FakeRAGChunk(**fields)

    def citation(self, **fields):
        return FakeRAGCitation(**fields)

    def result(self, **fields):
        return FakeRAGResult(**fields)


@dataclass
class FakeRequest:
    query: str = "What is RAG?"
    kb_id: Optional[str] = "kb-1"
    mode: Any = FakeRAGMode.NAIVE
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeContext:
    retriever: Any
    llm: Any
    trace_recorder: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {"trace_id": "trace-test"}
    )
