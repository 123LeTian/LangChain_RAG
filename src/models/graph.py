"""Knowledge graph data contracts for Graph Index."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class GraphSourceRef(BaseModel):
    """Traceability pointer from graph data back to a source chunk."""

    document_id: str = Field(..., description="Source document identifier")
    chunk_id: str = Field(..., description="Source chunk identifier")
    filename: Optional[str] = Field(default=None)
    page: Optional[int] = Field(default=None)
    section: Optional[str] = Field(default=None)
    quote: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "chunk_id")
    @classmethod
    def required_ids(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source reference ids must be non-empty strings")
        return value.strip()


class GraphEntity(BaseModel):
    """Entity node in the graph index."""

    id: str
    name: str
    type: str = Field(default="concept")
    description: str = Field(default="")
    source_refs: List[GraphSourceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Directed relationship edge between two graph entities."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str = Field(default="")
    weight: float = Field(default=1.0)
    source_refs: List[GraphSourceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def positive_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("relationship weight must be positive")
        return value


class GraphCommunity(BaseModel):
    """Detected graph community."""

    id: str
    level: int = Field(default=0)
    entity_ids: List[str] = Field(default_factory=list)
    relationship_ids: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommunityReport(BaseModel):
    """Deterministic summary report for a graph community."""

    id: str
    community_id: str
    title: str
    summary: str
    key_entities: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list)
    source_refs: List[GraphSourceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphExtractionResult(BaseModel):
    """Extractor output for one or more chunks."""

    entities: List[GraphEntity] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class GraphBuildResult(BaseModel):
    """Result returned by GraphIndexBuilder."""

    kb_id: str
    entity_count: int
    relationship_count: int
    community_count: int
    report_count: int
    entities: List[GraphEntity] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    communities: List[GraphCommunity] = Field(default_factory=list)
    reports: List[CommunityReport] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CommunityReport",
    "GraphBuildResult",
    "GraphCommunity",
    "GraphEntity",
    "GraphExtractionResult",
    "GraphRelationship",
    "GraphSourceRef",
]
