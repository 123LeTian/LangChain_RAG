"""
Graph Routes — knowledge graph visualization and graph model selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from src.api.document_store import apply_document_metadata
from src.api.dependencies import GraphRepoDep, ModelRegistryDep
from src.api.errors import NotFoundError, ValidationError
from src.api.graph_model_extractor import LLMGraphExtractor
from src.chat.llm_client_factory import LLMClientFactory
from src.chat.model_registry import ChatModel, ModelNotFoundError
from src.chat.model_runtime import ModelRuntimeError
from src.graph.builder import GraphIndexBuilder
from src.models.knowledge import ChunkRecord

router = APIRouter(prefix="/api/graphs", tags=["Graph"])


class GraphModelUpdate(BaseModel):
    model_id: str = Field(..., min_length=1)

    @field_validator("model_id")
    @classmethod
    def model_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model_id cannot be empty")
        return stripped


def _knowledge_module():
    from src.api.routes import knowledge as knowledge_routes

    return knowledge_routes


def _find_kb(kb_id: str) -> dict[str, Any]:
    knowledge = _knowledge_module()
    for kb in knowledge._MOCK_KBS:
        if kb.get("id") == kb_id:
            return kb
    raise NotFoundError(f"KB {kb_id} not found")


def _persist_knowledge() -> None:
    _knowledge_module()._persist()


def _safe_model_metadata(model: ChatModel) -> dict[str, Any]:
    return {
        "id": model.id,
        "provider": model.provider,
        "display_name": model.display_name,
        "model_name": model.model_name,
        "base_url": model.base_url,
        "supports_stream": model.supports_stream,
        "supports_tools": model.supports_tools,
        "supports_vision": model.supports_vision,
        "key_required": model.public_dict().get("key_required", False),
        "key_configured": model.public_dict().get("key_configured", False),
    }


def _resolve_graph_model(
    kb: dict[str, Any],
    registry: Any,
    model_id: Optional[str],
) -> ChatModel:
    selected = model_id or kb.get("graph_model_id")
    try:
        model = registry.get_enabled(selected) if selected else registry.default_model()
    except ModelNotFoundError:
        raise ValidationError(f"Model {selected} is not available")
    kb["graph_model_id"] = model.id
    _persist_knowledge()
    return model


def _load_graph_chunks(kb_id: str) -> list[ChunkRecord]:
    knowledge = _knowledge_module()
    docs = [doc for doc in knowledge._MOCK_DOCS if doc.get("kb_id") == kb_id]
    if not docs:
        return []

    from src.ingestion import LoaderFactory
    from src.ingestion.splitter import split_documents

    loaded_docs = []
    docs_dir = knowledge._documents_root()
    for doc in docs:
        path = knowledge._document_path(doc)
        if not path.exists():
            continue
        loaded = LoaderFactory.load(
            path,
            kb_id=kb_id,
            document_id=str(doc.get("id") or path.stem),
        )
        loaded_docs.append(
            apply_document_metadata(loaded, doc, path, docs_dir=docs_dir)
        )

    chunks = []
    for item in split_documents(loaded_docs, chunk_size=800, chunk_overlap=120):
        metadata = dict(item.metadata or {})
        chunks.append(
            ChunkRecord(
                id=str(metadata.get("chunk_id") or f"{metadata.get('document_id', 'doc')}:chunk-{len(chunks)}"),
                document_id=str(metadata.get("document_id") or metadata.get("source") or "unknown"),
                kb_id=str(metadata.get("kb_id") or kb_id),
                text=item.page_content,
                index=int(metadata.get("chunk_index") or len(chunks)),
                metadata=metadata,
            )
        )
    return chunks


def _graph_result_to_payload(result: Any, model: ChatModel) -> dict[str, Any]:
    entity_names = {entity.id: entity.name for entity in result.entities}
    nodes = []
    for entity in result.entities:
        ref = entity.source_refs[0] if entity.source_refs else None
        nodes.append(
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type or "entity",
                "chunk_id": ref.chunk_id if ref else "",
                "label": entity.type or "Entity",
                "source_text": ref.quote if ref else entity.description,
            }
        )

    links = []
    for relationship in result.relationships:
        ref = relationship.source_refs[0] if relationship.source_refs else None
        links.append(
            {
                "source": relationship.source_entity_id,
                "target": relationship.target_entity_id,
                "relation": relationship.relation_type,
                "label": relationship.relation_type,
                "chunk_id": ref.chunk_id if ref else "",
                "source_text": ref.quote if ref else relationship.description,
            }
        )

    communities = []
    reports_by_community = {report.community_id: report for report in result.reports}
    for community in result.communities:
        report = reports_by_community.get(community.id)
        communities.append(
            {
                "id": community.id,
                "name": report.title if report else f"社区 {community.id}",
                "summary": report.summary if report else community.summary,
                "entity_count": len(community.entity_ids),
                "relation_count": len(community.relationship_ids),
            }
        )

    return {
        "nodes": nodes,
        "links": links,
        "communities": communities,
        "hit_path": [],
        "model": _safe_model_metadata(model),
        "build": {
            "mode": str((result.metadata or {}).get("extractor") or "rule_based"),
            "model_id": model.id,
            "entity_count": result.entity_count,
            "relationship_count": result.relationship_count,
            "community_count": result.community_count,
            "report_count": result.report_count,
            "duration_ms": result.duration_ms,
            "warnings": result.warnings,
        },
        "metadata": {
            "entity_names": entity_names,
            **(result.metadata or {}),
        },
    }


def _empty_graph_payload(kb_id: str, model: ChatModel, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "nodes": [],
        "links": [],
        "communities": [],
        "hit_path": [],
        "model": _safe_model_metadata(model),
        "build": {
            "mode": "rule_based",
            "model_id": model.id,
            "entity_count": 0,
            "relationship_count": 0,
            "community_count": 0,
            "report_count": 0,
            "duration_ms": 0,
            "warnings": warnings or [f"knowledge graph for kb_id '{kb_id}' is empty"],
        },
        "metadata": {},
    }


@router.patch("/{kb_id}/model")
async def update_graph_model(
    kb_id: str,
    request: GraphModelUpdate,
    registry: ModelRegistryDep = None,  # type: ignore
) -> dict:
    kb = _find_kb(kb_id)
    try:
        model = registry.get_enabled(request.model_id)
    except ModelNotFoundError:
        raise ValidationError(f"Model {request.model_id} is not available")
    kb["graph_model_id"] = model.id
    _persist_knowledge()
    return {"kb_id": kb_id, "model": _safe_model_metadata(model)}


@router.get("/{kb_id}")
async def get_graph(
    kb_id: str,
    model_id: Optional[str] = None,
    graph_repo: GraphRepoDep = None,  # type: ignore
    registry: ModelRegistryDep = None,  # type: ignore
) -> dict:
    """Return graph data for a knowledge base and the selected graph model."""
    kb = _find_kb(kb_id)
    model = _resolve_graph_model(kb, registry, model_id)

    try:
        chunks = _load_graph_chunks(kb_id)
        if not chunks:
            return _empty_graph_payload(kb_id, model)
        uses_llm_extractor = model.provider.lower() != "mock"
        builder = _create_graph_builder(model)
        result = builder.build_from_chunks(kb_id=kb_id, chunks=chunks)
        result.metadata["extractor"] = "llm" if uses_llm_extractor else "rule_based"
        if not result.entities:
            return _empty_graph_payload(kb_id, model, result.warnings)
        return _graph_result_to_payload(result, model)
    except ModelRuntimeError as exc:
        raise ValidationError(str(exc))
    except Exception as exc:
        raise ValidationError(f"Graph build failed: {type(exc).__name__}: {exc}")


def _create_graph_builder(model: ChatModel) -> GraphIndexBuilder:
    if model.provider.lower() == "mock":
        return GraphIndexBuilder()
    llm = LLMClientFactory().create_chat_model(
        model,
        temperature=0.0,
        max_tokens=1800,
        streaming=False,
    )
    return GraphIndexBuilder(extractor=LLMGraphExtractor(llm, model_id=model.id))
