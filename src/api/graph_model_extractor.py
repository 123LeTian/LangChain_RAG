"""LLM-backed graph extraction adapter for the graph API layer."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from src.graph.extractor import (
    GraphExtractor,
    RuleBasedGraphExtractor,
    stable_entity_id,
    stable_relationship_id,
)
from src.models.graph import (
    GraphEntity,
    GraphExtractionResult,
    GraphRelationship,
    GraphSourceRef,
)


class LLMGraphExtractor(GraphExtractor):
    """Extract graph entities and relationships through a selected chat model."""

    def __init__(self, llm: Any, *, model_id: str, max_chars: int = 7000) -> None:
        self.llm = llm
        self.model_id = model_id
        self.max_chars = max_chars
        self.fallback = RuleBasedGraphExtractor()

    def extract_from_chunk(self, chunk: Any) -> GraphExtractionResult:
        return self.extract_from_chunks([chunk])

    def extract_from_chunks(self, chunks: Iterable[Any]) -> GraphExtractionResult:
        chunk_list = list(chunks)
        if not chunk_list:
            return GraphExtractionResult(warnings=["no chunks provided"])

        prompt = self._build_prompt(chunk_list)
        try:
            response = self.llm.invoke(prompt)
            payload = _parse_json_payload(getattr(response, "content", response))
            result = self._payload_to_result(payload, chunk_list)
            if result.entities:
                return result
            fallback = self.fallback.extract_from_chunks(chunk_list)
            fallback.warnings.append(
                f"model {self.model_id} returned no graph entities; used rule fallback"
            )
            return fallback
        except Exception as exc:
            fallback = self.fallback.extract_from_chunks(chunk_list)
            fallback.warnings.append(
                f"model {self.model_id} graph extraction failed: {type(exc).__name__}"
            )
            return fallback

    def _build_prompt(self, chunks: list[Any]) -> str:
        sections = []
        remaining = self.max_chars
        for chunk in chunks:
            data = _chunk_data(chunk)
            text = data["text"][:remaining]
            remaining -= len(text)
            sections.append(
                f"[chunk_id={data['chunk_id']} document_id={data['document_id']}]\n{text}"
            )
            if remaining <= 0:
                break

        return (
            "You are a knowledge graph extraction engine.\n"
            "Extract important entities and directed relationships from the source text.\n"
            "Return ONLY valid JSON with this exact shape:\n"
            "{\n"
            '  "entities": [{"name": "Entity Name", "type": "person|project|metric"}],\n'
            '  "relationships": [{"source": "Entity Name", "target": "Entity Name", "relation": "related_to"}]\n'
            "}\n"
            "Rules:\n"
            "- Use concise entity names.\n"
            "- Entity type must be one of: person, project, metric.\n"
            "- Use person for human names, project for projects/organizations/products/systems, metric for dates, money, numbers, amounts, and abstract concepts.\n"
            "- Prefer core concepts, people, products, organizations, metrics, dates, and modules.\n"
            "- Merge aliases and near-duplicates into one concise entity name.\n"
            "- Keep at most 36 entities and 56 relationships.\n"
            "- Every relationship source and target must appear in entities.\n"
            "- Do not include API keys, prompts, or hidden configuration.\n"
            "- If unsure, omit the item.\n\n"
            "Source text:\n"
            + "\n\n".join(sections)
        )

    def _payload_to_result(self, payload: Mapping[str, Any], chunks: list[Any]) -> GraphExtractionResult:
        first_chunk = _chunk_data(chunks[0])
        source_ref = GraphSourceRef(
            document_id=first_chunk["document_id"],
            chunk_id=first_chunk["chunk_id"],
            filename=_optional_str(first_chunk["metadata"].get("filename") or first_chunk["metadata"].get("source")),
            quote=first_chunk["text"][:500],
            metadata={**first_chunk["metadata"], "extractor_model_id": self.model_id},
        )
        kb_id = first_chunk["kb_id"]
        entities_by_name: dict[str, GraphEntity] = {}

        for item in _as_list(payload.get("entities"))[:36]:
            if not isinstance(item, Mapping):
                continue
            name = _clean_name(item.get("name"))
            if not name:
                continue
            entity_type = _infer_entity_type(name, item.get("type"))
            entities_by_name[_key(name)] = GraphEntity(
                id=stable_entity_id(name, entity_type),
                name=name,
                type=entity_type,
                description=f"Entity extracted by model {self.model_id}.",
                source_refs=[source_ref],
                metadata={"kb_id": kb_id, "extractor": "llm", "model_id": self.model_id},
            )

        relationships: list[GraphRelationship] = []
        for item in _as_list(payload.get("relationships"))[:56]:
            if not isinstance(item, Mapping):
                continue
            source_name = _clean_name(item.get("source"))
            target_name = _clean_name(item.get("target"))
            relation = _clean_relation(item.get("relation"))
            if not source_name or not target_name or _key(source_name) == _key(target_name):
                continue
            source_entity = entities_by_name.get(_key(source_name))
            target_entity = entities_by_name.get(_key(target_name))
            if not source_entity or not target_entity:
                continue
            relationships.append(
                GraphRelationship(
                    id=stable_relationship_id(source_entity.id, target_entity.id, relation),
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relation_type=relation,
                    description=f"{source_name} {relation} {target_name}",
                    weight=1.0,
                    source_refs=[source_ref],
                    metadata={"kb_id": kb_id, "extractor": "llm", "model_id": self.model_id},
                )
            )

        return GraphExtractionResult(
            entities=list(entities_by_name.values()),
            relationships=relationships,
        )


def _parse_json_payload(value: Any) -> Mapping[str, Any]:
    text = str(value or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    elif "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("model response is not a JSON object")
    return payload


def _chunk_data(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "to_dict"):
        data = chunk.to_dict()
    elif isinstance(chunk, Mapping):
        data = dict(chunk)
    else:
        data = {
            key: getattr(chunk, key)
            for key in ("id", "document_id", "kb_id", "text", "metadata")
            if hasattr(chunk, key)
        }
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    return {
        "chunk_id": str(data.get("id") or data.get("chunk_id") or "chunk-unknown"),
        "document_id": str(data.get("document_id") or "document-unknown"),
        "kb_id": str(data.get("kb_id") or data.get("knowledge_base_id") or "default"),
        "text": str(data.get("text") or data.get("content") or "").strip(),
        "metadata": dict(metadata),
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:80].strip(" `\"'.,;:()[]{}")


def _clean_type(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(value or "concept").strip())
    return (text or "concept")[:32]


def _infer_entity_type(name: str, raw_type: Any = None) -> str:
    text = name.strip()
    lowered_type = str(raw_type or "").strip().lower()
    if _looks_like_metric(text):
        return "metric"
    if _looks_like_person(text, lowered_type):
        return "person"
    if _looks_like_project_or_org(text, lowered_type):
        return "project"
    if lowered_type in {"person", "people", "human", "人物"}:
        return "person"
    if lowered_type in {"metric", "number", "date", "amount", "数值", "日期"}:
        return "metric"
    if lowered_type in {"project", "organization", "org", "institution", "company", "项目", "机构"}:
        return "project"
    return "metric" if lowered_type == "concept" else _clean_type(raw_type)


def _looks_like_metric(text: str) -> bool:
    patterns = [
        r"\d+(?:\.\d+)?\s*(?:万|万元|元|亿|%|kg|g|GB|MB|tokens?)",
        r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*[日号])?",
        r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]",
        r"^\d+(?:\.\d+)?$",
    ]
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _looks_like_person(text: str, lowered_type: str) -> bool:
    if lowered_type in {"person", "people", "human", "人物"}:
        return True
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", text):
        non_person_keywords = {
            "项目",
            "计划",
            "成果",
            "系统",
            "模型",
            "技术",
            "知识",
            "图谱",
            "机构",
            "公司",
            "大学",
            "学院",
            "团队",
        }
        return not any(keyword in text for keyword in non_person_keywords)
    return False


def _looks_like_project_or_org(text: str, lowered_type: str) -> bool:
    if lowered_type in {"project", "organization", "org", "institution", "company", "项目", "机构"}:
        return True
    keywords = [
        "项目",
        "计划",
        "系统",
        "平台",
        "机构",
        "公司",
        "大学",
        "学院",
        "成果",
        "实验室",
        "知识库",
        "模型",
    ]
    return any(keyword in text for keyword in keywords)


def _clean_relation(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(value or "related_to").strip())
    return (text or "related_to")[:48]


def _key(value: str) -> str:
    return value.casefold().strip()


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = ["LLMGraphExtractor"]
