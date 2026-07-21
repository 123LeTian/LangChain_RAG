"""Evaluation dataset loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REQUIRED_SAMPLE_FIELDS = {
    "id",
    "question",
    "answer_points",
    "expected_sources",
    "tags",
    "difficulty",
}


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class ExpectedSource:
    """A source that should support a correct answer."""

    document_id: str | None = None
    chunk_id: str | None = None
    filename: str | None = None
    quote: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, sample_id: str, line_number: int
    ) -> "ExpectedSource":
        if not isinstance(data, Mapping):
            raise EvaluationDatasetError(
                f"line {line_number}, sample {sample_id}: "
                "expected_sources entries must be objects"
            )
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"document_id", "chunk_id", "filename", "quote"}
        }
        return cls(
            document_id=_optional_str(data.get("document_id")),
            chunk_id=_optional_str(data.get("chunk_id")),
            filename=_optional_str(data.get("filename")),
            quote=_optional_str(data.get("quote")),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "quote": self.quote,
        }
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class EvaluationSample:
    """A single reusable RAG evaluation sample."""

    id: str
    question: str
    answer_points: List[str]
    expected_sources: List[ExpectedSource]
    tags: List[str]
    difficulty: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, line_number: int
    ) -> "EvaluationSample":
        if not isinstance(data, Mapping):
            raise EvaluationDatasetError(f"line {line_number}: sample must be an object")

        missing = sorted(REQUIRED_SAMPLE_FIELDS - set(data))
        sample_id = str(data.get("id", "<missing-id>"))
        if missing:
            raise EvaluationDatasetError(
                f"line {line_number}, sample {sample_id}: missing required fields: "
                f"{', '.join(missing)}"
            )

        sample_id = _required_str(data["id"], "id", line_number, sample_id)
        question = _required_str(data["question"], "question", line_number, sample_id)
        difficulty = _required_str(
            data["difficulty"], "difficulty", line_number, sample_id
        )
        answer_points = _required_str_list(
            data["answer_points"], "answer_points", line_number, sample_id
        )
        tags = _required_str_list(data["tags"], "tags", line_number, sample_id)

        expected_sources_raw = data["expected_sources"]
        if not isinstance(expected_sources_raw, Sequence) or isinstance(
            expected_sources_raw, (str, bytes)
        ):
            raise EvaluationDatasetError(
                f"line {line_number}, sample {sample_id}: "
                "expected_sources must be a list"
            )
        expected_sources = [
            ExpectedSource.from_mapping(
                item, sample_id=sample_id, line_number=line_number
            )
            for item in expected_sources_raw
        ]

        metadata = {
            key: value
            for key, value in data.items()
            if key not in REQUIRED_SAMPLE_FIELDS
        }
        return cls(
            id=sample_id,
            question=question,
            answer_points=answer_points,
            expected_sources=expected_sources,
            tags=tags,
            difficulty=difficulty,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "question": self.question,
            "answer_points": list(self.answer_points),
            "expected_sources": [source.to_dict() for source in self.expected_sources],
            "tags": list(self.tags),
            "difficulty": self.difficulty,
        }
        payload.update(self.metadata)
        return payload


def load_jsonl_dataset(path: str | Path) -> List[EvaluationSample]:
    """Load and validate a JSONL RAG evaluation dataset."""

    dataset_path = Path(path)
    samples: List[EvaluationSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDatasetError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            samples.append(
                EvaluationSample.from_mapping(payload, line_number=line_number)
            )
    return samples


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _required_str(
    value: Any, field_name: str, line_number: int, sample_id: str
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(
            f"line {line_number}, sample {sample_id}: {field_name} "
            "must be a non-empty string"
        )
    return value.strip()


def _required_str_list(
    value: Any, field_name: str, line_number: int, sample_id: str
) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationDatasetError(
            f"line {line_number}, sample {sample_id}: {field_name} "
            "must be a list of strings"
        )
    items = list(value)
    if not items or not all(isinstance(item, str) and item.strip() for item in items):
        raise EvaluationDatasetError(
            f"line {line_number}, sample {sample_id}: {field_name} "
            "must contain non-empty strings"
        )
    return [item.strip() for item in items]


__all__ = [
    "EvaluationDatasetError",
    "EvaluationSample",
    "ExpectedSource",
    "load_jsonl_dataset",
]
