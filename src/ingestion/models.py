from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class DocumentRecord:
    """文档记录，Loader 解析后的统一输出格式。"""

    id: str
    kb_id: str
    filename: str
    file_type: str
    text: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "parsed"  # parsed, failed
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kb_id": self.kb_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "text": self.text,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        return cls(**data)
