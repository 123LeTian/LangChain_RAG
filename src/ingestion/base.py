from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from .models import DocumentRecord


class BaseLoader(ABC):
    """文档加载器抽象基类。"""

    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        """解析文件并返回 DocumentRecord。"""
        ...

    def _read_bytes(self, file_path: Union[str, Path]) -> bytes:
        with open(file_path, "rb") as f:
            return f.read()

    def _compute_checksum(self, content: bytes) -> str:
        import hashlib
        return hashlib.sha256(content).hexdigest()

    def supports(self, file_path: Union[str, Path]) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions
