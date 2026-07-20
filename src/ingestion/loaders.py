import re
from pathlib import Path
from typing import Union
from .base import BaseLoader
from .models import DocumentRecord


class TextLoader(BaseLoader):
    """加载 TXT 文件。"""

    supported_extensions = [".txt"]

    def load(self, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        file_path = Path(file_path)
        content_bytes = self._read_bytes(file_path)
        checksum = self._compute_checksum(content_bytes)

        # 尝试 UTF-8，失败则尝试 GBK
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("gbk", errors="ignore")

        return DocumentRecord(
            id=document_id,
            kb_id=kb_id,
            filename=file_path.name,
            file_type="txt",
            text=text,
            checksum=checksum,
            metadata={"file_size": len(content_bytes)},
        )


class MarkdownLoader(BaseLoader):
    """加载 Markdown 文件。"""

    supported_extensions = [".md", ".markdown"]

    def load(self, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        file_path = Path(file_path)
        content_bytes = self._read_bytes(file_path)
        checksum = self._compute_checksum(content_bytes)

        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("gbk", errors="ignore")

        return DocumentRecord(
            id=document_id,
            kb_id=kb_id,
            filename=file_path.name,
            file_type="markdown",
            text=text,
            checksum=checksum,
            metadata={"file_size": len(content_bytes)},
        )


class PDFLoader(BaseLoader):
    """加载 PDF 文件。"""

    supported_extensions = [".pdf"]

    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images

    def load(self, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        file_path = Path(file_path)
        content_bytes = self._read_bytes(file_path)
        checksum = self._compute_checksum(content_bytes)

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

        reader = PdfReader(str(file_path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        text = "\n\n".join(pages)
        # 清理多余空格和换行
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)

        return DocumentRecord(
            id=document_id,
            kb_id=kb_id,
            filename=file_path.name,
            file_type="pdf",
            text=text,
            checksum=checksum,
            metadata={"file_size": len(content_bytes), "page_count": len(reader.pages)},
        )


class DocxLoader(BaseLoader):
    """加载 DOCX 文件。"""

    supported_extensions = [".docx"]

    def load(self, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        file_path = Path(file_path)
        content_bytes = self._read_bytes(file_path)
        checksum = self._compute_checksum(content_bytes)

        try:
            from docx import Document
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

        return DocumentRecord(
            id=document_id,
            kb_id=kb_id,
            filename=file_path.name,
            file_type="docx",
            text=text,
            checksum=checksum,
            metadata={"file_size": len(content_bytes)},
        )


class LoaderFactory:
    """根据文件后缀自动选择 Loader。"""

    _loaders = [
        TextLoader(),
        MarkdownLoader(),
        PDFLoader(),
        DocxLoader(),
    ]

    @classmethod
    def get_loader(cls, file_path: Union[str, Path]) -> BaseLoader:
        file_path = Path(file_path)
        for loader in cls._loaders:
            if loader.supports(file_path):
                return loader
        raise ValueError(f"不支持的文件类型: {file_path.suffix}")

    @classmethod
    def load(cls, file_path: Union[str, Path], kb_id: str, document_id: str) -> DocumentRecord:
        loader = cls.get_loader(file_path)
        return loader.load(file_path, kb_id, document_id)

    @classmethod
    def register(cls, loader: BaseLoader) -> None:
        """注册自定义 Loader。"""
        cls._loaders.append(loader)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """返回所有支持的文件后缀。"""
        extensions = []
        for loader in cls._loaders:
            extensions.extend(loader.supported_extensions)
        return list(set(extensions))
