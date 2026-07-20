import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import LoaderFactory, TextLoader, MarkdownLoader
from ingestion.models import DocumentRecord


def test_text_loader():
    """测试 TXT 文件加载。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("这是第一段。\n\n这是第二段。")
        temp_path = f.name

    try:
        loader = TextLoader()
        doc = loader.load(temp_path, kb_id="kb-001", document_id="doc-001")

        assert isinstance(doc, DocumentRecord)
        assert doc.file_type == "txt"
        assert "这是第一段" in doc.text
        assert doc.kb_id == "kb-001"
        assert len(doc.checksum) == 64
    finally:
        os.unlink(temp_path)


def test_markdown_loader():
    """测试 Markdown 文件加载。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# 标题\n\n这是正文。")
        temp_path = f.name

    try:
        doc = MarkdownLoader().load(temp_path, kb_id="kb-001", document_id="doc-002")
        assert doc.file_type == "markdown"
        assert "# 标题" in doc.text
    finally:
        os.unlink(temp_path)


def test_loader_factory():
    """测试 LoaderFactory 自动选择。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("工厂模式测试")
        temp_path = f.name

    try:
        doc = LoaderFactory.load(temp_path, kb_id="kb-001", document_id="doc-003")
        assert doc.file_type == "txt"
        assert "工厂模式测试" in doc.text
    finally:
        os.unlink(temp_path)


def test_checksum_consistency():
    """测试相同内容生成相同校验和。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("相同内容")
        temp_path = f.name

    try:
        doc1 = TextLoader().load(temp_path, kb_id="kb-001", document_id="doc-a")
        doc2 = TextLoader().load(temp_path, kb_id="kb-001", document_id="doc-b")
        assert doc1.checksum == doc2.checksum
    finally:
        os.unlink(temp_path)


def test_unsupported_file():
    """测试不支持的文件类型。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        temp_path = f.name

    try:
        try:
            LoaderFactory.load(temp_path, kb_id="kb-001", document_id="doc-004")
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "不支持的文件类型" in str(e)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_text_loader()
    print("test_text_loader passed")
    
    test_markdown_loader()
    print("test_markdown_loader passed")
    
    test_loader_factory()
    print("test_loader_factory passed")
    
    test_checksum_consistency()
    print("test_checksum_consistency passed")
    
    test_unsupported_file()
    print("test_unsupported_file passed")
    
    print("\nAll tests passed!")
