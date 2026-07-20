"""引用生成模块：把检索结果转成可溯源的引用。"""
from typing import List
from src.models.schemas import RetrievalHit, Citation


class CitationBuilder:
    """从检索命中结果生成引用。"""

    def build_citations(self, hits: List[RetrievalHit]) -> List[Citation]:
        """把 RetrievalHit 列表转成 Citation 列表。"""
        citations = []
        for hit in hits:
            meta = hit.chunk.metadata or {}
            filename = meta.get("filename", "未知文档")
            page = meta.get("page")
            quote = hit.chunk.text[:200]
            citations.append(Citation(
                document_id=hit.chunk.document_id,
                chunk_id=hit.chunk.id,
                filename=filename,
                page=page,
                quote=quote,
                score=hit.score,
            ))
        return citations

    def format_citations(self, citations: List[Citation]) -> str:
        """把引用格式化成文本。"""
        lines = []
        for i, cite in enumerate(citations, 1):
            page_str = f"第{cite.page}页, " if cite.page else ""
            lines.append(f"[{i}] {cite.filename}, {page_str}chunk={cite.chunk_id}, 相关度={cite.score:.4f}")
        return "\n".join(lines)
