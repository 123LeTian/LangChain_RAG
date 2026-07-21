"""
Knowledge Model — Owner: Shared
知识库数据模型：KnowledgeBase、DocumentRecord、ChunkRecord。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class KnowledgeBaseStatus(str, Enum):
    """知识库生命周期状态"""
    CREATING = "creating"      # 创建中
    READY = "ready"            # 就绪，可检索
    INDEXING = "indexing"      # 索引构建中
    ERROR = "error"            # 出错


class KnowledgeBase(BaseModel):
    """知识库元数据 — 所有文档的逻辑容器"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16], description="知识库唯一 ID")
    owner_id: str = Field(..., description="所有者用户 ID")
    name: str = Field(..., min_length=1, description="知识库名称")
    description: str = Field(default="", description="知识库描述")
    status: KnowledgeBaseStatus = Field(default=KnowledgeBaseStatus.CREATING, description="生命周期状态")
    embedding_model: str = Field(default="text-embedding-3-small", description="使用的 Embedding 模型名")
    doc_count: int = Field(default=0, description="已索引文档数")
    chunk_count: int = Field(default=0, description="已切分 Chunk 数")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentStatus(str, Enum):
    """文档处理状态"""
    UPLOADED = "uploaded"      # 已上传，等待处理
    PARSING = "parsing"        # 解析中
    CHUNKING = "chunking"      # 切分中
    INDEXED = "indexed"        # 已索引，可检索
    ERROR = "error"            # 处理失败


class DocumentType(str, Enum):
    """支持的文档类型"""
    TXT = "txt"
    MD = "md"
    PDF = "pdf"
    DOCX = "docx"


class DocumentRecord(BaseModel):
    """文档记录 — 一个上传文件在系统中的完整生命周期"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16], description="文档唯一 ID")
    kb_id: str = Field(..., description="所属知识库 ID")
    filename: str = Field(..., description="原始文件名")
    type: DocumentType = Field(..., description="文档类型（txt/md/pdf/docx）")
    checksum: str = Field(..., description="文件哈希值，用于去重")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="处理状态")
    chunk_count: int = Field(default=0, description="已生成的 Chunk 数量")
    size_bytes: int = Field(default=0, description="文件大小（字节）")
    error_message: Optional[str] = Field(default=None, description="处理失败时的错误信息")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_at: Optional[datetime] = Field(default=None, description="索引完成时间")


class ChunkRecord(BaseModel):
    """Chunk 记录 — 所有检索模式共享的最小文本单元"""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16], description="Chunk 唯一 ID（稳定，不可变）")
    document_id: str = Field(..., description="来源文档 ID")
    kb_id: str = Field(..., description="所属知识库 ID")
    text: str = Field(..., description="Chunk 文本内容")
    index: int = Field(..., description="Chunk 在文档中的序号（从 0 开始）")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="来源元数据：filename/page/section/token_count 等"
    )
    embedding: Optional[list[float]] = Field(default=None, description="向量嵌入（缓存用，不持久化到此模型）")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
