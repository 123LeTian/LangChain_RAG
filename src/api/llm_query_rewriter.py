"""LLM-powered query rewriter for RAG.

Uses DeepSeek to rewrite user queries into multiple search-friendly variants.
For example, "半年利润" -> ["归属于母公司股东的净利润", "半年度净利润", "利润总额"].
This improves retrieval recall by expanding vague queries into precise terms.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List

from src.retrieval.query_rewriter import QueryRewriterProtocol, normalize_queries


class LLMQueryRewriter:
    """Rewrite queries using DeepSeek LLM for better retrieval coverage.

    Instead of a simple callable adapter, this uses a real LLM to:
    1. Expand abbreviations (半年利润 -> 半年度净利润)
    2. Generate synonyms (利润 -> 净利润, 利润总额, 归母净利润)
    3. Produce formal document-style queries
    """

    def __init__(self, llm=None):
        """Initialize with an optional pre-configured LLM instance."""
        self._llm = llm
        self._init_lock = asyncio.Lock()

    def _get_llm(self):
        """Lazily create LLM if not provided."""
        if self._llm is not None:
            return self._llm

        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.3,
            max_tokens=200,
        )
        return self._llm

    def _build_prompt(self, query: str, max_queries: int) -> str:
        """Build the rewrite prompt for the LLM."""
        return (
            "你是一个查询改写助手。用户的问题可能含糊或口语化，"
            "请将其改写为更精确的检索查询，以便在知识库中找到相关文档。\n\n"
            "规则：\n"
            "1. 保留原始查询的含义\n"
            "2. 使用正式文档中可能出现的术语\n"
            "3. 生成同义词和变体\n"
            f"4. 最多生成 {max_queries} 个改写查询\n"
            "5. 每行一个查询，不要编号\n\n"
            f"用户问题：{query}\n\n"
            "改写查询："
        )

    def _parse_response(self, response_text: str) -> List[str]:
        """Parse LLM response into a list of queries."""
        lines = response_text.strip().split("\n")
        queries = []
        for line in lines:
            # Remove numbering like "1. " or "- " or "（1）"
            cleaned = re.sub(r'^[\d\-•（(][\d）).]*\s*', '', line).strip()
            # Remove quotes
            cleaned = cleaned.strip('"\'「」""')
            if cleaned and len(cleaned) > 1:
                queries.append(cleaned)
        return queries

    async def rewrite(
        self,
        query: str,
        *,
        max_queries: int = 3,
    ) -> List[str]:
        """Rewrite a query into multiple variants using LLM.

        Args:
            query: Original user query.
            max_queries: Max number of rewritten queries (excluding original).

        Returns:
            List of queries including the original + rewrites.
        """
        try:
            llm = self._get_llm()
            prompt = self._build_prompt(query, max_queries)
            response = await llm.ainvoke(prompt)
            rewrites = self._parse_response(response.content)
            return normalize_queries(query, rewrites, max_queries=max_queries + 1)
        except Exception as e:
            print(f"[QueryRewriter] Rewrite failed, using original: {e}", flush=True)
            return [query.strip()]


# Singleton instance (lazy LLM init)
_rewriter: LLMQueryRewriter | None = None


def get_query_rewriter(llm=None) -> LLMQueryRewriter:
    """Get or create the singleton LLMQueryRewriter."""
    global _rewriter
    if _rewriter is None:
        _rewriter = LLMQueryRewriter(llm=llm)
    return _rewriter