"""LLM-powered query rewriter for RAG.

Uses DeepSeek to rewrite user queries into multiple search-friendly variants.
Generates SHORT, focused search terms rather than full sentences.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import List

from src.retrieval.query_rewriter import QueryRewriterProtocol, normalize_queries


class LLMQueryRewriter:
    """Rewrite queries using DeepSeek LLM for better retrieval coverage."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
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
        """Build the rewrite prompt for the LLM.

        Instructs DeepSeek to generate SHORT, focused search terms (2-8 chars)
        without company names or stock codes.
        """
        return (
            "\u4f60\u662f\u67e5\u8be2\u6539\u5199\u52a9\u624b\u3002"
            "\u5c06\u7528\u6237\u95ee\u9898\u6539\u5199\u4e3a\u7b80\u77ed\u7684\u68c0\u7d22\u8bcd\u3002\n\n"
            "\u89c4\u5219\uff1a\n"
            "1. \u6bcf\u4e2a\u67e5\u8be2\u53ea\u5305\u542b\u5173\u952e\u8d22\u52a1\u672f\u8bed\uff0c2-8\u4e2a\u5b57\n"
            "2. \u4e0d\u8981\u5305\u542b\u516c\u53f8\u540d\u3001\u80a1\u7968\u4ee3\u7801\u7b49\u5e38\u89c1\u8bcd\n"
            "3. \u751f\u6210\u540c\u4e49\u8bcd\u548c\u53d8\u4f53\n"
            f"4. \u6700\u591a {max_queries} \u4e2a\uff0c\u6bcf\u884c\u4e00\u4e2a\uff0c\u65e0\u7f16\u53f7\n\n"
            f"\u95ee\u9898\uff1a{query}\n\n"
            "\u6539\u5199\uff1a"
        )

    def _parse_response(self, response_text: str) -> List[str]:
        lines = response_text.strip().split("\n")
        queries = []
        for line in lines:
            cleaned = re.sub(r"^[\d\-\u2500\u3010(][\d\uff1a.]*\s*", "", line).strip()
            cleaned = cleaned.strip("\"'")
            if cleaned and len(cleaned) > 1:
                queries.append(cleaned)
        return queries

    async def rewrite(self, query: str, *, max_queries: int = 3) -> List[str]:
        try:
            llm = self._get_llm()
            prompt = self._build_prompt(query, max_queries)
            response = await asyncio.to_thread(llm.invoke, prompt)
            rewrites = self._parse_response(response.content)
            return normalize_queries(query, rewrites, max_queries=max_queries + 1)
        except Exception as e:
            print(f"[QueryRewriter] Rewrite failed, using original: {e}", flush=True)
            return [query.strip()]


_rewriter: LLMQueryRewriter | None = None

def get_query_rewriter(llm=None) -> LLMQueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = LLMQueryRewriter(llm=llm)
    return _rewriter
