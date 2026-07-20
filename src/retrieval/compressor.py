"""上下文压缩模块：控制传给 LLM 的文本量。"""
from typing import List
from src.models.schemas import RetrievalHit


class ContextCompressor:
    """压缩检索结果，控制 Token 预算。"""

    def __init__(self, max_tokens: int = 2000, chars_per_token: int = 2):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def compress(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """按分数从高到低，保留不超过 Token 预算的 Chunk。"""
        max_chars = self.max_tokens * self.chars_per_token
        total = 0
        result = []
        for hit in sorted(hits, key=lambda h: h.score, reverse=True):
            text_len = len(hit.chunk.text)
            if total + text_len > max_chars:
                # 截断最后一个
                remaining = max_chars - total
                if remaining > 50:
                    hit.chunk.text = hit.chunk.text[:remaining] + "..."
                    result.append(hit)
                break
            result.append(hit)
            total += text_len
        # 重新编号
        for rank, hit in enumerate(result):
            hit.rank = rank
        return result

    def deduplicate(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """去重：相同文本只保留分数最高的。"""
        seen = {}
        for hit in hits:
            key = hit.chunk.text.strip()
            if key not in seen or hit.score > seen[key].score:
                seen[key] = hit
        result = sorted(seen.values(), key=lambda h: h.score, reverse=True)
        for rank, hit in enumerate(result):
            hit.rank = rank
        return result
