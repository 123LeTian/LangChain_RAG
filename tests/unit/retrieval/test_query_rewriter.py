import asyncio

import pytest

from src.retrieval.query_rewriter import (
    QueryRewriter,
    normalize_queries,
    rewrite_queries,
)


def run(coro):
    return asyncio.run(coro)


def test_sync_rewriter_keeps_original_deduplicates_and_drops_blanks():
    rewriter = QueryRewriter(
        lambda query: [" rewrite one ", "", query, "rewrite one", "rewrite two"]
    )

    result = run(rewriter.rewrite(" original ", max_queries=3))

    assert result == ["original", "rewrite one", "rewrite two"]


def test_async_rewriter_honors_max_query_limit():
    async def rewrite(_query):
        return ["one", "two", "three"]

    result = run(QueryRewriter(rewrite).rewrite("original", max_queries=2))

    assert result == ["original", "one"]


def test_empty_rewrite_falls_back_to_original_query():
    assert normalize_queries("original", [], max_queries=3) == ["original"]
    assert run(QueryRewriter(lambda _query: None).rewrite("original")) == [
        "original"
    ]


def test_generic_rewriter_adapter_supports_sync_objects():
    class Rewriter:
        def rewrite(self, query, *, max_queries):
            return [query, "alternative"]

    assert run(rewrite_queries(Rewriter(), "original", max_queries=3)) == [
        "original",
        "alternative",
    ]


@pytest.mark.parametrize("value", ["single string", 42])
def test_invalid_rewrite_shape_is_explicit(value):
    with pytest.raises(TypeError, match="iterable of strings"):
        run(QueryRewriter(lambda _query: value).rewrite("original"))
