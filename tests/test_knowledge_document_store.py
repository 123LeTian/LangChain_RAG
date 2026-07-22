from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.api.app import app


def _kb(kb_id: str, name: str) -> dict:
    return {
        "id": kb_id,
        "name": name,
        "description": "",
        "status": "ready",
        "owner_id": "user_001",
        "doc_count": 0,
        "chunk_count": 0,
    }


@pytest.fixture
def knowledge_client(tmp_path, monkeypatch):
    from src.api.routes import knowledge as knowledge_routes

    monkeypatch.setattr(knowledge_routes, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        knowledge_routes,
        "_DATA_FILE",
        tmp_path / "data" / "knowledge_bases.json",
    )
    monkeypatch.setattr(
        knowledge_routes,
        "_MOCK_KBS",
        [_kb("kb_alpha", "Alpha"), _kb("kb_beta", "Beta")],
    )
    monkeypatch.setattr(knowledge_routes, "_MOCK_DOCS", [])
    monkeypatch.setattr(knowledge_routes, "_MOCK_JOBS", {})

    with TestClient(app) as client:
        yield client, knowledge_routes


def _upload(client: TestClient, kb_id: str, content: bytes) -> dict:
    response = client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        files={"file": ("plan.txt", content, "text/plain")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_same_filename_uploads_are_isolated_by_kb_storage_path(knowledge_client):
    client, knowledge_routes = knowledge_client

    alpha = _upload(client, "kb_alpha", b"Alpha owner is Alice.")
    beta = _upload(client, "kb_beta", b"Beta owner is Bob.")

    assert alpha["filename"] == "plan.txt"
    assert beta["filename"] == "plan.txt"
    assert alpha["storage_path"] != beta["storage_path"]
    assert alpha["storage_path"].startswith("kb_alpha/")
    assert beta["storage_path"].startswith("kb_beta/")

    alpha_path = knowledge_routes._document_path(alpha)
    beta_path = knowledge_routes._document_path(beta)
    assert alpha_path.read_bytes() == b"Alpha owner is Alice."
    assert beta_path.read_bytes() == b"Beta owner is Bob."

    alpha_preview = client.get(
        f"/api/knowledge-bases/kb_alpha/documents/{alpha['id']}/preview"
    )
    beta_preview = client.get(
        f"/api/knowledge-bases/kb_beta/documents/{beta['id']}/preview"
    )
    assert alpha_preview.status_code == 200
    assert beta_preview.status_code == 200
    assert "Alice" in alpha_preview.json()["text"]
    assert "Bob" in beta_preview.json()["text"]


def test_deleting_one_document_does_not_delete_same_filename_in_other_kb(
    knowledge_client,
):
    client, knowledge_routes = knowledge_client
    alpha = _upload(client, "kb_alpha", b"Alpha owner is Alice.")
    beta = _upload(client, "kb_beta", b"Beta owner is Bob.")
    alpha_path = knowledge_routes._document_path(alpha)
    beta_path = knowledge_routes._document_path(beta)

    response = client.delete(
        f"/api/knowledge-bases/kb_alpha/documents/{alpha['id']}"
    )

    assert response.status_code == 204
    assert not alpha_path.exists()
    assert beta_path.exists()
    beta_preview = client.get(
        f"/api/knowledge-bases/kb_beta/documents/{beta['id']}/preview"
    )
    assert beta_preview.status_code == 200
    assert "Bob" in beta_preview.json()["text"]


def test_deleting_kb_cascades_only_its_own_document_files(knowledge_client):
    client, knowledge_routes = knowledge_client
    alpha = _upload(client, "kb_alpha", b"Alpha owner is Alice.")
    beta = _upload(client, "kb_beta", b"Beta owner is Bob.")
    alpha_path = knowledge_routes._document_path(alpha)
    beta_path = knowledge_routes._document_path(beta)

    response = client.delete("/api/knowledge-bases/kb_alpha")

    assert response.status_code == 204
    assert not alpha_path.exists()
    assert beta_path.exists()
    docs = client.get("/api/knowledge-bases/kb_beta/documents")
    assert docs.status_code == 200
    assert docs.json()[0]["id"] == beta["id"]


def test_graph_chunk_loader_uses_document_storage_path(tmp_path, monkeypatch):
    from src.api.routes import graph as graph_routes
    from src.api.routes import knowledge as knowledge_routes

    docs_dir = tmp_path / "documents" / "kb_graph"
    docs_dir.mkdir(parents=True)
    stored = docs_dir / "doc_graph__graph.txt"
    stored.write_text(
        "RAG depends on Retrieval. Retrieval contains Vector Search.",
        encoding="utf-8",
    )
    monkeypatch.setattr(knowledge_routes, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        knowledge_routes,
        "_MOCK_DOCS",
        [
            {
                "id": "doc_graph",
                "kb_id": "kb_graph",
                "filename": "graph.txt",
                "storage_path": "kb_graph/doc_graph__graph.txt",
                "type": "txt",
                "checksum": "sha",
                "status": "indexed",
                "chunk_count": 0,
                "size_bytes": stored.stat().st_size,
            }
        ],
    )

    chunks = graph_routes._load_graph_chunks("kb_graph")

    assert chunks
    assert chunks[0].kb_id == "kb_graph"
    assert chunks[0].document_id == "doc_graph"
    assert chunks[0].metadata["filename"] == "graph.txt"
    assert chunks[0].metadata["storage_path"] == "kb_graph/doc_graph__graph.txt"
