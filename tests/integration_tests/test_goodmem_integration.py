"""Integration tests for GoodMem LangChain tools.

These tests run against a live GoodMem API instance. Set the following
environment variables before running:

    GOODMEM_BASE_URL    - e.g. https://localhost:8080
    GOODMEM_API_KEY     - your GoodMem API key
    GOODMEM_TEST_PDF    - path to a PDF file for upload testing
    GOODMEM_EMBEDDER_ID - the embedder ID to use when creating spaces
    GOODMEM_VERIFY_SSL  - set to "true" to verify SSL (default: false)

Run with:
    uv run --group test pytest tests/integration_tests/ -v
"""

import json
import os
import time

import pytest

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemGetMemory,
    GoodMemListEmbedders,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
)

BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
API_KEY = os.environ.get("GOODMEM_API_KEY", "")
TEST_PDF = os.environ.get("GOODMEM_TEST_PDF", "")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "false").lower() == "true"

_COMMON_KWARGS: dict[str, object] = {
    "goodmem_base_url": BASE_URL,
    "goodmem_api_key": API_KEY,
    "goodmem_verify_ssl": VERIFY_SSL,
}

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="GOODMEM_API_KEY not set",
)


def _get_embedder_id() -> str:
    """Return the configured embedder ID, or fetch the first available one."""
    if EMBEDDER_ID:
        return EMBEDDER_ID
    tool = GoodMemListEmbedders(**_COMMON_KWARGS)
    raw = tool.invoke({})
    result = json.loads(raw)
    return result["embedders"][0]["embedderId"]


class TestListEmbedders:
    def test_list_embedders_returns_results(self) -> None:
        tool = GoodMemListEmbedders(**_COMMON_KWARGS)
        raw = tool.invoke({})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["totalEmbedders"] > 0
        assert len(result["embedders"]) > 0


class TestListSpaces:
    def test_list_spaces_succeeds(self) -> None:
        tool = GoodMemListSpaces(**_COMMON_KWARGS)
        raw = tool.invoke({})
        result = json.loads(raw)
        assert result["success"] is True
        assert isinstance(result["spaces"], list)


class TestCreateSpace:
    def test_create_space(self) -> None:
        embedder_id = _get_embedder_id()
        space_name = f"langchain-test-{int(time.time())}"
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)
        raw = tool.invoke({"name": space_name, "embedder_id": embedder_id})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["spaceId"]
        assert result["name"] == space_name

    def test_create_space_reuses_existing(self) -> None:
        embedder_id = _get_embedder_id()
        space_name = f"langchain-reuse-{int(time.time())}"
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)

        # Create first
        raw1 = tool.invoke({"name": space_name, "embedder_id": embedder_id})
        result1 = json.loads(raw1)
        assert result1["success"] is True

        # Create again - should reuse
        raw2 = tool.invoke({"name": space_name, "embedder_id": embedder_id})
        result2 = json.loads(raw2)
        assert result2["success"] is True
        assert result2["reused"] is True
        assert result2["spaceId"] == result1["spaceId"]


class TestEndToEndFlow:
    """Full lifecycle: create space -> create memory -> retrieve -> get -> delete."""

    @pytest.fixture()
    def space_id(self) -> str:
        embedder_id = _get_embedder_id()
        space_name = f"langchain-e2e-{int(time.time())}"
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)
        raw = tool.invoke({"name": space_name, "embedder_id": embedder_id})
        result = json.loads(raw)
        assert result["success"] is True
        return result["spaceId"]

    def test_text_memory_lifecycle(self, space_id: str) -> None:
        # Create text memory
        create_tool = GoodMemCreateMemory(**_COMMON_KWARGS)
        raw = create_tool.invoke({
            "space_id": space_id,
            "text_content": (
                "LangChain is a framework for developing applications "
                "powered by large language models."
            ),
            "metadata": {"source": "test", "tags": ["langchain", "test"]},
        })
        result = json.loads(raw)
        assert result["success"] is True
        memory_id = result["memoryId"]
        assert memory_id

        # Get memory
        get_tool = GoodMemGetMemory(**_COMMON_KWARGS)
        raw = get_tool.invoke({"memory_id": memory_id})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["memory"]["memoryId"] == memory_id

        # Retrieve memories (wait for indexing so embeddings are ready)
        retrieve_tool = GoodMemRetrieveMemories(**_COMMON_KWARGS)
        raw = retrieve_tool.invoke({
            "query": "LangChain framework",
            "space_ids": space_id,
            "max_results": 5,
            "wait_for_indexing": True,
        })
        result = json.loads(raw)
        assert result["success"] is True
        assert result["totalResults"] > 0
        # Verify chunk structure
        first_chunk = result["results"][0]
        assert "chunkText" in first_chunk
        assert "relevanceScore" in first_chunk

        # Delete memory
        delete_tool = GoodMemDeleteMemory(**_COMMON_KWARGS)
        raw = delete_tool.invoke({"memory_id": memory_id})
        result = json.loads(raw)
        assert result["success"] is True

    @pytest.mark.skipif(not TEST_PDF, reason="GOODMEM_TEST_PDF not set")
    def test_pdf_memory_creation(self, space_id: str) -> None:
        create_tool = GoodMemCreateMemory(**_COMMON_KWARGS)
        raw = create_tool.invoke({
            "space_id": space_id,
            "file_path": TEST_PDF,
            "metadata": {"source": "pdf-test"},
        })
        result = json.loads(raw)
        assert result["success"] is True
        assert result["memoryId"]
        assert result["contentType"] == "application/pdf"

        # Clean up
        delete_tool = GoodMemDeleteMemory(**_COMMON_KWARGS)
        delete_tool.invoke({"memory_id": result["memoryId"]})
