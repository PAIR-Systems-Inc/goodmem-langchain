"""LangChain Documents use the SDK's batch API and retain write receipts on failure."""

import json

import httpx
import pytest
from langchain_core.documents import Document

from langchain_goodmem import GoodMemIngestionError, add_documents
from tests.unit_tests.conftest import MEMORY, MEMORY_ID, MEMORY_ID_2, SPACE_ID, Wire


def test_add_documents_preserves_metadata_ids_and_waits(wire: Wire) -> None:
    documents = [
        Document(
            id="00000000-0000-4000-8000-000000000001",
            page_content="Café launch notes",
            metadata={
                "source": "https://example.org/cobalt",
                "team": "blue",
                "tags": ["launch"],
            },
        ),
        Document(page_content="Second document", metadata={"team": "red"}),
    ]
    memories = [
        MEMORY | {"memoryId": mid, "metadata": document.metadata}
        for mid, document in zip([documents[0].id, MEMORY_ID_2], documents)
    ]
    wire.responses.extend(
        [
            httpx.Response(
                200,
                json={
                    "results": [
                        {"success": True, "memory": m | {"processingStatus": "PENDING"}}
                        for m in memories
                    ]
                },
            ),
            *(httpx.Response(200, json=m) for m in memories),
        ]
    )
    ids = add_documents(wire.sdk, SPACE_ID, iter(documents))
    assert ids == [documents[0].id, MEMORY_ID_2]
    request = wire.requests[0]
    assert request.url.path == "/v1/memories:batchCreate"
    items = json.loads(request.content)["requests"]
    assert items[0]["memoryId"] == documents[0].id
    assert [item["originalContent"] for item in items] == [
        doc.page_content for doc in documents
    ]
    assert [item["metadata"] for item in items] == [doc.metadata for doc in documents]
    assert [request.url.path for request in wire.requests[1:]] == [
        f"/v1/memories/{mid}" for mid in ids
    ]
    assert not wire.http.is_closed


def test_background_ingestion_returns_ids_without_polling(wire: Wire) -> None:
    wire.responses.append(
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "success": True,
                        "memory": MEMORY | {"processingStatus": "PENDING"},
                    }
                ]
            },
        )
    )
    assert add_documents(
        wire.sdk, SPACE_ID, [Document(page_content="Text")], wait=False
    ) == [MEMORY_ID]
    assert len(wire.requests) == 1


def test_partial_batch_retains_created_ids(wire: Wire) -> None:
    wire.responses.append(
        httpx.Response(
            200,
            json={
                "results": [
                    {"success": True, "memory": MEMORY},
                    {
                        "success": False,
                        "error": {"code": 409, "message": "ID already exists"},
                        "requestIndex": 1,
                    },
                ]
            },
        )
    )
    with pytest.raises(GoodMemIngestionError, match="ID already exists") as error:
        add_documents(
            wire.sdk,
            SPACE_ID,
            [Document(page_content="First"), Document(page_content="Second")],
        )
    assert error.value.created_memory_ids == [MEMORY_ID]
    assert len(wire.requests) == 1


def test_indexing_timeout_retains_all_created_ids(wire: Wire) -> None:
    second = MEMORY | {"memoryId": MEMORY_ID_2}
    wire.responses.extend(
        [
            httpx.Response(
                200,
                json={
                    "results": [
                        {"success": True, "memory": m} for m in [MEMORY, second]
                    ]
                },
            ),
            httpx.Response(200, json=MEMORY | {"processingStatus": "PENDING"}),
        ]
    )
    with pytest.raises(GoodMemIngestionError, match="exceeded") as error:
        add_documents(
            wire.sdk,
            SPACE_ID,
            [Document(page_content="First"), Document(page_content="Second")],
            indexing_timeout=0,
        )
    assert error.value.created_memory_ids == [MEMORY_ID, MEMORY_ID_2]
    assert [request.method for request in wire.requests] == ["POST", "GET"]


def test_empty_input_does_not_connect(wire: Wire) -> None:
    assert add_documents(wire.sdk, SPACE_ID, []) == []
    assert not wire.requests
