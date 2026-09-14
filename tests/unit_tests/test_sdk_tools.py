"""SDK-backed tool behavior without a second client or legacy response contract."""

import base64
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from goodmem import Goodmem
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, ToolException
from pydantic import ValidationError

import langchain_goodmem
from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    GoodMemUpdateSpace,
    wait_for_memory,
)
from tests.unit_tests.conftest import CHUNK, MEMORY, SPACE, Wire, ndjson


def test_space_crud_uses_sdk_defaults_and_fields(wire: Wire) -> None:
    wire.responses.extend([httpx.Response(200, json=SPACE)] * 3 + [httpx.Response(204)])
    created = GoodMemCreateSpace(client=wire.sdk).invoke(
        {"name": "Docs", "embedder_id": "embedder-1"}
    )
    assert created["space_id"] == "space-1"
    body = json.loads(wire.requests[0].content)
    assert body["defaultChunkingConfig"]["recursive"]["chunkOverlap"] == 64
    assert wire.requests[0].method == "POST"  # No find/reuse policy before creation.
    assert GoodMemGetSpace(client=wire.sdk).invoke({"space_id": "space-1"}) == created
    GoodMemUpdateSpace(client=wire.sdk).invoke(
        {"space_id": "space-1", "merge_labels": {"team": "docs"}}
    )
    assert json.loads(wire.requests[-1].content) == {"mergeLabels": {"team": "docs"}}
    assert GoodMemDeleteSpace(client=wire.sdk).invoke({"space_id": "space-1"}) is None
    assert [r.method for r in wire.requests] == ["POST", "GET", "PUT", "DELETE"]


@pytest.mark.parametrize("kind", ["spaces", "memories"])
def test_list_tools_follow_sdk_pages_and_respect_limits(wire: Wire, kind: str) -> None:
    model, tool, args = (
        (SPACE, GoodMemListSpaces, {"name_filter": "Doc*"})
        if kind == "spaces"
        else (MEMORY, GoodMemListMemories, {"space_id": "space-1"})
    )
    wire.responses.extend(
        [
            httpx.Response(200, json={kind: [model], "nextToken": "page-2"}),
            httpx.Response(200, json={kind: [model]}),
            httpx.Response(200, json={kind: [model, model], "nextToken": "unused"}),
        ]
    )
    assert len(tool(client=wire.sdk).invoke(args | {"max_items": None})) == 2
    assert wire.requests[1].url.params["next_token"] == "page-2"
    assert len(tool(client=wire.sdk).invoke(args | {"max_items": 1})) == 1
    assert len(wire.requests) == 3


def test_creation_conflict_is_a_langchain_tool_error(wire: Wire) -> None:
    wire.responses.extend([httpx.Response(409, json={"message": "Already exists"})] * 2)
    args = {"name": "Docs", "embedder_id": "embedder-1"}
    with pytest.raises(ToolException, match="Already exists"):
        GoodMemCreateSpace(client=wire.sdk).invoke(args)
    result = GoodMemCreateSpace(client=wire.sdk, handle_tool_error=True).invoke(
        {
            "type": "tool_call",
            "id": "call",
            "name": "goodmem_create_space",
            "args": args,
        }
    )
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert "Already exists" in result.content
    assert all(r.method == "POST" for r in wire.requests)


@pytest.mark.parametrize(
    "content",
    [
        {},
        {"original_content": "text", "file_path": "unused"},
        {"file_path": "/nonexistent-goodmem-test-file"},
    ],
)
def test_sdk_input_and_file_errors_use_langchain_error_handling(
    wire: Wire,
    content: dict[str, str],
) -> None:
    result = GoodMemCreateMemory(client=wire.sdk, handle_tool_error=True).invoke(
        {
            "type": "tool_call",
            "id": "call",
            "name": "goodmem_create_memory",
            "args": {"space_id": "space-1", **content},
        }
    )
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert result.content and not wire.requests


def test_creation_waits_for_its_memory_before_search(
    wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = Mock()
    monkeypatch.setattr("langchain_goodmem.ingestion.time.sleep", sleeps)
    wire.responses.extend(
        httpx.Response(200, json=MEMORY | {"processingStatus": status})
        for status in ["PENDING", "PENDING", "PROCESSING", "COMPLETED"]
    )
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    result = GoodMemCreateMemory(client=wire.sdk).invoke(
        {
            "space_id": "space-1",
            "original_content": "Evidence",
            "memory_id": "memory-1",
            "metadata": {"source": "https://example.org"},
        }
    )
    assert (
        result["memory_id"] == "memory-1" and result["processing_status"] == "COMPLETED"
    )
    body = json.loads(wire.requests[0].content)
    assert body["memoryId"] == "memory-1" and body["originalContent"] == "Evidence"
    assert all(
        r.method == "GET" and r.url.path == "/v1/memories/memory-1"
        for r in wire.requests[1:]
    )
    assert sleeps.call_count == 2
    assert GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("Evidence")
    assert wire.requests[-1].url.path == "/v1/memories:retrieve"
    assert sleeps.call_count == 2


def test_early_return_can_be_followed_by_explicit_wait(wire: Wire) -> None:
    wire.responses.extend(
        [
            httpx.Response(200, json=MEMORY | {"processingStatus": "PENDING"}),
            httpx.Response(200, json=MEMORY),
        ]
    )
    result = GoodMemCreateMemory(client=wire.sdk).invoke(
        {
            "space_id": "space-1",
            "original_content": "Evidence",
            "wait": False,
        }
    )
    assert result["processing_status"] == "PENDING" and len(wire.requests) == 1
    assert (
        wait_for_memory(wire.sdk, result["memory_id"]).processing_status == "COMPLETED"
    )


@pytest.mark.parametrize(
    "response,diagnostic",
    [
        (httpx.Response(200, json=MEMORY | {"processingStatus": "FAILED"}), "FAILED"),
        (
            httpx.Response(200, json=MEMORY | {"processingStatus": "PENDING"}),
            "exceeded",
        ),
        (httpx.Response(503, json={"message": "Unavailable"}), "Unavailable"),
    ],
)
def test_wait_failure_reports_created_id_without_retrying_creation(
    wire: Wire,
    response: httpx.Response,
    diagnostic: str,
) -> None:
    wire.responses.extend(
        [httpx.Response(200, json=MEMORY | {"processingStatus": "PENDING"}), response]
    )
    with pytest.raises(ToolException, match="Memory memory-1 was created") as error:
        GoodMemCreateMemory(client=wire.sdk).invoke(
            {
                "space_id": "space-1",
                "original_content": "Evidence",
                "indexing_timeout": 0,
            }
        )
    assert diagnostic in str(error.value)
    assert [r.method for r in wire.requests] == ["POST", "GET"]


def test_get_memory_uses_native_inline_content_then_delete(wire: Wire) -> None:
    content = base64.b64encode(b"binary\x00\xff").decode()
    wire.responses.extend(
        [
            httpx.Response(200, json=MEMORY | {"originalContent": content}),
            httpx.Response(204),
        ]
    )
    result = GoodMemGetMemory(client=wire.sdk).invoke(
        {"memory_id": "memory-1", "include_content": True}
    )
    assert (
        result["original_content"] == content
        and result["metadata"] == MEMORY["metadata"]
    )
    assert wire.requests[0].url.params["include_content"] == "true"
    assert len(wire.requests) == 1
    assert (
        GoodMemDeleteMemory(client=wire.sdk).invoke({"memory_id": "memory-1"}) is None
    )


def test_retrieval_keeps_sdk_events_even_when_summary_fails(wire: Wire) -> None:
    wire.responses.append(
        ndjson(
            CHUNK,
            {"memoryDefinition": MEMORY},
            {"status": {"code": "SUMMARIZATION_FAILED", "message": "No summary"}},
        )
    )
    events = GoodMemRetrieveMemories(client=wire.sdk).invoke(
        {
            "message": "question",
            "space_ids": ["space-1"],
            "llm_id": "llm-1",
            "reranker_id": "reranker-1",
            "requested_size": 20,
            "max_results": 3,
        }
    )
    assert (
        events[0]["retrieved_item"]["chunk"]["chunk"]["chunk_text"]
        == "Retrieved evidence"
    )
    assert events[1]["memory_definition"]["metadata"] == MEMORY["metadata"]
    assert events[2]["status"]["code"] == "SUMMARIZATION_FAILED"
    body = json.loads(wire.requests[0].content)
    assert (
        body["requestedSize"] == 20
        and body["postProcessor"]["config"]["max_results"] == 3
    )


def test_sdk_results_are_serialized_by_langchain_for_agents(wire: Wire) -> None:
    wire.responses.append(httpx.Response(200, json=SPACE))
    message = GoodMemGetSpace(client=wire.sdk).invoke(
        {
            "type": "tool_call",
            "name": "goodmem_get_space",
            "id": "call",
            "args": {"space_id": "space-1"},
        }
    )
    assert isinstance(message, ToolMessage) and message.status == "success"
    assert isinstance(message.content, str)
    assert json.loads(message.content)["space_id"] == "space-1"


def test_connection_settings_are_not_model_arguments(wire: Wire) -> None:
    for name in langchain_goodmem.__all__:
        cls = getattr(langchain_goodmem, name)
        if isinstance(cls, type) and issubclass(cls, BaseTool):
            schema = cls(client=wire.sdk).get_input_schema().model_json_schema()
            assert (
                not {"client", "goodmem_api_key", "goodmem_base_url"}
                & schema["properties"].keys()
            )
            assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        GoodMemGetSpace(client=wire.sdk).invoke(
            {"space_id": "space-1", "goodmem_base_url": "https://unexpected.test"}
        )
    assert not wire.requests


def test_file_upload_uses_sdk_multipart(wire: Wire, tmp_path: Path) -> None:
    path = tmp_path / "example.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture")
    wire.responses.append(
        httpx.Response(200, json=MEMORY | {"contentType": "application/pdf"})
    )
    result = GoodMemCreateMemory(client=wire.sdk).invoke(
        {
            "space_id": "space-1",
            "file_path": str(path),
            "metadata": {"title": "PDF evidence"},
            "wait": False,
        }
    )
    assert result["content_type"] == "application/pdf"
    request = wire.requests[0]
    assert request.headers["content-type"].startswith("multipart/form-data;")
    assert b"%PDF-1.4" in request.content and b"PDF evidence" in request.content


def test_environment_secrets_lifecycle_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://goodmem.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "synthetic-key")
    monkeypatch.setenv("GOODMEM_VERIFY_SSL", "false")
    sdk = Mock(spec=Goodmem)
    sdk.spaces = Mock()
    sdk.spaces.list.return_value = []
    sdk.__enter__ = Mock(return_value=sdk)
    sdk.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=sdk)
    monkeypatch.setattr("langchain_goodmem._connection.Goodmem", factory)
    tool = GoodMemListSpaces(goodmem_timeout=120)
    assert "synthetic-key" not in repr(tool) and "synthetic-key" not in str(
        tool.model_dump()
    )
    assert "synthetic-key" not in str(tool.to_json())
    assert tool.invoke({}) == []
    factory.assert_called_once_with(
        base_url="https://goodmem.test",
        api_key="synthetic-key",
        timeout=120,
        verify=False,
    )
    sdk.__exit__.assert_called_once()


def test_missing_configuration_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOODMEM_BASE_URL", raising=False)
    monkeypatch.delenv("GOODMEM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Provide client"):
        GoodMemRetriever(space_ids=["space-1"]).invoke("question")


def test_explicit_string_key_is_stored_as_a_secret() -> None:
    from pydantic import SecretStr

    tool = GoodMemListSpaces(
        goodmem_base_url="https://example.org", goodmem_api_key="synthetic-direct-key"
    )
    assert isinstance(tool.goodmem_api_key, SecretStr)
    assert "synthetic-direct-key" not in repr(tool)
    assert "synthetic-direct-key" not in str(tool.model_dump())
