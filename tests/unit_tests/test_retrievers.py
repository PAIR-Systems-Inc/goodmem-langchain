"""Document retrieval and LangChain interoperability over the real SDK."""

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from goodmem.errors import GoodMemError
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, create_retriever_tool
from pydantic import ValidationError

from langchain_goodmem import (
    GoodMemRetrievalError,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
)
from tests.unit_tests.conftest import CHUNK, MEMORY, Wire, ndjson


def test_citations_join_by_uuid_and_preserve_metadata_and_scores(wire: Wire) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}, CHUNK))
    docs = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")
    assert len(docs) == 1
    assert docs[0].id == "chunk-1"
    assert docs[0].metadata == MEMORY["metadata"] | dict(
        memory_id="memory-1", chunk_id="chunk-1", space_id="space-1", score=-0.82
    )
    body = json.loads(wire.requests[0].content)
    assert body["fetchMemory"] is True and body["fetchMemoryContent"] is False
    assert "postProcessor" not in body


def test_missing_definition_is_an_error(wire: Wire) -> None:
    wire.responses.append(ndjson(CHUNK))
    with pytest.raises(GoodMemRetrievalError, match="Missing memory metadata"):
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")


def test_original_content_ref_fallback_and_k_cap(wire: Wire) -> None:
    other = json.loads(json.dumps(CHUNK))
    other["retrievedItem"]["chunk"]["chunk"]["chunkId"] = "chunk-2"
    memory = MEMORY | {
        "metadata": {},
        "originalContentRef": "https://example.org/fallback",
    }
    wire.responses.append(ndjson(CHUNK, other, {"memoryDefinition": memory}))
    docs = GoodMemRetriever(
        client=wire.sdk, space_ids=["space-1"], k=1, fetch_k=2
    ).invoke("question")
    assert (
        len(docs) == 1 and docs[0].metadata["source"] == "https://example.org/fallback"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(space_ids=[]),
        dict(space_ids=[" "]),
        dict(space_ids=["space"], k=0),
        dict(space_ids=["space"], fetch_k=1, k=5),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GoodMemRetriever(**kwargs)


def test_empty_query_never_connects(wire: Wire) -> None:
    with pytest.raises(ValueError, match="empty"):
        GoodMemRetriever(client=wire.sdk, space_ids=["space"]).invoke(" ")
    assert not wire.requests


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("start")

    def on_retriever_end(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("end")

    def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("error")


async def test_async_callbacks_and_lcel(wire: Wire) -> None:
    wire.responses.extend(
        [
            ndjson(CHUNK, {"memoryDefinition": MEMORY}),
            ndjson({"status": dict(code="EMBEDDER_FAILED", message="Failure")}),
        ]
    )
    recorder = Recorder()
    retriever = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"])
    chain = retriever | (lambda docs: docs[0].page_content)
    assert (
        await chain.ainvoke("question", config={"callbacks": [recorder]})
        == "Retrieved evidence"
    )
    # Contract Q4b: a failed search is empty and flagged, not raised.
    with pytest.warns(UserWarning, match="EMBEDDER_FAILED"):
        assert (
            await retriever.ainvoke("question", config={"callbacks": [recorder]}) == []
        )
    assert recorder.events == ["start", "end", "start", "end"]


def test_reranking_and_standard_query_only_tool(wire: Wire) -> None:
    wire.responses.append(
        ndjson(
            {
                "status": {
                    "code": "FEATURE_DISABLED",
                    "message": "No summary",
                    "details": {"feature": "summarization", "required_param": "llm_id"},
                }
            },
            CHUNK,
            {"memoryDefinition": MEMORY},
        )
    )
    retriever = GoodMemRetriever(
        client=wire.sdk,
        space_ids=["space-1"],
        reranker_id="reranker-1",
        fetch_k=20,
        filter="CAST(val('$.team') AS TEXT) = 'blue'",
    )
    tool = create_retriever_tool(
        retriever, "docs", "Search docs", response_format="content_and_artifact"
    )
    assert set(tool.get_input_schema().model_fields) == {"query"}
    result = tool.invoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "docs",
            "args": {
                "query": "question",
                "space_ids": ["untrusted-space"],
                "filter": "TRUE",
            },
        }
    )
    assert isinstance(result, ToolMessage)
    assert result.content == "Retrieved evidence"
    assert result.artifact[0].metadata["source"] == "https://example.org/docs"
    request = json.loads(wire.requests[0].content)
    assert request["spaceKeys"] == [
        {
            "spaceId": "space-1",
            "filter": "CAST(val('$.team') AS TEXT) = 'blue'",
        }
    ]
    assert request["requestedSize"] == 20
    assert request["postProcessor"]["config"] == {
        "reranker_id": "reranker-1",
        "max_results": 5,
        "chronological_resort": False,
    }


@pytest.mark.parametrize("code", ["RERANKING_FAILED", "VECTOR_SEARCH_PARTIAL"])
def test_retriever_surfaces_incomplete_search(wire: Wire, code: str) -> None:
    wire.responses.append(
        ndjson(
            CHUNK,
            {"memoryDefinition": MEMORY},
            {"status": {"code": code, "message": "Diagnostic"}},
        )
    )
    # Contract Q4a: the Documents the server returned are kept and flagged.
    docs = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")
    assert [doc.page_content for doc in docs] == ["Retrieved evidence"]
    assert docs[0].metadata["goodmem_partial"] is True
    assert docs[0].metadata["goodmem_statuses"] == [
        {"code": code, "message": "Diagnostic"}
    ]


@pytest.mark.parametrize("use_async", [False, True])
async def test_future_status_preserves_documents_and_callbacks(
    wire: Wire, use_async: bool
) -> None:
    after = deepcopy(CHUNK)
    after["retrievedItem"]["chunk"]["chunk"].update(
        chunkId="chunk-2", chunkText="Evidence after the unfamiliar status"
    )
    wire.responses.append(
        ndjson(
            {"memoryDefinition": MEMORY},
            CHUNK,
            {
                "status": {
                    "code": "TEST_ONLY_FUTURE_RETRIEVAL_STATUS_9E4AD2",
                    "message": "Future server notice",
                    "details": {"feature": "future-feature"},
                }
            },
            after,
        )
    )
    recorder = Recorder()
    retriever = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"])
    config: RunnableConfig = {"callbacks": [recorder]}
    docs = (
        await retriever.ainvoke("question", config=config)
        if use_async
        else retriever.invoke("question", config=config)
    )
    assert [doc.page_content for doc in docs] == [
        "Retrieved evidence",
        "Evidence after the unfamiliar status",
    ]
    assert [doc.id for doc in docs] == ["chunk-1", "chunk-2"]
    assert all(doc.metadata["source"] == MEMORY["metadata"]["source"] for doc in docs)
    assert recorder.events == ["start", "end"]
    # Contract Q3: surfaced as UNKNOWN and flagged, never dropped, never raised.
    assert all(doc.metadata["goodmem_partial"] is True for doc in docs)
    status = docs[0].metadata["goodmem_statuses"][0]
    assert status["code"] == "UNKNOWN" and status["unrecognized"] is True
    assert status["message"] == "Future server notice"


def test_feature_disabled_is_informational_whatever_its_details(wire: Wire) -> None:
    """Contract Q1: the code alone decides; details are not inspected."""
    wire.responses.append(
        ndjson(
            {
                "status": {
                    "code": "FEATURE_DISABLED",
                    "message": "Reranking disabled: no reranker configured.",
                    "details": {
                        "feature": "reranking",
                        "required_param": "reranker_id",
                    },
                }
            },
            CHUNK,
            {"memoryDefinition": MEMORY},
        )
    )
    docs = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")
    assert len(docs) == 1
    assert "goodmem_partial" not in docs[0].metadata
    assert "goodmem_statuses" not in docs[0].metadata


def test_clean_retrieval_metadata_is_unchanged(wire: Wire) -> None:
    """The flag keys appear only on degraded retrievals, so a clean result's
    metadata is exactly what 0.2.1 produced."""
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    docs = GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")
    assert not {"goodmem_partial", "goodmem_statuses"} & docs[0].metadata.keys()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="{broken"),
        httpx.Response(401, json={"message": "Unauthorized"}),
    ],
)
def test_protocol_and_auth_errors_propagate(
    wire: Wire, response: httpx.Response
) -> None:
    wire.responses.extend([response, response])
    with pytest.raises(GoodMemError):
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("question")
    with pytest.raises(ToolException):
        GoodMemRetrieveMemories(client=wire.sdk).invoke(
            {"message": "question", "space_ids": ["space-1"]}
        )


def test_empty_results_are_immediate_and_caller_client_stays_open(wire: Wire) -> None:
    wire.responses.extend([ndjson(), ndjson()])
    assert (
        GoodMemRetriever(client=wire.sdk, space_ids=["space-1"]).invoke("absent") == []
    )
    assert (
        GoodMemRetrieveMemories(client=wire.sdk).invoke(
            {"message": "absent", "space_ids": ["space-1"]}
        )
        == []
    )
    assert len(wire.requests) == 2 and not wire.http.is_closed


@pytest.mark.parametrize("reranker", [None, "reranker-1"])
def test_metadata_filter_is_sent_to_each_space_before_retrieval(
    wire: Wire, reranker: str | None
) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    predicate = "CAST(val('$.team') AS TEXT) = 'blue'"
    retriever = GoodMemRetriever(
        client=wire.sdk,
        space_ids=["space-1", "space-2"],
        filter=predicate,
        reranker_id=reranker,
    )
    assert retriever.invoke("question", k=1)
    request = json.loads(wire.requests[0].content)
    assert request["spaceKeys"] == [
        {"spaceId": sid, "filter": predicate} for sid in ["space-1", "space-2"]
    ]
    assert request["requestedSize"] == (4 if reranker else 1)
    if reranker:
        assert request["postProcessor"]["config"]["max_results"] == 1
