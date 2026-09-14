"""Live SDK-backed tools and RAG. Creates and deletes only its own test space.

Requires GOODMEM_BASE_URL, GOODMEM_API_KEY, GOODMEM_EMBEDDER_ID,
and GOODMEM_RERANKER_ID. No LLM is needed.
"""

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from goodmem import Goodmem
from langchain_core.tools import create_retriever_tool

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListEmbedders,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrievalError,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    GoodMemUpdateSpace,
)


def test_live_sdk_rag() -> None:
    required = [
        "GOODMEM_BASE_URL",
        "GOODMEM_API_KEY",
        "GOODMEM_EMBEDDER_ID",
        "GOODMEM_RERANKER_ID",
    ]
    if any(not os.getenv(key) for key in required):
        pytest.skip(
            "Set endpoint, credentials, embedder and reranker IDs for the live test"
        )
    report: dict[str, Any] = {}
    with Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        timeout=120,
    ) as sdk:

        def invoke(cls: Any, **args: Any) -> Any:
            result = cls(client=sdk).invoke(args)
            report[cls.__name__] = True
            return result

        embedders = invoke(GoodMemListEmbedders)
        assert any(
            e["embedder_id"] == os.environ["GOODMEM_EMBEDDER_ID"] for e in embedders
        )
        name = f"langchain-sdk-test-{uuid.uuid4()}"
        sid = invoke(
            GoodMemCreateSpace, name=name, embedder_id=os.environ["GOODMEM_EMBEDDER_ID"]
        )["space_id"]
        try:
            assert invoke(GoodMemGetSpace, space_id=sid)["name"] == name
            assert invoke(GoodMemListSpaces, name_filter=name)[0]["space_id"] == sid
            updated = invoke(
                GoodMemUpdateSpace, space_id=sid, merge_labels={"test": "langchain-sdk"}
            )
            assert updated["labels"]["test"] == "langchain-sdk"
            content = (
                "Project Cobalt's launch owner is Ada. Its launch date is October 17."
            )
            mid = str(uuid.uuid4())
            created = invoke(
                GoodMemCreateMemory,
                space_id=sid,
                memory_id=mid,
                original_content=content,
                metadata={"source": "https://example.org/cobalt", "title": "Cobalt"},
                indexing_timeout=120,
            )
            assert (
                created["memory_id"] == mid
                and created["processing_status"] == "COMPLETED"
            )
            report["default_create_waits_for_indexing"] = True
            fetched = invoke(GoodMemGetMemory, memory_id=mid, include_content=True)
            assert base64.b64decode(fetched["original_content"]).decode() == content
            assert (
                invoke(GoodMemListMemories, space_id=sid, max_items=1)[0]["memory_id"]
                == mid
            )
            for reranker in [None, os.environ["GOODMEM_RERANKER_ID"]]:
                retriever = GoodMemRetriever(
                    client=sdk, space_ids=[sid], reranker_id=reranker
                )
                docs = retriever.invoke("Who owns Project Cobalt's launch?")
                assert docs and "Ada" in docs[0].page_content
                assert docs[0].metadata["source"] == "https://example.org/cobalt"
                tool = create_retriever_tool(
                    retriever,
                    "project_search",
                    "Search project records",
                    response_format="content_and_artifact",
                )
                result = tool.invoke(
                    {
                        "type": "tool_call",
                        "id": "call",
                        "name": "project_search",
                        "args": {"query": "Cobalt launch owner"},
                    }
                )
                assert result.artifact[0].metadata["memory_id"] == mid
            report["plain_and_reranked_document_tools_without_llm"] = True
            events = invoke(
                GoodMemRetrieveMemories,
                message="Cobalt launch owner",
                space_ids=[sid],
                requested_size=20,
                max_results=5,
                reranker_id=os.environ["GOODMEM_RERANKER_ID"],
            )
            assert any("retrieved_item" in event for event in events)
            # Invalid optional resources exercise server diagnostics without an LLM call.
            for resource, code in [
                ("llm_id", "SUMMARIZATION_FAILED"),
                ("reranker_id", "RERANKING_FAILED"),
            ]:
                missing = str(uuid.uuid4())
                events = invoke(
                    GoodMemRetrieveMemories,
                    message="Cobalt launch owner",
                    space_ids=[sid],
                    **{resource: missing},
                )
                assert any(
                    event.get("status", {}).get("code") == code for event in events
                )
                assert any("retrieved_item" in event for event in events)
                if resource == "reranker_id":
                    with pytest.raises(GoodMemRetrievalError, match="RERANKING_FAILED"):
                        GoodMemRetriever(
                            client=sdk, space_ids=[sid], reranker_id=missing
                        ).invoke("Cobalt owner")
            report["sdk_tool_preserves_statuses_and_fallback_chunks"] = True
            report["document_retriever_reports_reranker_failure"] = True
            invoke(GoodMemDeleteMemory, memory_id=mid)
            assert (
                GoodMemRetriever(client=sdk, space_ids=[sid]).invoke("Cobalt owner")
                == []
            )
            report["empty_space_returns_empty"] = True
        finally:
            invoke(GoodMemDeleteSpace, space_id=sid)
    report["temporary_resources_deleted"] = True
    if path := os.getenv("GOODMEM_LIVE_REPORT"):
        Path(path).write_text(json.dumps(report, indent=2) + "\n")
