"""Citation-ready LangChain retrieval backed by the official GoodMem SDK."""

from collections.abc import Iterable
from typing import Any

from goodmem.models.good_mem_status import GoodMemStatus
from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent
from goodmem.models.space_key import SpaceKey
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from langchain_goodmem._connection import GoodMemConnection


class GoodMemRetrievalError(RuntimeError):
    """A failed/partial retrieval must not look like an empty or successful search."""

    def __init__(
        self, message: str, *, statuses: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(message)
        self.statuses = statuses or []


def _is_informational(status: GoodMemStatus) -> bool:
    """Recognize notices that do not indicate incomplete retrieval."""
    details = status.details or {}
    return status.code == "LLM_CAPABILITY_INFERRED" or (
        status.code == "FEATURE_DISABLED"
        and details.get("feature") == "summarization"
        and details.get("required_param") == "llm_id"
    )


def checked_events(events: Iterable[RetrieveMemoryEvent]) -> list[RetrieveMemoryEvent]:
    """Apply the Document retriever's strict policy to non-informational statuses."""
    events = list(events)
    failures = [
        event.status.model_dump(exclude_none=True)
        for event in events
        if event.status is not None and not _is_informational(event.status)
    ]
    if failures:
        raise GoodMemRetrievalError(
            "; ".join(f"{s.get('code', 'UNKNOWN')}: {s['message']}" for s in failures),
            statuses=failures,
        )
    return events


def documents_from_events(events: Iterable[RetrieveMemoryEvent]) -> list[Document]:
    """Join chunks to memory metadata by UUID, independently of event ordering.

    Preserve server ordering and opaque scores (including negative vector scores).
    A source URL is copied from metadata or original_content_ref when available;
    memory/chunk IDs remain available when a document has no external source.
    """
    events = checked_events(events)
    memories = {
        event.memory_definition.memory_id: event.memory_definition
        for event in events
        if event.memory_definition is not None
    }
    documents = []
    seen = set()
    for event in events:
        item = event.retrieved_item
        if item is None or item.chunk is None:
            continue
        hit, chunk = item.chunk, item.chunk.chunk
        if chunk.chunk_id in seen or not chunk.chunk_text:
            continue
        memory = memories.get(chunk.memory_id)
        if memory is None:
            raise GoodMemRetrievalError(
                f"Missing memory metadata for chunk {chunk.chunk_id}"
            )
        seen.add(chunk.chunk_id)
        metadata = dict(memory.metadata or {})
        metadata.setdefault("source", memory.original_content_ref or memory.memory_id)
        metadata.setdefault("title", "")
        metadata.update(
            memory_id=memory.memory_id,
            chunk_id=chunk.chunk_id,
            space_id=memory.space_id,
            score=hit.relevance_score,
        )
        documents.append(
            Document(
                id=chunk.chunk_id, page_content=chunk.chunk_text, metadata=metadata
            )
        )
    return documents


class GoodMemRetriever(GoodMemConnection, BaseRetriever):
    """Retrieve Documents from configured spaces, with optional LLM-free reranking.

    Supports invoke/ainvoke, batch/abatch, callbacks and LCEL via BaseRetriever.
    Async calls use LangChain's thread executor with the synchronous SDK. Pass a
    caller-owned ``client`` to share a connection pool across calls. Otherwise
    each call opens and closes its own SDK client using explicit/env settings.
    """

    space_ids: list[str] = Field(min_length=1)
    k: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(default=None, gt=0)
    reranker_id: str | None = None
    filter: str | None = Field(
        default=None,
        description="GoodMem metadata filter expression applied to every configured space.",
    )

    @model_validator(mode="after")
    def _validate_search(self) -> "GoodMemRetriever":
        if any(not sid.strip() for sid in self.space_ids):
            raise ValueError("space_ids must not contain empty IDs")
        if self.fetch_k is not None and self.fetch_k < self.k:
            raise ValueError("fetch_k must be at least k")
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        if not query.strip():
            raise ValueError("Search query must not be empty")
        limit = self.k if k is None else k
        if limit <= 0:
            raise ValueError("k must be positive")
        if self.fetch_k is not None and self.fetch_k < limit:
            raise ValueError("fetch_k must be at least k")
        options: dict[str, Any] = {}
        if self.reranker_id:
            options = dict(
                reranker_id=self.reranker_id,
                max_results=limit,
                chronological_resort=False,
            )
        if self.filter is None:
            options["space_ids"] = self.space_ids
        else:
            options["space_keys"] = [
                SpaceKey.model_validate({"spaceId": sid, "filter": self.filter})
                for sid in self.space_ids
            ]
        with self._session() as client:
            events = client.memories.retrieve(
                message=query,
                requested_size=self.fetch_k
                or (limit * 4 if self.reranker_id else limit),
                fetch_memory=True,
                fetch_memory_content=False,
                stream=False,
                **options,
            )
        return documents_from_events(events)[:limit]
