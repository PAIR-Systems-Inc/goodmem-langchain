"""Citation-ready LangChain retrieval backed by the official GoodMem SDK."""

import logging
import warnings
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
from langchain_goodmem._ids import UUIDStr, require_uuid

logger = logging.getLogger(__name__)


class GoodMemRetrievalError(RuntimeError):
    """A retrieval stream that cannot be interpreted.

    Raised for a malformed stream, such as a chunk whose memory definition
    never arrived. Server *statuses* are not errors: they are reported on the
    Documents (see :func:`documents_from_events`).
    """

    def __init__(
        self, message: str, *, statuses: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(message)
        self.statuses = statuses or []


# Notices that carry no loss of results.
#
# FEATURE_DISABLED is informational by its code alone. The server defines it as
# "feature disabled due to missing configuration" (common.proto, under
# "Informational status messages (non-error)"): the caller did not configure an
# optional feature, so nothing the caller asked for is missing. A feature that
# was requested and could not be delivered arrives as a different code
# (NOT_FOUND, RERANKING_FAILED, ...). Retrieval status contract, Q1.
_INFORMATIONAL_CODES = frozenset({"LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED"})


def _is_informational(status: GoodMemStatus) -> bool:
    """Recognize notices that do not indicate incomplete retrieval."""
    return status.code is not None and status.code in _INFORMATIONAL_CODES


def classify_statuses(events: Iterable[RetrieveMemoryEvent]) -> list[dict[str, Any]]:
    """Return the statuses that indicate a real problem, never raising.

    Known informational notices are dropped. A code this SDK does not
    recognize decodes as ``None`` and is surfaced as ``UNKNOWN`` with
    ``unrecognized: True`` -- a newer server must not silently change what
    the retriever reports (retrieval status contract, Q3).
    """
    surfaced: list[dict[str, Any]] = []
    for event in events:
        status = event.status
        if status is None or _is_informational(status):
            continue
        entry = status.model_dump(exclude_none=True)
        if status.code is None:
            entry["code"] = "UNKNOWN"
            entry["unrecognized"] = True
        surfaced.append(entry)
    return surfaced


def documents_from_events(events: Iterable[RetrieveMemoryEvent]) -> list[Document]:
    """Join chunks to memory metadata by UUID, independently of event ordering.

    Preserve server ordering and opaque scores (including negative vector scores).
    A source URL is copied from metadata or original_content_ref when available;
    memory/chunk IDs remain available when a document has no external source.

    When the server reported a real problem, every Document carries
    ``goodmem_partial=True`` and ``goodmem_statuses`` in its metadata; the
    Documents are still returned (retrieval status contract, Q4a). A problem
    with no Documents at all returns an empty list and emits a warning and a
    log line carrying the statuses, because a bare list has nowhere to carry
    the flag (Q4b). Neither case raises.
    """
    events = list(events)
    statuses = classify_statuses(events)
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
        if statuses:
            metadata["goodmem_partial"] = True
            metadata["goodmem_statuses"] = statuses
        documents.append(
            Document(
                id=chunk.chunk_id, page_content=chunk.chunk_text, metadata=metadata
            )
        )
    if statuses and not documents:
        summary = "; ".join(
            f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}" for s in statuses
        )
        warnings.warn(
            f"GoodMem retrieval returned no Documents and reported a problem: {summary}",
            stacklevel=2,
        )
        logger.warning(
            "GoodMem retrieval failed with no Documents; statuses=%s", statuses
        )
    return documents


class GoodMemRetriever(GoodMemConnection, BaseRetriever):
    """Retrieve Documents from configured spaces, with optional LLM-free reranking.

    Supports invoke/ainvoke, batch/abatch, callbacks and LCEL via BaseRetriever.
    Async calls use LangChain's thread executor with the synchronous SDK. Pass a
    caller-owned ``client`` to share a connection pool across calls. Otherwise
    each call opens and closes its own SDK client using explicit/env settings.
    """

    space_ids: list[UUIDStr] = Field(min_length=1)
    k: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(default=None, gt=0)
    reranker_id: UUIDStr | None = None
    filter: str | None = Field(
        default=None,
        description="GoodMem metadata filter expression applied to every configured space.",
    )

    @model_validator(mode="after")
    def _validate_search(self) -> "GoodMemRetriever":
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
        # Fields can be reassigned after validation; check again before sending.
        space_ids = [require_uuid(sid, "space_ids") for sid in self.space_ids]
        reranker_id = (
            None
            if self.reranker_id is None
            else require_uuid(self.reranker_id, "reranker_id")
        )
        options: dict[str, Any] = {}
        if reranker_id:
            options = dict(
                reranker_id=reranker_id,
                max_results=limit,
                chronological_resort=False,
            )
        if self.filter is None:
            options["space_ids"] = space_ids
        else:
            options["space_keys"] = [
                SpaceKey.model_validate({"spaceId": sid, "filter": self.filter})
                for sid in space_ids
            ]
        with self._session() as client:
            events = client.memories.retrieve(
                message=query,
                requested_size=self.fetch_k or (limit * 4 if reranker_id else limit),
                fetch_memory=True,
                fetch_memory_content=False,
                stream=False,
                **options,
            )
        return documents_from_events(events)[:limit]
