"""Wait for the memory that was written, rather than guessing from search hits."""

import time
from collections.abc import Iterable
from typing import cast

from goodmem import Goodmem, MemoryCreationRequest
from goodmem.models.memory import Memory
from langchain_core.documents import Document

from langchain_goodmem._connection import GoodMemSDK


class GoodMemIngestionError(RuntimeError):
    """A batch write or indexing wait failed; retain IDs known to be created."""

    def __init__(self, message: str, *, created_memory_ids: list[str]) -> None:
        super().__init__(message)
        self.created_memory_ids = created_memory_ids


def add_documents(
    client: Goodmem | GoodMemSDK,
    space_id: str,
    documents: Iterable[Document],
    *,
    wait: bool = True,
    indexing_timeout: float = 60,
) -> list[str]:
    """Write LangChain Documents using the SDK batch API.

    Args:
        client: Caller-owned GoodMem SDK client.
        space_id: Destination space UUID; its chunking and embedder settings apply.
        documents: Text and metadata to store. Optional Document IDs become memory
            IDs and must be UUIDs. Existing IDs produce conflicts, not updates.
        wait: Wait for each created memory to finish indexing before returning.
        indexing_timeout: Maximum polling time per memory, in seconds.

    Returns:
        Created memory IDs in input order. Retrieval returns chunks of these memories.

    Raises:
        GoodMemIngestionError: A per-document write or indexing wait failed.
            created_memory_ids identifies successful writes; they are not rolled back.
        ValueError: The indexing timeout is negative.
    """
    if indexing_timeout < 0:
        raise ValueError("indexing_timeout must be nonnegative")
    requests = [
        MemoryCreationRequest.model_validate(
            {
                "space_id": space_id,
                "original_content": document.page_content,
                "metadata": document.metadata,
                **({"memory_id": document.id} if document.id is not None else {}),
            }
        )
        for document in documents
    ]
    if not requests:
        return []
    results = cast(GoodMemSDK, client).memories.batch_create(requests=requests).results
    memory_ids = [
        result.memory.memory_id
        for result in results
        if result.success and result.memory is not None
    ]
    if len(memory_ids) != len(requests):
        errors = "; ".join(
            result.error.message for result in results if result.error is not None
        )
        raise GoodMemIngestionError(
            f"Document ingestion was incomplete: {errors or 'missing creation results'}",
            created_memory_ids=memory_ids,
        )
    if wait:
        try:
            for memory_id in memory_ids:
                wait_for_memory(client, memory_id, indexing_timeout)
        except Exception as exc:
            raise GoodMemIngestionError(
                str(exc), created_memory_ids=memory_ids
            ) from exc
    return memory_ids


def wait_for_memory(
    client: Goodmem | GoodMemSDK,
    memory_id: str,
    timeout: float = 60,
    *,
    poll_interval: float = 1,
) -> Memory:
    """Poll processing status; propagate failed, unknown, and timed-out ingestion.

    ``timeout`` limits polling; an in-flight request is bounded separately by
    the SDK client's HTTP timeout. The caller retains ownership of the client.
    """
    if timeout < 0 or poll_interval <= 0:
        raise ValueError(
            "timeout must be nonnegative and poll_interval must be positive"
        )
    deadline = time.monotonic() + timeout
    while True:
        memory = cast(GoodMemSDK, client).memories.get(id=memory_id)
        if memory.processing_status == "COMPLETED":
            return memory
        if memory.processing_status not in {"PENDING", "PROCESSING"}:
            raise RuntimeError(
                f"Memory {memory_id}: indexing {memory.processing_status}. "
                "Inspect it with include_processing_history=True."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Memory {memory_id}: indexing exceeded {timeout}s")
        time.sleep(min(poll_interval, remaining))
