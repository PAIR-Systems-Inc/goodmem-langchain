"""Wait for the memory that was written, rather than guessing from search hits."""

import time
from typing import cast

from goodmem import Goodmem
from goodmem.models.memory import Memory

from langchain_goodmem._connection import GoodMemSDK


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
