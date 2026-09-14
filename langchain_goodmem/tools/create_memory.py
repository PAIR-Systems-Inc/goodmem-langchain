"""Create a memory and optionally wait for its indexing to finish."""

from typing import Any

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from langchain_goodmem.ingestion import wait_for_memory
from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class CreateMemoryInput(ToolInput):
    """Supply exactly one of original_content or file_path."""

    space_id: str = Field(description="UUID of the space to store the memory in.")
    original_content: str | None = Field(default=None, description="Text to store.")
    file_path: str | None = Field(default=None, description="Local file to upload.")
    content_type: str | None = None
    original_content_ref: str | None = Field(
        default=None, description="Source URI for citations; does not download content."
    )
    metadata: dict[str, Any] | None = None
    memory_id: str | None = Field(
        default=None, description="Optional client-assigned UUID."
    )
    wait: bool = Field(
        default=True, description="Wait for this memory to finish indexing."
    )
    indexing_timeout: float = Field(
        default=60, ge=0, description="Polling timeout in seconds."
    )


class GoodMemCreateMemory(GoodMemTool):
    """Store text or a file, waiting for indexing by default."""

    name: str = "goodmem_create_memory"
    description: str = (
        "Create a memory from exactly one of original_content or file_path. "
        "Waits for indexing by default. Returns the memory's SDK fields. "
        "If waiting fails, the error includes the created memory ID; check it "
        "before creating another memory."
    )
    args_schema: type[BaseModel] = CreateMemoryInput

    def _run(
        self,
        space_id: str,
        original_content: str | None = None,
        file_path: str | None = None,
        content_type: str | None = None,
        original_content_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        wait: bool = True,
        indexing_timeout: float = 60,
    ) -> dict[str, Any]:
        """Create a memory and return its fields, retaining its ID on wait failures."""
        content: dict[str, Any] = {
            "original_content": original_content,
            "file_path": file_path,
        }
        with self._session() as client:
            memory = client.memories.create(
                space_id=space_id,
                content_type=content_type,
                original_content_ref=original_content_ref,
                metadata=metadata,
                memory_id=memory_id,
                **content,
            )
            if wait:
                try:
                    memory = wait_for_memory(client, memory.memory_id, indexing_timeout)
                except Exception as exc:
                    raise ToolException(
                        f"Memory {memory.memory_id} was created, but waiting for indexing failed: {exc}"
                    ) from exc
            return memory.model_dump(mode="json", exclude_none=True)
