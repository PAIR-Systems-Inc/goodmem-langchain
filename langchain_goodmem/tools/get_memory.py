"""Fetch a memory and optionally its content and processing history."""

from typing import Any

from pydantic import BaseModel, Field

from langchain_goodmem._ids import UUIDStr
from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class GetMemoryInput(ToolInput):
    """Select a memory and optional SDK response fields."""

    memory_id: UUIDStr = Field(description="UUID of the memory.")
    include_content: bool = Field(
        default=False, description="Include base64-encoded original content."
    )
    include_processing_history: bool = False


class GoodMemGetMemory(GoodMemTool):
    """Fetch a memory in one SDK request."""

    name: str = "goodmem_get_memory"
    description: str = (
        "Get a GoodMem memory by UUID. Returns SDK fields including metadata "
        "and processing_status. Optionally include original content or processing history."
    )
    args_schema: type[BaseModel] = GetMemoryInput

    def _run(
        self,
        memory_id: str,
        include_content: bool = False,
        include_processing_history: bool = False,
    ) -> dict[str, Any]:
        """Return the memory using the SDK's native content representation."""
        memory_id = self._uuid(memory_id, "memory_id")
        with self._session() as client:
            return client.memories.get(
                id=memory_id,
                include_content=include_content,
                include_processing_history=include_processing_history,
            ).model_dump(mode="json", exclude_none=True)
