"""Delete a GoodMem memory by ID."""

from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class DeleteMemoryInput(ToolInput):
    """Identify the memory to delete."""

    memory_id: str = Field(description="UUID of the memory.")


class GoodMemDeleteMemory(GoodMemTool):
    """Delete a memory using the SDK."""

    name: str = "goodmem_delete_memory"
    description: str = "Delete a GoodMem memory by UUID. Deletion is permanent; returns null on success."
    args_schema: type[BaseModel] = DeleteMemoryInput

    def _run(self, memory_id: str) -> None:
        """Delete the specified memory."""
        with self._session() as client:
            client.memories.delete(id=memory_id)
