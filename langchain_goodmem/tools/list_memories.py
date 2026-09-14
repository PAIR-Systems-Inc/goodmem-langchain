"""List memories using SDK pagination."""

from typing import Any

from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class ListMemoriesInput(ToolInput):
    """Select a space and bound the number of memories returned."""

    space_id: str = Field(description="UUID of the space.")
    max_items: int | None = Field(
        default=100, gt=0, description="Maximum memories to return; null for all."
    )


class GoodMemListMemories(GoodMemTool):
    """List a space's memories, following SDK pages up to max_items."""

    name: str = "goodmem_list_memories"
    description: str = "List memories in a GoodMem space. Returns SDK memory fields including metadata and processing_status."
    args_schema: type[BaseModel] = ListMemoriesInput

    def _run(self, space_id: str, max_items: int | None = 100) -> list[dict[str, Any]]:
        """Return SDK memory dictionaries within the requested limit."""
        with self._session() as client:
            return [
                memory.model_dump(mode="json", exclude_none=True)
                for memory in client.memories.list(
                    space_id=space_id, max_items=max_items
                )
            ]
