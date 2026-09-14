"""Delete a GoodMem space by ID."""

from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class DeleteSpaceInput(ToolInput):
    """Identify the space to delete."""

    space_id: str = Field(description="UUID of the space.")


class GoodMemDeleteSpace(GoodMemTool):
    """Delete a space using the SDK."""

    name: str = "goodmem_delete_space"
    description: str = "Delete a GoodMem space by UUID. Deletion is permanent; returns null on success."
    args_schema: type[BaseModel] = DeleteSpaceInput

    def _run(self, space_id: str) -> None:
        """Delete the specified space."""
        with self._session() as client:
            client.spaces.delete(id=space_id)
