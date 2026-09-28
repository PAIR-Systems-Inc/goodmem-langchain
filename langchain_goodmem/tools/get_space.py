"""Get a GoodMem space by ID."""

from typing import Any

from pydantic import BaseModel, Field

from langchain_goodmem._ids import UUIDStr
from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class GetSpaceInput(ToolInput):
    """Identify the space to get."""

    space_id: UUIDStr = Field(description="UUID of the space.")


class GoodMemGetSpace(GoodMemTool):
    """Get a space using the SDK."""

    name: str = "goodmem_get_space"
    description: str = "Get a GoodMem space by UUID. Returns its SDK fields."
    args_schema: type[BaseModel] = GetSpaceInput

    def _run(self, space_id: str) -> dict[str, Any]:
        """Get the specified space."""
        space_id = self._uuid(space_id, "space_id")
        with self._session() as client:
            return client.spaces.get(id=space_id).model_dump(
                mode="json", exclude_none=True
            )
