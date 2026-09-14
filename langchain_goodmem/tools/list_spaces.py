"""List spaces using SDK pagination."""

from typing import Any

from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class ListSpacesInput(ToolInput):
    """Filter spaces and bound the number returned."""

    name_filter: str | None = Field(
        default=None, description="Name filter supporting * and ? wildcards."
    )
    label: dict[str, str] | None = None
    max_items: int | None = Field(
        default=100, gt=0, description="Maximum spaces to return; null for all."
    )


class GoodMemListSpaces(GoodMemTool):
    """List matching spaces, following SDK pages up to max_items."""

    name: str = "goodmem_list_spaces"
    description: str = "List accessible GoodMem spaces, optionally filtered by name or labels. Returns a list of spaces."
    args_schema: type[BaseModel] = ListSpacesInput

    def _run(
        self,
        name_filter: str | None = None,
        label: dict[str, str] | None = None,
        max_items: int | None = 100,
    ) -> list[dict[str, Any]]:
        """Return SDK space dictionaries within the requested limit."""
        with self._session() as client:
            return [
                space.model_dump(mode="json", exclude_none=True)
                for space in client.spaces.list(
                    name_filter=name_filter, label=label, max_items=max_items
                )
            ]
