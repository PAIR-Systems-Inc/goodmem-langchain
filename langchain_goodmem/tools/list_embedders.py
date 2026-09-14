"""List available GoodMem embedders."""

from typing import Any

from pydantic import BaseModel

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class GoodMemListEmbedders(GoodMemTool):
    """List embedders using the SDK."""

    name: str = "goodmem_list_embedders"
    description: str = (
        "List available GoodMem embedders and their SDK configuration fields."
    )
    args_schema: type[BaseModel] = ToolInput

    def _run(self) -> list[dict[str, Any]]:
        """Return a list of SDK embedder dictionaries."""
        with self._session() as client:
            return [
                embedder.model_dump(mode="json", exclude_none=True)
                for embedder in client.embedders.list()
            ]
