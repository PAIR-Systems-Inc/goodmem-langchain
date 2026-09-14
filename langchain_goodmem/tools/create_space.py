"""Create a space using the GoodMem SDK's configuration and defaults."""

from typing import Any

from goodmem.models.space_embedder_config import SpaceEmbedderConfig
from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class CreateSpaceInput(ToolInput):
    """Arguments for creating a space with one embedder."""

    name: str = Field(description="Name for the new space.")
    embedder_id: str = Field(description="UUID of the embedder to use.")
    labels: dict[str, str] | None = None


class GoodMemCreateSpace(GoodMemTool):
    """Create a new space. Existing spaces must be selected explicitly."""

    name: str = "goodmem_create_space"
    description: str = (
        "Create a new GoodMem space with an embedder. Returns the created space. "
        "Use goodmem_list_spaces to find an existing space."
    )
    args_schema: type[BaseModel] = CreateSpaceInput

    def _run(
        self,
        name: str,
        embedder_id: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a space and return its SDK fields as a dictionary."""
        with self._session() as client:
            space = client.spaces.create(
                name=name,
                space_embedders=[
                    SpaceEmbedderConfig.model_validate({"embedderId": embedder_id})
                ],
                labels=labels,
            )
            return space.model_dump(mode="json", exclude_none=True)
