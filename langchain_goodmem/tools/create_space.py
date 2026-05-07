"""GoodMem Create Space tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class CreateSpaceInput(BaseModel):
    """Input schema for the GoodMem Create Space tool."""

    name: str = Field(description="A unique name for the space.")
    embedder_id: str = Field(
        description=(
            "The ID of the embedder model that converts text into vector "
            "representations for similarity search."
        ),
    )
    chunking_strategy: str = Field(
        default="recursive",
        description=(
            "The chunking strategy for text processing. "
            "One of 'recursive', 'sentence', or 'none'."
        ),
    )
    chunk_size: int = Field(
        default=512,
        description="Maximum chunk size in characters (for recursive/sentence).",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between consecutive chunks in characters.",
    )


class GoodMemCreateSpace(BaseTool):
    """Create a new GoodMem space or reuse an existing one.

    A space is a logical container for organizing related memories,
    configured with embedders that convert text to vector embeddings.
    If a space with the given name already exists, its ID is returned
    instead of creating a duplicate.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemCreateSpace

            tool = GoodMemCreateSpace(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({
                "name": "my-space",
                "embedder_id": "emb-xxx",
            })
    """

    name: str = "goodmem_create_space"
    description: str = (
        "Create a new GoodMem space or reuse an existing one. "
        "A space is a logical container for organizing related memories, "
        "configured with an embedder for vector search."
    )
    args_schema: type[BaseModel] = CreateSpaceInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        name: str,
        embedder_id: str,
        chunking_strategy: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> str:
        """Create a space or return an existing one.

        Args:
            name: The space name.
            embedder_id: The embedder ID.
            chunking_strategy: The chunking strategy.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the operation result.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.create_space(
                name=name,
                embedder_id=embedder_id,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
