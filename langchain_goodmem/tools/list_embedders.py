"""GoodMem List Embedders tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class ListEmbeddersInput(BaseModel):
    """Input schema for the GoodMem List Embedders tool.

    This tool takes no user-facing inputs; the schema exists for
    framework compatibility.
    """


class GoodMemListEmbedders(BaseTool):
    """List all available GoodMem embedder models.

    Embedders convert text into vector representations used for similarity
    search. Use the returned embedder ID when creating a new space.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemListEmbedders

            tool = GoodMemListEmbedders(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({})
    """

    name: str = "goodmem_list_embedders"
    description: str = (
        "List all available GoodMem embedder models. "
        "Use the returned embedder ID when creating a new space."
    )
    args_schema: type[BaseModel] = ListEmbeddersInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(self, **kwargs: Any) -> str:
        """List all embedders.

        Args:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the list of embedders.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            embedders = client.list_embedders()
            result: dict[str, Any] = {
                "success": True,
                "embedders": embedders,
                "totalEmbedders": len(embedders),
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
