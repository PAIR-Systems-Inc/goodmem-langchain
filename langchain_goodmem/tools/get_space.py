"""GoodMem Get Space tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class GetSpaceInput(BaseModel):
    """Input schema for the GoodMem Get Space tool."""

    space_id: str = Field(
        description="The UUID of the space to fetch.",
    )


class GoodMemGetSpace(BaseTool):
    """Fetch a single GoodMem space by its ID.

    Returns the space metadata including name, embedder configuration,
    labels, and access settings.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemGetSpace

            tool = GoodMemGetSpace(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({"space_id": "space-uuid"})
    """

    name: str = "goodmem_get_space"
    description: str = (
        "Fetch a single GoodMem space by its ID, returning its name, "
        "embedder configuration, labels, and access settings."
    )
    args_schema: type[BaseModel] = GetSpaceInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(self, space_id: str, **kwargs: Any) -> str:
        """Fetch a space by ID.

        Args:
            space_id: The space UUID.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the space data.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.get_space(space_id=space_id)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
