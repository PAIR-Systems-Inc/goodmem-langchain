"""GoodMem List Spaces tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class ListSpacesInput(BaseModel):
    """Input schema for the GoodMem List Spaces tool.

    This tool takes no user-facing inputs; the schema exists for
    framework compatibility.
    """


class GoodMemListSpaces(BaseTool):
    """List all GoodMem spaces in your account.

    Returns each space with its ID, name, labels, embedder configuration,
    and access settings.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemListSpaces

            tool = GoodMemListSpaces(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({})
    """

    name: str = "goodmem_list_spaces"
    description: str = (
        "List all GoodMem spaces. Returns each space with its ID, name, "
        "embedder configuration, and access settings."
    )
    args_schema: type[BaseModel] = ListSpacesInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(self, **kwargs: Any) -> str:
        """List all spaces.

        Args:
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the list of spaces.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            spaces = client.list_spaces()
            result: dict[str, Any] = {
                "success": True,
                "spaces": spaces,
                "totalSpaces": len(spaces),
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
