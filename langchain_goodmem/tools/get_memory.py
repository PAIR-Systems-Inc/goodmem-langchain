"""GoodMem Get Memory tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class GetMemoryInput(BaseModel):
    """Input schema for the GoodMem Get Memory tool."""

    memory_id: str = Field(
        description="The UUID of the memory to fetch.",
    )
    include_content: bool = Field(
        default=True,
        description=(
            "Fetch the original document content of the memory in addition "
            "to its metadata."
        ),
    )


class GoodMemGetMemory(BaseTool):
    """Fetch a specific GoodMem memory by its ID.

    Returns the memory metadata, processing status, and optionally the
    original document content.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemGetMemory

            tool = GoodMemGetMemory(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({
                "memory_id": "memory-uuid",
                "include_content": True,
            })
    """

    name: str = "goodmem_get_memory"
    description: str = (
        "Fetch a specific GoodMem memory by its ID, including metadata, "
        "processing status, and optionally the original content."
    )
    args_schema: type[BaseModel] = GetMemoryInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        memory_id: str,
        include_content: bool = True,
        **kwargs: Any,
    ) -> str:
        """Fetch a memory by ID.

        Args:
            memory_id: The memory UUID.
            include_content: Whether to fetch the original content.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the memory data.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.get_memory(
                memory_id=memory_id,
                include_content=include_content,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
