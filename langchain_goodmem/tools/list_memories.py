"""GoodMem List Memories tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class ListMemoriesInput(BaseModel):
    """Input schema for the GoodMem List Memories tool."""

    space_id: str = Field(
        description="The UUID of the space whose memories to list.",
    )
    max_results: int | None = Field(
        default=None,
        description="Maximum number of memories to return per page.",
    )
    next_token: str | None = Field(
        default=None,
        description=(
            "Opaque pagination cursor returned from a previous call. "
            "Pass to fetch the next page."
        ),
    )
    status_filter: str | None = Field(
        default=None,
        description=(
            "Restrict to memories with this processing status: "
            "`PENDING`, `PROCESSING`, `COMPLETED`, or `FAILED`."
        ),
    )
    include_content: bool = Field(
        default=False,
        description="Inline the original content for each returned memory.",
    )
    filter_expression: str | None = Field(
        default=None,
        description="Server-side metadata filter expression.",
    )


class GoodMemListMemories(BaseTool):
    """Paginate memories within a GoodMem space.

    Supports filtering by processing status and metadata, and can optionally
    inline each memory's original content.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemListMemories

            tool = GoodMemListMemories(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({
                "space_id": "space-uuid",
                "max_results": 50,
                "status_filter": "COMPLETED",
            })
    """

    name: str = "goodmem_list_memories"
    description: str = (
        "Paginate memories within a GoodMem space. Supports filtering by "
        "processing status and metadata, and can inline original content."
    )
    args_schema: type[BaseModel] = ListMemoriesInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        space_id: str,
        max_results: int | None = None,
        next_token: str | None = None,
        status_filter: str | None = None,
        include_content: bool = False,
        filter_expression: str | None = None,
        **kwargs: Any,
    ) -> str:
        """List memories in a space.

        Args:
            space_id: The space UUID.
            max_results: Maximum memories to return per page.
            next_token: Pagination cursor from a prior call.
            status_filter: Restrict by processing status.
            include_content: Inline original content for each memory.
            filter_expression: Server-side metadata filter.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the page of memories.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.list_memories(
                space_id=space_id,
                max_results=max_results,
                next_token=next_token,
                status_filter=status_filter,
                include_content=include_content,
                filter_expression=filter_expression,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
