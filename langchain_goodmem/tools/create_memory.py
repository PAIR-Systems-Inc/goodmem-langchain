"""GoodMem Create Memory tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class CreateMemoryInput(BaseModel):
    """Input schema for the GoodMem Create Memory tool."""

    space_id: str = Field(
        description="The UUID of the space to store the memory in.",
    )
    text_content: str | None = Field(
        default=None,
        description=(
            "Plain text content to store as memory. "
            "If both file_path and text_content are provided, "
            "the file takes priority."
        ),
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Local file path to upload as memory (PDF, DOCX, image, etc.). "
            "Content type is auto-detected from the file extension."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value metadata as a dictionary.",
    )


class GoodMemCreateMemory(BaseTool):
    """Store a document as a new memory in a GoodMem space.

    The memory is processed asynchronously: chunked into searchable pieces
    and embedded into vectors. Accepts a local file path or plain text.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemCreateMemory

            tool = GoodMemCreateMemory(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation with text:
        .. code-block:: python

            result = tool.invoke({
                "space_id": "space-uuid",
                "text_content": "Some important information.",
            })

    Invocation with file:
        .. code-block:: python

            result = tool.invoke({
                "space_id": "space-uuid",
                "file_path": "/path/to/document.pdf",
            })
    """

    name: str = "goodmem_create_memory"
    description: str = (
        "Store a document as a new memory in a GoodMem space. "
        "Accepts a local file path or plain text. "
        "The memory is chunked and embedded asynchronously."
    )
    args_schema: type[BaseModel] = CreateMemoryInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        space_id: str,
        text_content: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a memory in the specified space.

        Args:
            space_id: The target space UUID.
            text_content: Plain text content.
            file_path: Path to a local file.
            metadata: Optional metadata dictionary.
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
            result = client.create_memory(
                space_id=space_id,
                text_content=text_content,
                file_path=file_path,
                metadata=metadata,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
