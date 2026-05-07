"""GoodMem Update Space tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_goodmem._client import GoodMemClient


class UpdateSpaceInput(BaseModel):
    """Input schema for the GoodMem Update Space tool.

    Only fields that are explicitly set will be sent to the server.
    `replace_labels` and `merge_labels` are mutually exclusive on the server,
    so callers should pass at most one.
    """

    space_id: str = Field(description="The UUID of the space to update.")
    name: str | None = Field(
        default=None,
        description="New space name. Omit to leave the name unchanged.",
    )
    public_read: bool | None = Field(
        default=None,
        description="Whether the space should be publicly readable.",
    )
    replace_labels: dict[str, str] | None = Field(
        default=None,
        description=(
            "Replace the entire label set with this map. "
            "Mutually exclusive with `merge_labels`."
        ),
    )
    merge_labels: dict[str, str] | None = Field(
        default=None,
        description=(
            "Merge these labels into the existing label set. "
            "Mutually exclusive with `replace_labels`."
        ),
    )


class GoodMemUpdateSpace(BaseTool):
    """Update mutable fields on a GoodMem space.

    Supports renaming, toggling public read access, and replacing or merging
    the label set. Only fields explicitly provided are sent in the request.

    Setup:
        Install ``langchain-goodmem`` and set environment variables:

        .. code-block:: bash

            pip install langchain-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langchain_goodmem import GoodMemUpdateSpace

            tool = GoodMemUpdateSpace(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({
                "space_id": "space-uuid",
                "name": "renamed-space",
            })
    """

    name: str = "goodmem_update_space"
    description: str = (
        "Update mutable fields on a GoodMem space (name, public_read, labels). "
        "Only fields explicitly provided are sent."
    )
    args_schema: type[BaseModel] = UpdateSpaceInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Update a space.

        Args:
            space_id: The space UUID.
            name: New space name.
            public_read: Whether the space is publicly readable.
            replace_labels: Replace the entire label set.
            merge_labels: Merge into the existing label set.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the updated space.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.update_space(
                space_id=space_id,
                name=name,
                public_read=public_read,
                replace_labels=replace_labels,
                merge_labels=merge_labels,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
