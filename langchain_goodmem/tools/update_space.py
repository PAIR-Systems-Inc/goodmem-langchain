"""Update a space's name or labels."""

from typing import Any

from pydantic import BaseModel, Field

from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class UpdateSpaceInput(ToolInput):
    """Mutable space fields. Replace and merge labels are mutually exclusive."""

    space_id: str = Field(description="UUID of the space to update.")
    name: str | None = None
    replace_labels: dict[str, str] | None = None
    merge_labels: dict[str, str] | None = None


class GoodMemUpdateSpace(GoodMemTool):
    """Rename a space or replace/merge its labels."""

    name: str = "goodmem_update_space"
    description: str = (
        "Update a GoodMem space's name or labels. Omitted fields are unchanged. "
        "Use at most one of replace_labels and merge_labels. Returns the updated space."
    )
    args_schema: type[BaseModel] = UpdateSpaceInput

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Update supplied fields and return the space's SDK fields."""
        request = {
            key: value
            for key, value in {
                "name": name,
                "replaceLabels": replace_labels,
                "mergeLabels": merge_labels,
            }.items()
            if value is not None
        }
        with self._session() as client:
            return client.spaces.update(id=space_id, request=request).model_dump(
                mode="json", exclude_none=True
            )
