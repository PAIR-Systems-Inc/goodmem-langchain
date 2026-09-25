"""Create a memory and optionally wait for its indexing to finish."""

from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, model_validator

from langchain_goodmem._ids import UUIDStr
from langchain_goodmem._uploads import GoodMemUploadError, resolve_upload_path
from langchain_goodmem.ingestion import wait_for_memory
from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class CreateMemoryInput(ToolInput):
    """Store text. File uploads are offered only when upload_dir is configured."""

    space_id: UUIDStr = Field(description="UUID of the space to store the memory in.")
    original_content: str | None = Field(default=None, description="Text to store.")
    content_type: str | None = None
    original_content_ref: str | None = Field(
        default=None, description="Source URI for citations; does not download content."
    )
    metadata: dict[str, Any] | None = None
    memory_id: UUIDStr | None = Field(
        default=None, description="Optional client-assigned UUID."
    )
    wait: bool = Field(
        default=True, description="Wait for this memory to finish indexing."
    )
    indexing_timeout: float = Field(
        default=60, ge=0, description="Polling timeout in seconds."
    )


class CreateMemoryFileInput(CreateMemoryInput):
    """Supply exactly one of original_content or file_path."""

    file_path: str | None = Field(
        default=None,
        description="Name of a file inside the approved upload directory.",
    )


_TEXT_DESCRIPTION = (
    "Create a memory from original_content. "
    "Waits for indexing by default. Returns the memory's SDK fields. "
    "If waiting fails, the error includes the created memory ID; check it "
    "before creating another memory."
)
_FILE_DESCRIPTION = (
    "Create a memory from exactly one of original_content or file_path, a file "
    "in the approved upload directory. "
    "Waits for indexing by default. Returns the memory's SDK fields. "
    "If waiting fails, the error includes the created memory ID; check it "
    "before creating another memory."
)


class GoodMemCreateMemory(GoodMemTool):
    """Store text, or a file from ``upload_dir``, waiting for indexing by default.

    ``file_path`` is chosen by the model, so it is offered only when the
    developer sets ``upload_dir``. Without it the model's schema has no
    ``file_path`` and ``_run`` refuses one. With it, the path is resolved
    (symlinks included) and refused unless it is a regular file inside
    ``upload_dir``, before anything is opened or sent. Developers can still
    upload any file through the SDK: ``client.memories.create(file_path=...)``.
    """

    name: str = "goodmem_create_memory"
    description: str = _TEXT_DESCRIPTION
    args_schema: type[BaseModel] = CreateMemoryInput
    upload_dir: str | Path | None = Field(
        default=None,
        description=(
            "Directory whose files the model may upload. Unset (the default) "
            "disables file uploads and hides file_path from the model."
        ),
    )

    @model_validator(mode="after")
    def _offer_file_path(self) -> "GoodMemCreateMemory":
        """Expose file_path to the model only when uploads are confined."""
        if self.upload_dir is not None:
            if self.args_schema is CreateMemoryInput:
                self.args_schema = CreateMemoryFileInput
            if self.description == _TEXT_DESCRIPTION:
                self.description = _FILE_DESCRIPTION
        return self

    def _run(
        self,
        space_id: str,
        original_content: str | None = None,
        file_path: str | None = None,
        content_type: str | None = None,
        original_content_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        wait: bool = True,
        indexing_timeout: float = 60,
    ) -> dict[str, Any]:
        """Create a memory and return its fields, retaining its ID on wait failures."""
        space_id = self._uuid(space_id, "space_id")
        if memory_id is not None:
            memory_id = self._uuid(memory_id, "memory_id")
        if file_path is not None:
            try:
                file_path = str(resolve_upload_path(file_path, self.upload_dir))
            except GoodMemUploadError as exc:
                raise ToolException(str(exc)) from exc
        content: dict[str, Any] = {
            "original_content": original_content,
            "file_path": file_path,
        }
        with self._session() as client:
            memory = client.memories.create(
                space_id=space_id,
                content_type=content_type,
                original_content_ref=original_content_ref,
                metadata=metadata,
                memory_id=memory_id,
                **content,
            )
            if wait:
                try:
                    memory = wait_for_memory(client, memory.memory_id, indexing_timeout)
                except Exception as exc:
                    raise ToolException(
                        f"Memory {memory.memory_id} was created, but waiting for indexing failed: {exc}"
                    ) from exc
            return memory.model_dump(mode="json", exclude_none=True)
