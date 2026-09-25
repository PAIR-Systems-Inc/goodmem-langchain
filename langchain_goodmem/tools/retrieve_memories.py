"""Expose GoodMem retrieval events without changing their meaning."""

from typing import Any, cast

from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent
from pydantic import BaseModel, Field

from langchain_goodmem._ids import UUIDStr
from langchain_goodmem.tools._base import GoodMemTool, ToolInput


class RetrieveMemoriesInput(ToolInput):
    """Search arguments using SDK names and types."""

    message: str = Field(min_length=1, description="Natural language search query.")
    space_ids: list[UUIDStr] = Field(
        min_length=1, description="UUIDs of spaces to search."
    )
    requested_size: int = Field(
        default=5, gt=0, description="Number of vector candidates."
    )
    reranker_id: UUIDStr | None = Field(
        default=None, description="Optional reranker UUID; needs no LLM."
    )
    llm_id: UUIDStr | None = Field(
        default=None, description="Optional LLM UUID for summarization."
    )
    max_results: int | None = Field(
        default=None, gt=0, description="Limit after post-processing."
    )
    relevance_threshold: float | None = None
    chronological_resort: bool | None = None
    llm_temp: float | None = None


class GoodMemRetrieveMemories(GoodMemTool):
    """Return SDK retrieval events, including chunks, summaries and diagnostics."""

    name: str = "goodmem_retrieve_memories"
    description: str = (
        "Search GoodMem spaces. Returns SDK events with retrieved_item, "
        "memory_definition, abstract_reply and status fields as applicable. "
        "Inspect status events for incomplete results. Reranking needs no LLM; "
        "llm_id optionally enables summarization."
    )
    args_schema: type[BaseModel] = RetrieveMemoriesInput

    def _run(
        self,
        message: str,
        space_ids: list[str],
        requested_size: int = 5,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        max_results: int | None = None,
        relevance_threshold: float | None = None,
        chronological_resort: bool | None = None,
        llm_temp: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve once and return serialized SDK events in server order."""
        space_ids = [self._uuid(sid, "space_ids") for sid in space_ids]
        if reranker_id is not None:
            reranker_id = self._uuid(reranker_id, "reranker_id")
        if llm_id is not None:
            llm_id = self._uuid(llm_id, "llm_id")
        with self._session() as client:
            events = client.memories.retrieve(
                message=message,
                space_ids=space_ids,
                requested_size=requested_size,
                reranker_id=reranker_id,
                llm_id=llm_id,
                max_results=max_results,
                relevance_threshold=relevance_threshold,
                chronological_resort=chronological_resort,
                llm_temp=llm_temp,
                fetch_memory=True,
                stream=False,
            )
            # The SDK's list method shadows list in its retrieve return annotation.
            return [
                event.model_dump(mode="json", exclude_none=True)
                for event in cast(list[RetrieveMemoryEvent], events)
            ]
