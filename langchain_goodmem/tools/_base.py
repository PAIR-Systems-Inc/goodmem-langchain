"""Shared connection and LangChain error handling for SDK-backed tools."""

from collections.abc import Iterator
from contextlib import contextmanager

from goodmem.errors import GoodMemError
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict

from langchain_goodmem._connection import GoodMemConnection, GoodMemSDK
from langchain_goodmem._ids import require_uuid


class ToolInput(BaseModel):
    """Validate tool arguments without silently ignoring unknown fields."""

    model_config = ConfigDict(extra="forbid")


class GoodMemTool(GoodMemConnection, BaseTool):
    """Use standard LangChain ToolException handling for SDK failures."""

    @staticmethod
    def _uuid(value: str, field: str) -> str:
        """Refuse a non-UUID ID before any request; the schema is not the guard.

        ``_run`` can be called without schema validation, so every tool checks
        its IDs here, immediately before the SDK call. The SDK puts IDs into
        URL paths unescaped, where ``../`` would reach another resource.
        """
        try:
            return require_uuid(value, field)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

    @contextmanager
    def _session(self) -> Iterator[GoodMemSDK]:
        """Yield an SDK session, translating SDK errors into tool errors."""
        try:
            with super()._session() as client:
                yield client
        except (GoodMemError, ValueError, OSError) as exc:
            raise ToolException(str(exc)) from exc
