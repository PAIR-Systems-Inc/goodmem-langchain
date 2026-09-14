"""Run LangChain's standard suites against the real SDK without external sockets."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx
import pytest
from goodmem import Goodmem
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import RetrieversIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListEmbedders,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    GoodMemUpdateSpace,
)
from tests.unit_tests.conftest import CHUNK, MEMORY, ndjson

TOOL_CASES = [
    (GoodMemCreateSpace, {"name": "Docs", "embedder_id": "embedder-1"}),
    (GoodMemCreateMemory, {"space_id": "space-1", "original_content": "Text"}),
    (GoodMemDeleteMemory, {"memory_id": "memory-1"}),
    (GoodMemDeleteSpace, {"space_id": "space-1"}),
    (GoodMemGetMemory, {"memory_id": "memory-1"}),
    (GoodMemGetSpace, {"space_id": "space-1"}),
    (GoodMemListEmbedders, {}),
    (GoodMemListMemories, {"space_id": "space-1"}),
    (GoodMemListSpaces, {}),
    (GoodMemRetrieveMemories, {"message": "question", "space_ids": ["space-1"]}),
    (GoodMemUpdateSpace, {"space_id": "space-1", "merge_labels": {"team": "blue"}}),
]


class TestGoodMemTools(ToolsUnitTests):
    @pytest.fixture(autouse=True, params=TOOL_CASES, ids=lambda case: case[0].__name__)
    def configure(self, request: pytest.FixtureRequest) -> None:
        self.constructor, self.arguments = request.param

    @property
    def tool_constructor(self) -> type[BaseTool]:
        return self.constructor

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return self.arguments

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {
                "GOODMEM_BASE_URL": "https://goodmem.test",
                "GOODMEM_API_KEY": "synthetic-key",
            },
            {},
            {
                "goodmem_base_url": "https://goodmem.test",
                "goodmem_api_key": "synthetic-key",
            },
        )


class TestGoodMemRetriever(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def configure(self) -> Iterator[None]:
        chunks = []
        for index in range(3):
            chunk = deepcopy(CHUNK)
            chunk["retrievedItem"]["chunk"]["chunk"]["chunkId"] = f"chunk-{index}"
            chunks.append(chunk)
        with httpx.Client(
            base_url="https://goodmem.test",
            transport=httpx.MockTransport(
                lambda request: ndjson(*chunks, {"memoryDefinition": MEMORY})
            ),
        ) as http:
            self.parameters = {
                "client": Goodmem(http_client=http),
                "space_ids": ["space-1"],
            }
            yield

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return GoodMemRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return self.parameters

    @property
    def retriever_query_example(self) -> str:
        return "Who owns Project Cobalt?"
