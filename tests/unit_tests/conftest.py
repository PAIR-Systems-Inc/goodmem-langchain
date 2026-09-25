"""Real SDK response parsing over an in-memory HTTP transport."""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from goodmem import Goodmem

# GoodMem IDs are UUIDs; the integration refuses anything else before a request.
SPACE_ID = "00000000-0000-4000-8000-00000000a001"
SPACE_ID_2 = "00000000-0000-4000-8000-00000000a002"
MEMORY_ID = "00000000-0000-4000-8000-00000000b001"
MEMORY_ID_2 = "00000000-0000-4000-8000-00000000b002"
EMBEDDER_ID = "00000000-0000-4000-8000-00000000c001"
RERANKER_ID = "00000000-0000-4000-8000-00000000d001"
LLM_ID = "00000000-0000-4000-8000-00000000e001"

AUDIT: dict[str, Any] = dict(
    createdAt=1, updatedAt=1, createdById="user", updatedById="user"
)
MEMORY: dict[str, Any] = dict(
    **AUDIT,
    memoryId=MEMORY_ID,
    spaceId=SPACE_ID,
    contentType="text/plain",
    processingStatus="COMPLETED",
    pageImageStatus="COMPLETED",
    pageImageCount=0,
    metadata={
        "source": "https://example.org/docs",
        "title": "Docs",
        "custom": {"nested": True},
    },
)
CHUNK: dict[str, Any] = dict(
    retrievedItem={
        "chunk": dict(
            resultSetId="set-1",
            memoryIndex=0,
            relevanceScore=-0.82,
            chunk=dict(
                **AUDIT,
                chunkId="chunk-1",
                memoryId=MEMORY_ID,
                chunkSequenceNumber=0,
                chunkText="Retrieved evidence",
                vectorStatus="COMPLETED",
            ),
        )
    }
)
CONFIG: dict[str, Any] = {
    "recursive": dict(
        chunkSize=512,
        chunkOverlap=50,
        keepStrategy="KEEP_END",
        lengthMeasurement="CHARACTER_COUNT",
    )
}
SPACE: dict[str, Any] = dict(
    **AUDIT,
    spaceId=SPACE_ID,
    name="Docs",
    ownerId="user",
    labels={},
    spaceEmbedders=[
        dict(
            **AUDIT,
            spaceId=SPACE_ID,
            embedderId=EMBEDDER_ID,
            defaultRetrievalWeight=1,
        )
    ],
    defaultChunkingConfig=CONFIG,
)


def ndjson(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n".join(json.dumps(e) for e in events),
        headers={"content-type": "application/x-ndjson"},
    )


class Wire:
    def __init__(self) -> None:
        self.responses: list[httpx.Response | Exception] = []
        self.requests: list[httpx.Request] = []
        self.http = httpx.Client(
            base_url="https://goodmem.test", transport=httpx.MockTransport(self.handle)
        )
        self.sdk = Goodmem(http_client=self.http)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.responses, f"Unexpected request: {request.method} {request.url}"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def wire() -> Iterator[Wire]:
    transport = Wire()
    try:
        yield transport
        assert not transport.responses, "Expected HTTP requests were not made"
    finally:
        transport.http.close()
