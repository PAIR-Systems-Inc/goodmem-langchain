"""Run the same official retriever suite against an owned live GoodMem corpus."""

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from goodmem import Goodmem
from goodmem.errors import GoodMemError
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_goodmem import (
    GoodMemCreateSpace,
    GoodMemDeleteSpace,
    GoodMemIngestionError,
    GoodMemRetriever,
    add_documents,
)


@pytest.fixture(scope="module")
def corpus() -> Iterator[tuple[Goodmem, str]]:
    required = ["GOODMEM_BASE_URL", "GOODMEM_API_KEY", "GOODMEM_EMBEDDER_ID"]
    if any(not os.getenv(key) for key in required):
        pytest.skip(
            "Set endpoint, credentials and embedder ID for the live standard suite"
        )
    with Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        timeout=120,
    ) as sdk:
        space = GoodMemCreateSpace(client=sdk).invoke(
            {
                "name": f"langchain-standard-{uuid.uuid4()}",
                "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"],
            }
        )
        sid = space["space_id"]
        try:
            documents = [
                Document(
                    id=str(uuid.uuid4()),
                    page_content=f"Project Cobalt record {index}: {fact}",
                    metadata={
                        "source": f"https://example.org/cobalt/{index}",
                        "team": "blue",
                    },
                )
                for index, fact in enumerate(
                    [
                        "Ada owns the launch.",
                        "The launch is October 17.",
                        "Mira reviews the launch.",
                    ]
                )
            ]
            documents.append(
                Document(
                    page_content="Who owns Project Cobalt? This red team's owner is Sam.",
                    metadata={"team": "red"},
                )
            )
            ids = add_documents(sdk, sid, documents, indexing_timeout=120)
            assert len(ids) == len(documents) and len(set(ids)) == len(ids)
            assert ids[0] == documents[0].id
            yield sdk, sid
        finally:
            GoodMemDeleteSpace(client=sdk).invoke({"space_id": sid})


class TestGoodMemRetrieverLive(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def configure(self, corpus: tuple[Goodmem, str]) -> Iterator[None]:
        sdk, sid = corpus
        self.parameters: dict[str, Any] = {
            "client": sdk,
            "space_ids": [sid],
            "filter": "CAST(val('$.team') AS TEXT) = 'blue'",
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


@pytest.mark.parametrize("rerank", [False, True])
def test_live_filter_excludes_other_metadata_before_limit(
    corpus: tuple[Goodmem, str],
    rerank: bool,
) -> None:
    sdk, sid = corpus
    reranker = os.getenv("GOODMEM_RERANKER_ID") if rerank else None
    if rerank and not reranker:
        pytest.skip("Set GOODMEM_RERANKER_ID for filtered reranking")
    retriever = GoodMemRetriever(
        client=sdk,
        space_ids=[sid],
        k=3,
        filter="CAST(val('$.team') AS TEXT) = 'blue'",
        reranker_id=reranker,
    )
    docs = retriever.invoke("Who owns Project Cobalt?")
    assert len(docs) == 3
    assert all(doc.metadata["team"] == "blue" for doc in docs)
    assert all(
        doc.metadata["source"].startswith("https://example.org/cobalt/") for doc in docs
    )
    assert len(retriever.invoke("Who owns Project Cobalt?", k=1)) == 1
    assert (
        GoodMemRetriever(
            client=sdk, space_ids=[sid], filter="CAST(val('$.team') AS TEXT) = 'absent'"
        ).invoke("Cobalt")
        == []
    )


def test_invalid_filter_raises_instead_of_returning_empty(
    corpus: tuple[Goodmem, str],
) -> None:
    sdk, sid = corpus
    with pytest.raises(GoodMemError):
        GoodMemRetriever(client=sdk, space_ids=[sid], filter="val(").invoke("Cobalt")


def test_live_batch_conflict_retains_successful_write(
    corpus: tuple[Goodmem, str],
) -> None:
    sdk, sid = corpus
    existing = GoodMemRetriever(client=sdk, space_ids=[sid]).invoke("Cobalt")[0]
    new_id = str(uuid.uuid4())
    with pytest.raises(GoodMemIngestionError) as error:
        add_documents(
            sdk,
            sid,
            [
                Document(
                    id=new_id,
                    page_content="A new batch receipt.",
                    metadata={"team": "batch-test"},
                ),
                Document(
                    id=existing.metadata["memory_id"], page_content="Conflicting ID"
                ),
            ],
        )
    assert error.value.created_memory_ids == [new_id]
