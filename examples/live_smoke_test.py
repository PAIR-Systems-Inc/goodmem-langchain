"""Write and retrieve LangChain Documents with filtering and optional reranking.

Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID.
Optionally set GOODMEM_RERANKER_ID. No LLM is required.
Run: python examples/live_smoke_test.py
"""

import os
import uuid

from goodmem import Goodmem
from langchain_core.documents import Document

from langchain_goodmem import (
    GoodMemCreateSpace,
    GoodMemDeleteSpace,
    GoodMemRetriever,
    add_documents,
)


def main() -> None:
    with Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"], api_key=os.environ["GOODMEM_API_KEY"]
    ) as client:
        space = GoodMemCreateSpace(client=client).invoke(
            {
                "name": f"langchain-smoke-{uuid.uuid4()}",
                "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"],
            }
        )
        sid = space["space_id"]
        try:
            ids = add_documents(
                client,
                sid,
                [
                    Document(
                        page_content="Project Cobalt's launch owner is Ada.",
                        metadata={
                            "source": "https://example.org/cobalt",
                            "team": "blue",
                        },
                    ),
                    Document(
                        page_content="Project Cobalt's other team is led by Sam.",
                        metadata={"team": "red"},
                    ),
                ],
            )
            retriever = GoodMemRetriever(
                client=client,
                space_ids=[sid],
                reranker_id=os.getenv("GOODMEM_RERANKER_ID"),
                filter="CAST(val('$.team') AS TEXT) = 'blue'",
            )
            docs = retriever.invoke("Who owns Cobalt's launch?")
            assert len(docs) == 1 and "Ada" in docs[0].page_content
            assert docs[0].metadata["memory_id"] == ids[0]
            print(docs[0].page_content, docs[0].metadata["source"])
        finally:
            GoodMemDeleteSpace(client=client).invoke({"space_id": sid})


if __name__ == "__main__":
    main()
