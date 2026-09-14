"""Create, index, retrieve and delete a temporary memory. No LLM required.

Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID.
Optionally set GOODMEM_RERANKER_ID to exercise reranking.
Run: python examples/live_smoke_test.py
"""

import os
import uuid

from langchain_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteSpace,
    GoodMemRetriever,
)


def main() -> None:
    space = GoodMemCreateSpace().invoke(
        {
            "name": f"langchain-smoke-{uuid.uuid4()}",
            "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"],
        }
    )
    space_id = space["space_id"]
    try:
        memory = GoodMemCreateMemory().invoke(
            {
                "space_id": space_id,
                "original_content": "Project Cobalt's launch owner is Ada.",
                "metadata": {"source": "https://example.org/cobalt"},
            }
        )
        retriever = GoodMemRetriever(
            space_ids=[space_id], reranker_id=os.getenv("GOODMEM_RERANKER_ID")
        )
        docs = retriever.invoke("Who owns Cobalt's launch?")
        assert docs and "Ada" in docs[0].page_content
        assert docs[0].metadata["memory_id"] == memory["memory_id"]
        print(docs[0].page_content, docs[0].metadata["source"])
    finally:
        GoodMemDeleteSpace().invoke({"space_id": space_id})


if __name__ == "__main__":
    main()
