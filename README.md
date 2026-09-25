# langchain-goodmem

Use [GoodMem](https://goodmem.ai) from LangChain for document ingestion, retrieval, and agent memory. GoodMem handles storage, chunking, embeddings, and optional reranking.

Version **0.2** intentionally breaks compatibility with 0.1. See [CHANGELOG.md](CHANGELOG.md) for migration details.

## Install and connect

Requires Python 3.10+, a running GoodMem server, an API key, and a space configured with an embedder.

```bash
pip install langchain-goodmem
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="your-key"
```

Retrievers and tools read these environment variables. You can also pass a configured `goodmem.Goodmem` instance as `client=`; you retain ownership of it.

## Write Documents and retrieve

```python
import os
from goodmem import Goodmem
from langchain_core.documents import Document
from langchain_goodmem import GoodMemRetriever, add_documents

space_id = "your-space-uuid"
with Goodmem(
    base_url=os.environ["GOODMEM_BASE_URL"],
    api_key=os.environ["GOODMEM_API_KEY"],
) as client:
    memory_ids = add_documents(client, space_id, [
        Document(
            page_content="Project Cobalt's launch owner is Ada.",
            metadata={"source": "https://example.org/cobalt", "team": "blue"},
        )
    ])

retriever = GoodMemRetriever(
    space_ids=[space_id], k=5, filter="CAST(val('$.team') AS TEXT) = 'blue'",
)
for document in retriever.invoke("Who owns Cobalt's launch?"):
    print(document.page_content, document.metadata["source"])
```

`add_documents` batches text and metadata through the SDK and waits for indexing by default. Set `wait=False` for background ingestion, then use `wait_for_memory(client, memory_id)` when readiness matters. Optional Document IDs must be UUIDs; existing IDs produce conflicts. `GoodMemIngestionError.created_memory_ids` identifies successful writes if part of ingestion fails.

Filters use [GoodMem expressions](https://docs.goodmem.ai/docs/reference/filter-expressions/) and execute on the server before retrieval. Omit `filter` to search all memories in the configured spaces. Searches run once; empty results return immediately.

The retriever returns Documents with source metadata, memory/chunk/space IDs, and scores. It supports LCEL, callbacks, batching, per-call `k`, and `ainvoke` through LangChain's thread executor.

If the server reports a real problem during a search — a reranker was unavailable, one space was unreachable — the Documents it did return are still returned, and each carries `metadata["goodmem_partial"] = True` and `metadata["goodmem_statuses"]` saying why. A problem that left no Documents at all returns an empty list and emits a `UserWarning` and a warning log line with the statuses, so it is distinguishable from a search that matched nothing. Neither case raises. Notices that carry no loss (`FEATURE_DISABLED`, `LLM_CAPABILITY_INFERRED`) are dropped; a status code this version does not know is reported as `UNKNOWN`.

For reranking, add `reranker_id="your-reranker-uuid"` and optionally `fetch_k=20`. **Reranking requires no LLM.**

## Give an agent a search tool

```python
from langchain_core.tools import create_retriever_tool

tool = create_retriever_tool(
    retriever, "search_project_records", "Search project records.",
    response_format="content_and_artifact",
)
```

The agent supplies only a query. Spaces and filters remain configured by the developer. Retrieved Documents are available in `ToolMessage.artifact` for citations.

## Other tools and development

The package also provides space/memory management tools. `GoodMemRetrieveMemories` returns SDK events, including chunks, optional summaries, and statuses; `GoodMemRetriever` flags incomplete retrieval on the Documents' metadata rather than raising. Tools use SDK data shapes and LangChain `ToolException` handling.

IDs must be UUIDs because the SDK puts them unescaped into URL paths. `GoodMemCreateMemory` offers `file_path` only inside an operator-set `upload_dir`, symlinks resolved.

See [the smoke example](examples/live_smoke_test.py) for a complete workflow.

```bash
uv sync --all-groups
uv run pytest --disable-socket --allow-unix-socket tests/unit_tests
uv run ruff check .
uv run mypy .
```
