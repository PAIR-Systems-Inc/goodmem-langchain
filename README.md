# goodmem-langchain

Use [GoodMem](https://goodmem.ai) from LangChain for document ingestion, retrieval, and agent memory. GoodMem handles storage, chunking, embeddings, and optional reranking.

Formerly `langchain-goodmem` (last release 0.2.3); moved into the PAIR Systems PyPI organisation under the `goodmem-<framework>` naming used by goodmem-adk and goodmem-semantic-kernel. Imports (`langchain_goodmem`) are unchanged. Both names ship the same files, so upgrade with `pip uninstall -y langchain-goodmem && pip install goodmem-langchain`.

Version **0.2** breaks compatibility with 0.1; see [CHANGELOG.md](CHANGELOG.md) for migration.

## Install and connect

Requires Python 3.10+, a running GoodMem server, an API key, and a space configured with an embedder.

```bash
pip install goodmem-langchain
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="your-key"
```

Retrievers and tools read these variables, or take a configured `goodmem.Goodmem` as `client=`; you keep ownership of it.

## Write Documents and retrieve

```python
import os
from goodmem import Goodmem
from langchain_core.documents import Document
from langchain_goodmem import GoodMemRetriever, add_documents

space_id = "5f6b1c2e-8a4d-4e3b-9c7a-2d1e0f9a8b7c"  # your space's UUID
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

retriever = GoodMemRetriever(space_ids=[space_id], k=5, metadata_filter={"team": "blue"})
for document in retriever.invoke("Who owns Cobalt's launch?"):
    print(document.page_content, document.metadata["source"])
```

`add_documents` batches text and metadata through the SDK and waits for indexing by default. Set `wait=False` for background ingestion, then use `wait_for_memory(client, memory_id)` when readiness matters. Optional Document IDs must be UUIDs; existing IDs produce conflicts. `GoodMemIngestionError.created_memory_ids` identifies successful writes if part of ingestion fails.

`metadata_filter` pairs must all match; values are escaped and typed. Build richer filters with `langchain_goodmem.filters` (`all_of`, `compare`, `one_of`); never format user input into `filter`. Empty results return immediately.

The retriever returns Documents with source metadata, memory/chunk/space IDs, and scores. It supports LCEL, callbacks, batching, per-call `k`, and `ainvoke` through LangChain's thread executor.

If a search hits a real problem (a reranker unavailable, a space unreachable), the Documents it found are still returned, each carrying `metadata["goodmem_partial"] = True` and `metadata["goodmem_statuses"]` saying why. A problem that left no Documents returns `[]` with a `UserWarning` and a warning log line, unlike a search that matched nothing. Neither case raises. Notices that carry no loss (`FEATURE_DISABLED`, `LLM_CAPABILITY_INFERRED`) are dropped; a status code this version does not know is reported as `UNKNOWN`.

For reranking, set `reranker_id` and optionally `fetch_k=20`. **Reranking requires no LLM.**

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

The package also provides space/memory management tools. `GoodMemRetrieveMemories` returns SDK events, including chunks, optional summaries, and statuses. Tools use SDK data shapes and LangChain `ToolException` handling.

IDs must be UUIDs because the SDK puts them unescaped into URL paths. Model file uploads need `GoodMemCreateMemory(upload_dir="/srv/uploads")`, confined there, symlinks resolved.

See [the smoke example](examples/live_smoke_test.py) for a complete workflow.

```bash
uv sync --all-groups
uv run pytest --disable-socket --allow-unix-socket tests/unit_tests
uv run ruff check .
uv run mypy .
```
