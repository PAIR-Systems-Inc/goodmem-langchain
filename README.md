# langchain-goodmem

Use [GoodMem](https://goodmem.ai) from LangChain for document retrieval and agent memory. GoodMem handles storage, chunking, embeddings, and optional reranking.

This is the **0.2 development API**, which intentionally breaks compatibility with 0.1. See [CHANGELOG.md](CHANGELOG.md) for migration details.

## Install and connect

Requires Python 3.10+, a running GoodMem server, an API key, and a space configured with an embedder. Install this development version from its source checkout:

```bash
pip install .
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="your-key"
```

Components read these environment variables. You can instead pass a configured `goodmem.Goodmem` instance as `client=`; you retain ownership of that client.

## Store and retrieve

```python
from langchain_goodmem import GoodMemCreateMemory, GoodMemRetriever

space_id = "your-space-uuid"

memory = GoodMemCreateMemory().invoke({
    "space_id": space_id,
    "original_content": "Project Cobalt's launch owner is Ada.",
    "metadata": {"source": "https://example.org/cobalt"},
})

retriever = GoodMemRetriever(space_ids=[space_id], k=5)
documents = retriever.invoke("Who owns Cobalt's launch?")
for document in documents:
    print(document.page_content, document.metadata["source"])
```

Creation waits for that memory to finish indexing. For background ingestion, set `wait=False`, then call `wait_for_memory(client, memory_id)` when readiness matters. Searches run once; empty results return immediately.

The retriever returns LangChain `Document` objects with source metadata, memory/chunk/space IDs, and scores. It supports LCEL, callbacks, batching, and `ainvoke` through LangChain's thread executor. Retrieval failures raise exceptions.

For reranking, add `reranker_id="your-reranker-uuid"` and optionally `fetch_k=20`. **Reranking requires no LLM.**

## Give an agent a search tool

Use LangChain's standard factory. The agent supplies a query; the developer configures the spaces.

```python
from langchain_core.tools import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "search_project_records",
    "Search project records for factual answers.",
    response_format="content_and_artifact",
)
```

Retrieved Documents remain available in `ToolMessage.artifact` for citations.

## Other tools

The package also provides create/get/update/delete/list space tools, create/get/delete/list memory tools, and `GoodMemListEmbedders`.

`GoodMemRetrieveMemories` exposes SDK retrieval events, including optional LLM summaries and server statuses. Use it when you need that detail; use `GoodMemRetriever` for Documents.

Tools return SDK-shaped dictionaries/lists with snake_case fields, without a `success` envelope. SDK failures raise LangChain `ToolException`; standard `handle_tool_error` configuration is available.

See [the smoke example](examples/live_smoke_test.py) for a complete create, retrieve, and cleanup workflow.

## Development

```bash
uv sync --all-groups
uv run pytest --disable-socket --allow-unix-socket tests/unit_tests
uv run ruff check .
uv run mypy .
```
