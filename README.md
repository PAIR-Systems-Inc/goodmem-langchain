# langchain-goodmem

[![PyPI](https://img.shields.io/pypi/v/langchain-goodmem.svg)](https://pypi.org/project/langchain-goodmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

LangChain integration for [GoodMem](https://goodmem.ai) — long-term agent memory with semantic storage and retrieval.

GoodMem is a memory layer for AI agents that handles embedding, vector search, reranking, and LLM-powered answering server-side. This package exposes GoodMem operations as LangChain `BaseTool`s that can be wired into any LangChain agent or chain.

## Installation

```bash
pip install langchain-goodmem
```

Requires Python 3.10+.

## Tools

| Tool | Description |
|---|---|
| `GoodMemListEmbedders` | List available embedder models |
| `GoodMemListSpaces` | List all spaces in your account |
| `GoodMemGetSpace` | Fetch a single space by ID |
| `GoodMemCreateSpace` | Create a new space or reuse an existing one |
| `GoodMemUpdateSpace` | Update name / public-read / labels on a space |
| `GoodMemDeleteSpace` | Delete a space (cascades to its memories) |
| `GoodMemCreateMemory` | Store text or files as memories |
| `GoodMemListMemories` | Paginate memories within a space |
| `GoodMemRetrieveMemories` | Semantic search with optional reranker / LLM summary |
| `GoodMemGetMemory` | Fetch a specific memory by ID |
| `GoodMemDeleteMemory` | Permanently delete a memory |

## Quick start

```python
from langchain_goodmem import (
    GoodMemCreateSpace,
    GoodMemCreateMemory,
    GoodMemRetrieveMemories,
)

goodmem_kwargs = {
    "goodmem_base_url": "http://localhost:8080",
    "goodmem_api_key": "your-api-key",
}

tools = [
    GoodMemCreateSpace(**goodmem_kwargs),
    GoodMemCreateMemory(**goodmem_kwargs),
    GoodMemRetrieveMemories(**goodmem_kwargs),
]
```

## Use with a LangChain agent

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

llm = init_chat_model("openai:gpt-4o")
agent = create_agent(llm, tools)

response = agent.invoke({"messages": [{"role": "user", "content": "Save this fact and recall it later."}]})
```

## Retrieve with reranking and LLM summary

`GoodMemRetrieveMemories` exposes the full server-side post-processor. Set any of these fields to enable reranking, threshold filtering, chronological re-sort, or an LLM-generated `abstractReply` summary:

```python
result = retrieve.invoke({
    "query": "Which framework develops applications with language models?",
    "space_ids": "space-uuid-1,space-uuid-2",
    "max_results": 5,
    "reranker_id": "reranker-uuid",
    "llm_id": "llm-uuid",
    "llm_temperature": 0.2,
    "relevance_threshold": 0.1,
    "chronological_resort": False,
})
```

Leave them all unset for plain semantic search.

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server |
| `GOODMEM_API_KEY` | API key for authentication |
| `GOODMEM_VERIFY_SSL` | Set to `false` to skip TLS verification for self-signed dev certs (default: `true`) |

## Examples

A live end-to-end smoke test exercising every tool and every post-processor knob is in [examples/live_smoke_test.py](examples/live_smoke_test.py).

```bash
export GOODMEM_BASE_URL=https://localhost:8080
export GOODMEM_API_KEY=...
export GOODMEM_VERIFY_SSL=false
python examples/live_smoke_test.py
```

## License

MIT — see [LICENSE](LICENSE).
