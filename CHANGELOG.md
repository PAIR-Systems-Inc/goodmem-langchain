# Changelog

## 0.2.0.dev0 — unreleased

0.2 is an intentional API break. The integration uses the official `goodmem` SDK and adds `GoodMemRetriever`, a LangChain `BaseRetriever` returning Documents with joined source metadata and identifiers.

### Migration from 0.1

- Replace `GoodMemClient` with `goodmem.Goodmem`. Pass the SDK instance through `client=`, or configure tools/retrievers with `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`.
- Tool calls return SDK-shaped Python dictionaries/lists, using snake_case fields, rather than JSON strings with `success`, counts, or custom content fields. Delete tools return `None`. LangChain serializes results for agents. SDK errors become `ToolException`, compatible with LangChain's `handle_tool_error` option.
- `GoodMemCreateSpace` creates a new space. It does not find or reuse spaces by name. Select existing spaces with `GoodMemListSpaces`; supply `default_chunking_config` for custom chunking, otherwise the SDK defaults apply. The previous chunking shortcut arguments are removed.
- `GoodMemCreateMemory` takes `original_content` or `file_path`, exactly one. It waits for the created memory to finish indexing by default. Set `wait=False` for background ingestion. A wait failure includes the created memory ID in the exception.
- `GoodMemRetrieveMemories` takes `message`, a list of `space_ids`, and SDK post-processing parameters (`requested_size`, `max_results`, `llm_temp`, etc.). It returns SDK events, preserving chunks and server statuses together. It does not classify them into `success` or `partial`. Use `GoodMemRetriever` for Documents with retrieval failures raised as exceptions.
- `GoodMemGetMemory` fetches metadata by default. Set `include_content=True` to receive the SDK's base64 `original_content` field in the same request. The separate content fetch, decoding fallbacks, and `contentError` wrapper are removed.
- Space/memory list tools use SDK pagination and `max_items` (default 100; `None` for all).
- `wait_for_indexing` and `public_read` are absent. Unknown tool arguments fail normal schema validation; there are no deprecated argument handlers. Search never polls empty results.
- API keys are stored as `SecretStr` and excluded from serialization.

### LangChain usage

`GoodMemRetriever` supports callbacks, LCEL, batching and executor-backed async. Reranking needs no LLM. Use LangChain's `create_retriever_tool` to expose it to an agent; there is no additional GoodMem factory. `wait_for_memory` checks a specific memory's processing status when coordinating ingestion through the SDK.

Native `AsyncGoodmem` support and the `langchain-tests` standard suites remain follow-up work. Internal review notes and validation reports are excluded from distributions.
