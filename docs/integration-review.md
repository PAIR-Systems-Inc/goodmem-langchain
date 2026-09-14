# Integration review: clean 0.2 API

The SDK migration and clean break are in `569a739`, followed by the Document ingestion, metadata filtering and standard test coverage below. Version 0.2.0 includes both sets of changes. Publication was authorized on September 14, 2026; [verification](validation/verification.json) records the checks performed before publication.

## Follow-up changes

- `add_documents(client, space_id, documents)` writes LangChain Documents through the SDK batch API, preserving metadata and optional UUID IDs. It returns memory IDs and waits for indexing by default. `GoodMemIngestionError.created_memory_ids` retains known successful writes when a batch or wait fails.
- `GoodMemRetriever.filter` passes a native GoodMem expression to each configured space before vector search and reranking. Agent search tools still expose only a query. Per-call `k` now works, as required by the standard retriever suite.
- The create-space tool no longer exposes chunking configuration. Developers configure custom chunking through the SDK. Creation remains explicit; name matching and automatic reuse stay removed.
- All eleven tools use LangChain's `ToolsUnitTests`; the retriever uses `RetrieversIntegrationTests` over both an in-memory HTTP transport and a live corpus. The test dependency is pinned to 1.1.6.
- The docs site's LangChain page covers ingestion, filtered retrieval, reranking and the standard tool factory. The README remains below 500 words.

The older review's 17-code JSON classification, special NOT_FOUND rule, overlap capping, and field-by-field space matching were already deleted in the clean break. The low-level retrieval tool still returns SDK events unchanged. No success/partial envelope or name-matching policy has been restored.

A full VectorStore and native `AsyncGoodmem` support remain separate work. The supplied write path accepts ordinary LangChain Documents; GoodMem can split them into multiple retrieved chunks. Async retrieval currently uses LangChain's executor.

## Validation and size

114 offline tests pass on Python 3.10 and 3.13, including 71 inherited standard tests. Ten live tests cover all eleven tools, Document ingestion, filtered plain/reranked retrieval without an LLM, per-call limits, diagnostics and cleanup. The demo's 12 tests pass. The smoke example and all five Python snippets from the updated docs page also run successfully against temporary spaces.

Production Python grows from 824 to 915 lines for these features (313 to 343 AST statements). That remains 54.1% smaller than the original release's 1,994 lines. The README has 372 words including code. [Measurements](validation/code-size.json) and [verification](validation/verification.json) record the details. Distributions exclude this review and its artifacts.

## GoodMem filter rough edges observed live

- `val('$.application') = 'agentic-rag-goodmem'` returned no matches despite matching stored metadata. `CAST(val('$.application') AS TEXT) = 'agentic-rag-goodmem'` returned the expected five chunks. Examples use the explicit cast; `val` returns JSON.
- A bare `TRUE` predicate produced `VECTOR_SEARCH_FAILED` with `Expected a condition but got a org.jooq.impl.Val`. The strict retriever exposed the server failure. This server issue was recorded, not patched or hidden in the integration.
