"""Test that all public imports work."""

from langchain_goodmem import (
    GoodMemClient,
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListEmbedders,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemUpdateSpace,
)


def test_all_imports() -> None:
    assert GoodMemClient is not None
    assert GoodMemCreateSpace is not None
    assert GoodMemCreateMemory is not None
    assert GoodMemRetrieveMemories is not None
    assert GoodMemGetMemory is not None
    assert GoodMemDeleteMemory is not None
    assert GoodMemListEmbedders is not None
    assert GoodMemListSpaces is not None
    assert GoodMemGetSpace is not None
    assert GoodMemUpdateSpace is not None
    assert GoodMemDeleteSpace is not None
    assert GoodMemListMemories is not None
