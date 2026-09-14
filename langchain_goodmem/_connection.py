"""Shared, non-serializable SDK connection settings."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

from goodmem import Goodmem
from goodmem.api.embedders import EmbeddersAPI
from goodmem.api.memories import MemoriesAPI
from goodmem.api.spaces import SpacesAPI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class GoodMemSDK(Protocol):
    """The SDK installs resources dynamically; expose their types to type checkers."""

    memories: MemoriesAPI
    spaces: SpacesAPI
    embedders: EmbeddersAPI


class GoodMemConnection(BaseModel):
    """Use a caller-owned SDK client, or open and close one per invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Goodmem | None = Field(default=None, exclude=True, repr=False)
    goodmem_base_url: str | None = Field(
        default_factory=lambda: os.getenv("GOODMEM_BASE_URL"),
    )
    goodmem_api_key: SecretStr | str | None = Field(
        default_factory=lambda: SecretStr(os.environ["GOODMEM_API_KEY"])
        if os.getenv("GOODMEM_API_KEY")
        else None,
        exclude=True,
        repr=False,
    )
    goodmem_verify_ssl: bool = Field(
        default_factory=lambda: os.getenv("GOODMEM_VERIFY_SSL", "true").lower()
        != "false"
    )
    goodmem_timeout: float = Field(default=30.0, gt=0)

    @field_validator("goodmem_api_key")
    @classmethod
    def _secret_key(cls, value: SecretStr | str | None) -> SecretStr | None:
        return SecretStr(value) if isinstance(value, str) else value

    @contextmanager
    def _session(self) -> Iterator[GoodMemSDK]:
        if self.client is not None:
            yield cast(GoodMemSDK, self.client)
            return
        if not self.goodmem_base_url or not self.goodmem_api_key:
            raise ValueError(
                "Provide client=Goodmem(...), or goodmem_base_url/goodmem_api_key "
                "(also read from GOODMEM_BASE_URL/GOODMEM_API_KEY)."
            )
        with Goodmem(
            base_url=self.goodmem_base_url,
            api_key=(
                self.goodmem_api_key.get_secret_value()
                if isinstance(self.goodmem_api_key, SecretStr)
                else self.goodmem_api_key
            ),
            timeout=self.goodmem_timeout,
            verify=self.goodmem_verify_ssl,
        ) as client:
            yield cast(GoodMemSDK, client)
