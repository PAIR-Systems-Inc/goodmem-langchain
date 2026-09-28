"""The one check every GoodMem ID passes before any request is made.

The GoodMem SDK interpolates IDs into URL paths unescaped, and httpx resolves
dot segments before sending, so ``memories.delete(id="../spaces/X")`` sends
``DELETE /v1/spaces/X``. Percent-encoded forms (``%2e%2e``, ``%2F``) are not
safe either: the server may decode them. Every GoodMem ID (memory, space,
embedder, reranker, LLM, chunk, API key) is a UUID, so anything else is
refused here rather than escaped.
"""

import re
from typing import Annotated

from pydantic import AfterValidator, ValidationInfo, WithJsonSchema

UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
# fullmatch, not match: "$" alone would also accept a trailing newline.
_UUID = re.compile(UUID_PATTERN)


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lowercase canonical UUID, or raise ``ValueError``.

    The result is always a plain ``str`` holding exactly the characters that
    were checked, so a ``str`` subclass cannot pass the check and then put a
    different string into the URL (through an overridden ``lower``,
    ``__str__`` or ``__format__``). Only strings are accepted: a
    ``uuid.UUID`` is refused; pass ``str(value)``.

    Args:
        value: The ID as supplied by a model, developer or configuration.
        field: The argument name, used in the error message.

    Raises:
        ValueError: ``value`` is not a canonical 8-4-4-4-12 hexadecimal UUID.
    """
    if isinstance(value, str):
        if _UUID.fullmatch(value):
            # str.lower, not value.lower: an override must not choose the result.
            return str.lower(value)
        shown = str.__repr__(value)
        expected = "a UUID"
    else:
        try:
            shown = repr(value)
        except Exception:  # The refusal must not depend on the value's repr.
            shown = f"a {type(value).__name__}"
        expected = "a UUID string"
    if len(shown) > 80:
        shown = shown[:77] + "..."
    raise ValueError(
        f"{field} must be {expected} (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx); "
        f"got {shown}"
    )


def _validate(value: str, info: ValidationInfo) -> str:
    return require_uuid(value, info.field_name or "id")


UUIDStr = Annotated[
    str,
    AfterValidator(_validate),
    # Declared in the model-visible schema so the model is told the format.
    WithJsonSchema({"type": "string", "format": "uuid", "pattern": UUID_PATTERN}),
]
"""A string field that only accepts a UUID, normalised to lowercase."""
