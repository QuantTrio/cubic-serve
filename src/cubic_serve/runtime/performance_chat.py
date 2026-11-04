"""Compatibility names for the shared token-to-chat transport.

Performance qualification belongs to the injected executor, not this adapter.
"""

from cubic_serve.runtime.token_chat import (
    TokenChatBackend,
)
from cubic_serve.runtime.token_chat import (
    TokenChatExecutor as PerformanceChatExecutor,
)
from cubic_serve.runtime.token_chat import (
    TokenGenerateResult as PerformanceGenerateResult,
)
from cubic_serve.runtime.token_chat import (
    TokenGenerationMetadata as PerformanceGenerationMetadata,
)
from cubic_serve.runtime.token_chat import (
    TokenSource as PerformanceTokenSource,
)


class PerformanceChatBackend(TokenChatBackend):
    """Compatibility name used by existing performance executor callers."""


__all__ = (
    "PerformanceChatBackend",
    "PerformanceChatExecutor",
    "PerformanceGenerateResult",
    "PerformanceGenerationMetadata",
    "PerformanceTokenSource",
)
