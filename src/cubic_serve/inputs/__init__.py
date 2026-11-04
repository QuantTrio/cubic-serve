"""Canonical input realization and encoded-feature ownership."""

from cubic_serve.inputs.chat import (
    ChatInputProcessor,
    ChatProcessorError,
    ReasoningEffort,
    resolve_chat_template_kwargs,
)
from cubic_serve.inputs.features import (
    EncodedFeature,
    EncodedFeatureBundle,
    EncodedFeatureCache,
    FeatureAccessError,
    FeatureAdmissionError,
    FeatureConsumerBinding,
    FeatureIntegrityError,
    FeatureReader,
    FeatureSpan,
    FeatureStore,
    FeatureStoreError,
    FeatureTensorLayout,
)
from cubic_serve.inputs.normalize import (
    MAX_INLINE_MEDIA_BASE64_CHARS,
    ChatNormalizationError,
    normalize_anthropic_messages,
    normalize_openai_chat_completions,
)

__all__ = (
    "MAX_INLINE_MEDIA_BASE64_CHARS",
    "ChatInputProcessor",
    "ChatNormalizationError",
    "ChatProcessorError",
    "EncodedFeature",
    "EncodedFeatureBundle",
    "EncodedFeatureCache",
    "FeatureAccessError",
    "FeatureAdmissionError",
    "FeatureConsumerBinding",
    "FeatureIntegrityError",
    "FeatureReader",
    "FeatureSpan",
    "FeatureStore",
    "FeatureStoreError",
    "FeatureTensorLayout",
    "ReasoningEffort",
    "normalize_anthropic_messages",
    "normalize_openai_chat_completions",
    "resolve_chat_template_kwargs",
)
