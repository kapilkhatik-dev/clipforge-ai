"""Environment-aware defaults shared by application entry points."""

from __future__ import annotations

import os
from enum import Enum
from typing import Final


class LLMProvider(str, Enum):
    CODEX = "codex"
    NVIDIA = "nvidia"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


DEFAULT_LLM_PROVIDER: Final[LLMProvider] = LLMProvider.NVIDIA
_DEFAULT_ANALYSIS_MODELS: Final[dict[LLMProvider, str]] = {
    LLMProvider.CODEX: "codex/default",
    LLMProvider.NVIDIA: "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
    LLMProvider.OPENROUTER: (
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    ),
    LLMProvider.OPENAI: "openai/gpt-4.1-mini",
    LLMProvider.ANTHROPIC: "anthropic/claude-sonnet-4-6",
}
DEFAULT_ANALYSIS_MODEL: Final[str] = _DEFAULT_ANALYSIS_MODELS[DEFAULT_LLM_PROVIDER]
_PROVIDER_API_KEY_ENV_NAMES: Final[dict[LLMProvider, tuple[str, ...]]] = {
    LLMProvider.CODEX: (),
    LLMProvider.NVIDIA: ("NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"),
    LLMProvider.OPENROUTER: ("OPENROUTER_API_KEY",),
    LLMProvider.OPENAI: ("OPENAI_API_KEY",),
    LLMProvider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
}
_PROVIDER_MODEL_ENV_NAMES: Final[dict[LLMProvider, str]] = {
    LLMProvider.CODEX: "CLIPPER_CODEX_MODEL",
    LLMProvider.NVIDIA: "CLIPPER_NVIDIA_MODEL",
    LLMProvider.OPENROUTER: "CLIPPER_OPENROUTER_MODEL",
    LLMProvider.OPENAI: "CLIPPER_OPENAI_MODEL",
    LLMProvider.ANTHROPIC: "CLIPPER_ANTHROPIC_MODEL",
}
_NVIDIA_API_MODEL: Final[str] = "nvidia/nemotron-3-ultra-550b-a55b"
_LEGACY_NEMOTRON_MODEL: Final[str] = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)
_NVIDIA_NEMOTRON_ALIASES: Final[frozenset[str]] = frozenset(
    {_NVIDIA_API_MODEL, _LEGACY_NEMOTRON_MODEL}
)


def normalize_llm_provider(provider: LLMProvider | str) -> LLMProvider:
    """Normalize a user-facing provider name."""
    if isinstance(provider, LLMProvider):
        return provider
    normalized = provider.strip().lower().replace("-", "_")
    if normalized == "nvidia_nim":
        normalized = LLMProvider.NVIDIA.value
    try:
        return LLMProvider(normalized)
    except ValueError as exc:
        supported = ", ".join(item.value for item in LLMProvider)
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Choose one of: {supported}."
        ) from exc


def resolve_llm_provider() -> LLMProvider:
    """Return the provider selected through the common application setting."""
    configured = os.getenv("CLIPPER_LLM_PROVIDER", DEFAULT_LLM_PROVIDER.value)
    return normalize_llm_provider(configured)


def default_analysis_model(provider: LLMProvider | str) -> str:
    """Return the provider-qualified default model for a supported provider."""
    return _DEFAULT_ANALYSIS_MODELS[normalize_llm_provider(provider)]


def normalize_analysis_model(model: str) -> str:
    """Normalize known model aliases to provider-qualified LiteLLM names."""
    normalized = model.strip()
    if normalized in _NVIDIA_NEMOTRON_ALIASES:
        return DEFAULT_ANALYSIS_MODEL
    return normalized


def resolve_analysis_model(provider: LLMProvider | str | None = None) -> str:
    """Return an explicit model override or the selected provider's default."""
    resolved_provider = (
        resolve_llm_provider() if provider is None else normalize_llm_provider(provider)
    )
    configured = os.getenv("CLIPPER_LLM_MODEL", "").strip()
    if configured:
        return normalize_analysis_model(configured)

    provider_model = os.getenv(
        _PROVIDER_MODEL_ENV_NAMES[resolved_provider],
        "",
    ).strip()
    if provider_model:
        return normalize_analysis_model(provider_model)

    legacy_model = os.getenv("CLIPPER_MODEL", "").strip()
    if legacy_model:
        return normalize_analysis_model(legacy_model)
    return default_analysis_model(resolved_provider)


def resolve_llm_api_key(provider: LLMProvider | str | None = None) -> str | None:
    """Resolve the shared key, falling back to legacy provider variables."""
    resolved_provider = (
        resolve_llm_provider() if provider is None else normalize_llm_provider(provider)
    )
    if resolved_provider == LLMProvider.CODEX:
        return None

    common_key = os.getenv("CLIPPER_LLM_API_KEY", "").strip()
    if common_key:
        return common_key

    for variable_name in _PROVIDER_API_KEY_ENV_NAMES[resolved_provider]:
        provider_key = os.getenv(variable_name, "").strip()
        if provider_key:
            return provider_key
    return None


def resolve_codex_binary() -> str:
    """Return the Codex CLI command or path configured for local analysis."""
    return os.getenv("CLIPPER_CODEX_BINARY", "codex").strip() or "codex"


def resolve_codex_timeout_seconds() -> int:
    """Return the maximum duration of one local Codex CLI invocation."""
    configured = os.getenv("CLIPPER_CODEX_TIMEOUT_SECONDS", "").strip()
    if not configured:
        return 300
    try:
        return int(configured)
    except ValueError as exc:
        raise ValueError("CLIPPER_CODEX_TIMEOUT_SECONDS must be an integer.") from exc
