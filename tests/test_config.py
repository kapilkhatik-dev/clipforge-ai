from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from yt_clipper import (
    ContentType,
    DEFAULT_ANALYSIS_MODEL,
    LLMProvider,
    PipelineConfig,
    VideoLayout,
    WhisperDevice,
)
from yt_clipper.domain.models import TranscriptSegment, VideoMetadata


@pytest.fixture(autouse=True)
def isolate_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep configuration tests independent from a developer's real .env."""
    for variable_name in (
        "CLIPPER_LLM_PROVIDER",
        "CLIPPER_LLM_MODEL",
        "CLIPPER_LLM_API_KEY",
        "CLIPPER_CODEX_MODEL",
        "CLIPPER_NVIDIA_MODEL",
        "CLIPPER_OPENROUTER_MODEL",
        "CLIPPER_OPENAI_MODEL",
        "CLIPPER_ANTHROPIC_MODEL",
        "CLIPPER_MODEL",
        "CLIPPER_CONTENT_TYPE",
        "CLIPPER_HIGHLIGHT_MONTAGE",
        "CLIPPER_CODEX_BINARY",
        "CLIPPER_CODEX_TIMEOUT_SECONDS",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(variable_name, raising=False)


def test_uses_nvidia_nim_model_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_CONTENT_TYPE", raising=False)

    assert DEFAULT_ANALYSIS_MODEL == (
        "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b"
    )
    config = PipelineConfig()
    assert config.model == DEFAULT_ANALYSIS_MODEL
    assert config.video_layout == VideoLayout.FILL_CROP
    assert config.max_clip_duration == 60
    assert config.clip_count is None
    assert config.content_type == ContentType.AUTO
    assert config.max_source_duration_seconds == 3600
    assert config.analysis_chunk_overlap_seconds == 60
    assert config.analysis_max_concurrency == 2
    assert config.whisper_device == WhisperDevice.AUTO
    assert config.whisper_batch_size == 1
    assert config.whisper_chunk_seconds == 300


def test_accepts_blurred_background_layout_from_user_input() -> None:
    config = PipelineConfig.model_validate({"video_layout": "fit-blur"})

    assert config.video_layout == VideoLayout.FIT_BLUR


def test_rejects_clip_duration_over_sixty_seconds() -> None:
    with pytest.raises(ValueError, match="less than or equal to 60"):
        _ = PipelineConfig(max_clip_duration=61)


def test_accepts_more_than_five_clips_up_to_twenty() -> None:
    assert PipelineConfig(clip_count=10).clip_count == 10
    with pytest.raises(ValueError, match="less than or equal to 20"):
        _ = PipelineConfig(clip_count=21)


def test_omitted_clip_count_enables_automatic_selection() -> None:
    assert PipelineConfig().clip_count is None
    assert PipelineConfig(clip_count=None).clip_count is None


def test_highlight_montage_is_opt_in_and_has_bounded_short_windows() -> None:
    default = PipelineConfig()
    enabled = PipelineConfig(
        highlight_montage=True,
        highlight_window_seconds=5,
        highlight_montage_max_duration=45,
        highlight_montage_max_moments=9,
        highlight_analysis_batch_windows=40,
    )

    assert default.highlight_montage is False
    assert enabled.highlight_montage is True
    assert enabled.highlight_window_seconds == 5
    assert enabled.highlight_montage_max_duration == 45
    assert enabled.highlight_montage_max_moments == 9
    assert enabled.highlight_analysis_batch_windows == 40

    with pytest.raises(ValueError, match="greater than or equal to 12"):
        PipelineConfig(highlight_montage_max_duration=11.9)


def test_highlight_montage_can_be_enabled_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_HIGHLIGHT_MONTAGE", "true")
    assert PipelineConfig().highlight_montage is True

    monkeypatch.setenv("CLIPPER_HIGHLIGHT_MONTAGE", "maybe")
    with pytest.raises(ValueError, match="CLIPPER_HIGHLIGHT_MONTAGE"):
        PipelineConfig()


def test_content_type_can_be_selected_from_environment_or_user_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_CONTENT_TYPE", "comedy")

    assert PipelineConfig().content_type == ContentType.COMEDY
    assert (
        PipelineConfig.model_validate({"content_type": "stand-up"}).content_type
        == ContentType.COMEDY
    )
    assert (
        PipelineConfig.model_validate({"content_type": "educational"}).content_type
        == ContentType.EDUCATION
    )


def test_rejects_unknown_content_type() -> None:
    with pytest.raises(ValueError, match="Unsupported content type"):
        _ = PipelineConfig.model_validate(
            {"content_type": "ignore-the-editorial-rules"}
        )


def test_rejects_source_duration_over_one_hour() -> None:
    with pytest.raises(ValueError, match="less than or equal to 3600"):
        _ = PipelineConfig(max_source_duration_seconds=3601)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: VideoMetadata(
                video_id="video",
                source_url="https://www.youtube.com/watch?v=video",
                title="Video",
                duration_seconds=math.inf,
            ),
            "duration_seconds",
        ),
        (
            lambda: TranscriptSegment(start=0, end=math.nan, text="Invalid"),
            "end",
        ),
    ],
)
def test_domain_models_reject_non_finite_timestamps(
    factory: Callable[[], object],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        factory()


def test_requires_analysis_overlap_to_cover_maximum_clip() -> None:
    with pytest.raises(ValueError, match="cannot be less than max_clip_duration"):
        _ = PipelineConfig(
            max_clip_duration=45,
            analysis_chunk_overlap_seconds=30,
        )


def test_bounds_long_video_resource_controls() -> None:
    with pytest.raises(ValueError, match="less than or equal to 8"):
        _ = PipelineConfig(whisper_batch_size=9)
    with pytest.raises(ValueError, match="less than or equal to 4"):
        _ = PipelineConfig(analysis_max_concurrency=5)


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        (LLMProvider.CODEX, "codex/default"),
        (
            LLMProvider.NVIDIA,
            "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        ),
        (
            LLMProvider.OPENROUTER,
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        ),
        (LLMProvider.OPENAI, "openai/gpt-4.1-mini"),
        (LLMProvider.ANTHROPIC, "anthropic/claude-sonnet-4-6"),
    ],
)
def test_provider_selects_suitable_default_model(
    monkeypatch: pytest.MonkeyPatch,
    provider: LLMProvider,
    expected_model: str,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv(
        f"CLIPPER_{provider.value.upper()}_MODEL",
        raising=False,
    )
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)

    config = PipelineConfig(llm_provider=provider)

    assert config.llm_provider == provider
    assert config.model == expected_model


def test_common_environment_configures_provider_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "OpenAI")
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "shared-test-key")
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)

    config = PipelineConfig()

    assert config.llm_provider == LLMProvider.OPENAI
    assert config.model == "openai/gpt-4.1-mini"
    assert config.get_llm_api_key() == "shared-test-key"
    assert "shared-test-key" not in repr(config)
    assert "llm_api_key" not in config.model_dump()


def test_codex_provider_uses_local_cli_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "codex")
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "must-not-be-used")
    monkeypatch.setenv("CLIPPER_CODEX_BINARY", "custom-codex")
    monkeypatch.setenv("CLIPPER_CODEX_TIMEOUT_SECONDS", "420")
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)

    config = PipelineConfig()

    assert config.llm_provider == LLMProvider.CODEX
    assert config.model == "codex/default"
    assert config.get_llm_api_key() is None
    assert config.codex_binary == "custom-codex"
    assert config.codex_timeout_seconds == 420


def test_openrouter_configuration_remains_provider_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "openrouter-test-key")
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)

    config = PipelineConfig()

    assert config.llm_provider == LLMProvider.OPENROUTER
    assert config.model == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert config.get_llm_api_key() == "openrouter-test-key"


def test_rejects_unsupported_llm_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider 'groq'"):
        _ = PipelineConfig.model_validate({"llm_provider": "groq"})


def test_legacy_provider_api_key_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-test-key")

    config = PipelineConfig(llm_provider=LLMProvider.ANTHROPIC)

    assert config.get_llm_api_key() == "legacy-test-key"


def test_common_environment_can_override_analysis_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "openrouter/openai/gpt-oss-120b:free"
    monkeypatch.setenv("CLIPPER_LLM_MODEL", model)
    monkeypatch.setenv("CLIPPER_NVIDIA_MODEL", "nvidia_nim/provider-specific")
    monkeypatch.setenv("CLIPPER_MODEL", "legacy/provider-model")

    assert PipelineConfig().model == model


def test_provider_specific_models_allow_switching_with_one_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_MODEL", raising=False)
    monkeypatch.delenv("CLIPPER_LLM_API_KEY", raising=False)
    monkeypatch.setenv("CLIPPER_CODEX_MODEL", "codex/gpt-5.6-terra")
    monkeypatch.setenv(
        "CLIPPER_OPENROUTER_MODEL",
        "openrouter/openai/gpt-oss-120b:free",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "codex")
    codex = PipelineConfig()

    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "openrouter")
    openrouter = PipelineConfig()

    assert codex.llm_provider == LLMProvider.CODEX
    assert codex.model == "codex/gpt-5.6-terra"
    assert codex.get_llm_api_key() is None
    assert openrouter.llm_provider == LLMProvider.OPENROUTER
    assert openrouter.model == "openrouter/openai/gpt-oss-120b:free"
    assert openrouter.get_llm_api_key() == "openrouter-test-key"


def test_legacy_environment_can_override_analysis_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.setenv("CLIPPER_MODEL", "test/provider-model")

    assert PipelineConfig().model == "test/provider-model"


@pytest.mark.parametrize(
    "alias",
    [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ],
)
def test_normalizes_nvidia_nemotron_aliases(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.setenv("CLIPPER_MODEL", alias)

    assert PipelineConfig().model == DEFAULT_ANALYSIS_MODEL
    assert PipelineConfig(model=alias).model == DEFAULT_ANALYSIS_MODEL


def test_preserves_explicit_openrouter_nemotron_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openrouter_model = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    monkeypatch.delenv("CLIPPER_LLM_MODEL", raising=False)
    monkeypatch.setenv("CLIPPER_MODEL", openrouter_model)

    assert PipelineConfig().model == openrouter_model
    assert PipelineConfig(model=openrouter_model).model == openrouter_model
