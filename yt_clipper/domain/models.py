"""Shared, serializable contracts used by application and UI boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from ..config import (
    ContentType,
    LLMProvider,
    normalize_analysis_model,
    normalize_content_type,
    normalize_llm_provider,
    resolve_analysis_model,
    resolve_codex_binary,
    resolve_codex_timeout_seconds,
    resolve_content_type,
    resolve_highlight_montage,
    resolve_llm_api_key,
    resolve_llm_provider,
)

MAX_SOURCE_DURATION_SECONDS: Final[int] = 60 * 60
MAX_CLIP_DURATION_SECONDS: Final[int] = 60
MAX_CLIP_COUNT: Final[int] = 20
ANALYSIS_SCHEMA_VERSION: Final[int] = 7
ANALYSIS_PROMPT_VERSION: Final[int] = 3
MONTAGE_ANALYSIS_SCHEMA_VERSION: Final[int] = 1
MONTAGE_ANALYSIS_PROMPT_VERSION: Final[int] = 1


def _resolve_default_llm_api_key() -> SecretStr | None:
    api_key = resolve_llm_api_key()
    return SecretStr(api_key) if api_key is not None else None


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class VideoLayout(str, Enum):
    FILL_CROP = "fill-crop"
    FIT_BLUR = "fit-blur"


class WhisperDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class TranscriptMode(str, Enum):
    AUTO = "auto"
    CAPTIONS = "captions"
    WHISPER = "whisper"


class TranscriptOrigin(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    WHISPER = "whisper"


class PipelineStage(str, Enum):
    SETUP = "setup"
    INSPECT = "inspect"
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    ANALYZE = "analyze"
    RENDER = "render"
    COMPLETE = "complete"


class PipelineConfig(StrictModel):
    llm_provider: LLMProvider = Field(
        default_factory=resolve_llm_provider,
        validate_default=True,
    )
    llm_api_key: SecretStr | None = Field(
        default_factory=_resolve_default_llm_api_key,
        exclude=True,
        repr=False,
    )
    model: str = Field(
        default_factory=resolve_analysis_model,
        min_length=1,
        validate_default=True,
    )
    output_dir: Path = Path("output")
    max_source_duration_seconds: int = Field(
        default=MAX_SOURCE_DURATION_SECONDS,
        ge=60,
        le=MAX_SOURCE_DURATION_SECONDS,
    )
    max_source_download_bytes: int = Field(
        default=4 * 1024**3,
        ge=256 * 1024**2,
        le=16 * 1024**3,
    )
    clip_count: int | None = Field(default=None, ge=1, le=MAX_CLIP_COUNT)
    content_type: ContentType = Field(
        default_factory=resolve_content_type,
        validate_default=True,
    )
    highlight_montage: bool = Field(
        default_factory=resolve_highlight_montage,
        validate_default=True,
    )
    highlight_window_seconds: float = Field(default=4.0, ge=3.0, le=6.0)
    highlight_montage_max_duration: float = Field(default=60.0, ge=12.0, le=60.0)
    highlight_montage_max_moments: int = Field(default=12, ge=2, le=20)
    highlight_analysis_batch_windows: int = Field(default=60, ge=10, le=100)
    min_clip_duration: float = Field(
        default=20.0,
        ge=5.0,
        le=MAX_CLIP_DURATION_SECONDS,
    )
    max_clip_duration: float = Field(
        default=60.0,
        ge=5.0,
        le=MAX_CLIP_DURATION_SECONDS,
    )
    language: str = Field(default="en", min_length=2, max_length=32)
    video_layout: VideoLayout = VideoLayout.FILL_CROP
    transcript_mode: TranscriptMode = TranscriptMode.AUTO
    whisper_model: str = Field(default="small", min_length=1)
    whisper_device: WhisperDevice = WhisperDevice.AUTO
    whisper_cpu_threads: int = Field(default=4, ge=1, le=32)
    whisper_batch_size: int = Field(default=1, ge=1, le=8)
    whisper_chunk_seconds: int = Field(default=300, ge=60, le=600)
    whisper_chunk_overlap_seconds: int = Field(default=5, ge=0, le=30)
    whisper_timeout_seconds: int = Field(default=3600, ge=300, le=14_400)
    analysis_chunk_max_characters: int = Field(default=45_000, ge=8_000, le=100_000)
    analysis_chunk_overlap_seconds: float = Field(default=60.0, ge=0, le=300)
    analysis_max_concurrency: int = Field(default=2, ge=1, le=4)
    analysis_request_max_attempts: int = Field(default=3, ge=1, le=6)
    codex_binary: str = Field(default_factory=resolve_codex_binary, min_length=1)
    codex_timeout_seconds: int = Field(
        default_factory=resolve_codex_timeout_seconds,
        ge=30,
        le=1800,
        validate_default=True,
    )
    cookies_from_browser: str | None = None
    analyze_only: bool = False
    force: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_llm_defaults(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values

        resolved = dict(values)
        raw_provider = resolved.get("llm_provider")
        provider = (
            raw_provider
            if isinstance(raw_provider, (LLMProvider, str))
            else None
        )
        if "model" not in resolved:
            resolved["model"] = resolve_analysis_model(provider)
        if "llm_api_key" not in resolved:
            resolved["llm_api_key"] = resolve_llm_api_key(provider)
        return resolved

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider_name(cls, provider: object) -> object:
        if isinstance(provider, (LLMProvider, str)):
            return normalize_llm_provider(provider)
        return provider

    @field_validator("content_type", mode="before")
    @classmethod
    def normalize_editorial_content_type(cls, content_type: object) -> object:
        if isinstance(content_type, (ContentType, str)):
            return normalize_content_type(content_type)
        return content_type

    @field_validator("model")
    @classmethod
    def normalize_model_name(cls, model: str) -> str:
        return normalize_analysis_model(model)

    def get_llm_api_key(self) -> str | None:
        """Return the unwrapped key only at the model request boundary."""
        if self.llm_api_key is None:
            return None
        value = self.llm_api_key.get_secret_value().strip()
        return value or None

    @model_validator(mode="after")
    def validate_duration_range(self) -> "PipelineConfig":
        if self.min_clip_duration > self.max_clip_duration:
            raise ValueError("min_clip_duration cannot exceed max_clip_duration")
        if self.analysis_chunk_overlap_seconds < self.max_clip_duration:
            raise ValueError(
                "analysis_chunk_overlap_seconds cannot be less than max_clip_duration"
            )
        if self.whisper_chunk_overlap_seconds * 2 >= self.whisper_chunk_seconds:
            raise ValueError(
                "whisper_chunk_overlap_seconds must be less than half the chunk size"
            )
        if self.highlight_montage_max_duration < self.highlight_window_seconds * 2:
            raise ValueError(
                "highlight_montage_max_duration must fit at least two highlight windows"
            )
        return self


class VideoMetadata(StrictModel):
    video_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0)
    uploader: str | None = None
    upload_date: str | None = None
    thumbnail_url: str | None = None


class DownloadedVideo(StrictModel):
    metadata: VideoMetadata
    video_path: Path
    metadata_path: Path
    work_dir: Path
    thumbnail_path: Path | None = None


class TranscriptWord(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_times(self) -> "TranscriptWord":
        if self.end <= self.start:
            raise ValueError("word end must be greater than word start")
        return self


class TranscriptSegment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    words: list[TranscriptWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_times(self) -> "TranscriptSegment":
        if self.end <= self.start:
            raise ValueError("segment end must be greater than segment start")
        return self


class TranscriptDocument(StrictModel):
    video_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    requested_language: str = Field(min_length=1)
    origin: TranscriptOrigin
    source_fingerprint: str = Field(min_length=64, max_length=64)
    whisper_model: str | None = None
    whisper_options_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    duration_seconds: float = Field(gt=0)
    segments: list[TranscriptSegment] = Field(min_length=1)

    @field_validator("segments")
    @classmethod
    def ensure_chronological_segments(
        cls, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        return sorted(segments, key=lambda segment: (segment.start, segment.end))

    @model_validator(mode="after")
    def validate_segment_bounds(self) -> "TranscriptDocument":
        for segment in self.segments:
            if segment.end > self.duration_seconds + 0.01:
                raise ValueError("transcript segment exceeds video duration")
            for word in segment.words:
                if word.start < segment.start - 0.01 or word.end > segment.end + 0.01:
                    raise ValueError("transcript word lies outside its segment")
        return self


class ClipCandidate(StrictModel):
    title: str = Field(min_length=1, max_length=120)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    hook: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1000)
    standalone: bool = False
    topic: str | None = Field(default=None, min_length=1, max_length=240)
    opening_context: str | None = Field(default=None, min_length=1, max_length=500)
    closing_resolution: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_times(self) -> "ClipCandidate":
        if self.end <= self.start:
            raise ValueError("clip end must be greater than clip start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class HighlightMoment(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    hook: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_times(self) -> "HighlightMoment":
        if self.end <= self.start:
            raise ValueError("highlight moment end must be greater than start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class HighlightMontage(StrictModel):
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=500)
    moments: list[HighlightMoment] = Field(min_length=2, max_length=20)

    @property
    def duration(self) -> float:
        return sum(moment.duration for moment in self.moments)


class AnalysisDocument(StrictModel):
    schema_version: int = Field(ge=1)
    video_id: str
    model: str
    analysis_backend: str | None = Field(default=None, min_length=1)
    clip_count: int | None = Field(default=None, ge=1, le=MAX_CLIP_COUNT)
    content_type: ContentType = ContentType.AUTO
    min_clip_duration: float
    max_clip_duration: float
    analysis_prompt_version: int
    chunk_max_characters: int
    chunk_overlap_seconds: float
    transcript_origin: TranscriptOrigin
    transcript_sha256: str = Field(min_length=64, max_length=64)
    candidates: list[ClipCandidate]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MontageAnalysisDocument(StrictModel):
    schema_version: int = Field(ge=1)
    analysis_prompt_version: int = Field(ge=1)
    video_id: str
    model: str
    analysis_backend: str = Field(min_length=1)
    content_type: ContentType
    window_seconds: float
    max_duration: float
    max_moments: int
    batch_windows: int
    transcript_sha256: str = Field(min_length=64, max_length=64)
    montage: HighlightMontage
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineEvent(StrictModel):
    stage: PipelineStage
    message: str
    progress: float | None = Field(default=None, ge=0, le=1)
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)


ProgressCallback = Callable[[PipelineEvent], None]


class PipelineResult(StrictModel):
    video: DownloadedVideo
    transcript: TranscriptDocument
    candidates: list[ClipCandidate]
    transcript_path: Path
    candidates_path: Path
    clip_paths: list[Path] = Field(default_factory=list)
    thumbnail_paths: list[Path] = Field(default_factory=list)
    poster_paths: list[Path] = Field(default_factory=list)
    montage: HighlightMontage | None = None
    montage_analysis_path: Path | None = None
    montage_video_path: Path | None = None
    montage_thumbnail_path: Path | None = None
    montage_poster_path: Path | None = None
