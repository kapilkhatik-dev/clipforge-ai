"""Public, browser-safe contracts for the local HTTP API."""

from __future__ import annotations

import re
import ipaddress
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)

from ..config import ContentType, LLMProvider
from ..domain.errors import (
    AnalysisError,
    ClipperError,
    DependencyError,
    DownloadError,
    DurationLimitError,
    RenderError,
    SourceTransferError,
    TranscriptError,
)
from ..domain.models import (
    MAX_CLIP_COUNT,
    TranscriptMode,
    VideoLayout,
)


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class ProviderModel(ApiModel):
    id: str
    label: str


ProviderConfigurationScalar = str | int | float | bool


class ProviderConfigurationField(ApiModel):
    """Browser-renderable schema for one provider-owned setting."""

    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9]*$", max_length=80)
    label: str = Field(min_length=1, max_length=120)
    input_type: Literal["text", "secret", "number"]
    section: str = Field(min_length=1, max_length=120)
    section_description: str | None = Field(default=None, max_length=320)
    help_text: str | None = Field(default=None, max_length=500)
    placeholder: str | None = Field(default=None, max_length=200)
    required: bool = False
    write_only: bool = False
    clearable: bool = False
    min_length: int | None = Field(default=None, ge=0, le=8192)
    max_length: int | None = Field(default=None, ge=1, le=8192)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = Field(default=None, gt=0)
    suffix: str | None = Field(default=None, max_length=24)

    @model_validator(mode="after")
    def validate_constraints(self) -> "ProviderConfigurationField":
        if self.input_type == "number":
            if self.min_length is not None or self.max_length is not None:
                raise ValueError("Number fields cannot declare text length constraints")
        elif self.minimum is not None or self.maximum is not None or self.step is not None:
            raise ValueError("Only number fields can declare numeric constraints")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("minLength cannot exceed maxLength")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot exceed maximum")
        if self.write_only and self.input_type != "secret":
            raise ValueError("Only secret fields can be write-only")
        return self


class ProviderDescriptor(ApiModel):
    id: LLMProvider
    display_name: str
    description: str
    transport: Literal["local-cli", "hosted-api"]
    requires_credential: bool
    default_model: str
    models: list[ProviderModel]
    allow_custom_model: bool = True
    capabilities: list[str]
    configuration_fields: list[ProviderConfigurationField] = Field(
        default_factory=list
    )


class CredentialState(ApiModel):
    configured: bool
    source: Literal["environment", "runtime"] | None = None


class ProviderRuntimeConfig(ApiModel):
    codex_binary: str | None = None
    codex_timeout_seconds: int | None = None


class ProviderConfigurationState(ApiModel):
    """Read-safe state for a field; write-only values are always omitted."""

    value: ProviderConfigurationScalar | None = None
    configured: bool = False
    source: Literal["environment", "runtime", "default"] | None = None


class ProviderTestResult(ApiModel):
    status: Literal["healthy", "unhealthy", "untested"]
    tested_at: datetime | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    message: str | None = None


class ProviderProfile(ApiModel):
    id: str
    provider_id: LLMProvider
    name: str
    model: str
    active: bool
    generation_ready: bool = False
    credential: CredentialState
    config: ProviderRuntimeConfig | None = None
    configuration: dict[str, ProviderConfigurationState] = Field(default_factory=dict)
    last_test: ProviderTestResult | None = None


class ProviderPatch(ApiModel):
    model: str | None = Field(default=None, min_length=1, max_length=240)
    # New provider-agnostic patch shape. Values are validated against the selected
    # provider's descriptor before any runtime setting is mutated. ``None`` means
    # clear/reset for fields whose descriptor permits it.
    configuration: dict[str, ProviderConfigurationScalar | None] = Field(
        default_factory=dict,
        repr=False,
    )
    # Compatibility aliases retained for existing API clients.
    api_key: SecretStr | None = Field(default=None, min_length=1, max_length=8192)
    clear_api_key: bool = False
    codex_binary: str | None = Field(default=None, min_length=1, max_length=1024)
    codex_timeout_seconds: int | None = Field(default=None, ge=30, le=1800)


class DefaultProviderPatch(ApiModel):
    provider_profile_id: str = Field(min_length=1, max_length=120)


class DefaultProviderResponse(ApiModel):
    default_provider_profile_id: str


class BootstrapCapabilities(ApiModel):
    local_upload: bool = False
    clip_editor: bool = False


class BootstrapResponse(ApiModel):
    app_name: str = "ClipForge AI"
    version: str = "0.1.0"
    default_provider_profile_id: str | None
    capabilities: BootstrapCapabilities = Field(default_factory=BootstrapCapabilities)


class SourceInput(ApiModel):
    kind: Literal["youtube"] = "youtube"
    url: HttpUrl

    @field_validator("url")
    @classmethod
    def reject_local_targets(cls, value: HttpUrl) -> HttpUrl:
        host = (value.host or "").casefold()
        is_private_address = False
        try:
            address = ipaddress.ip_address(host.strip("[]"))
            is_private_address = not address.is_global
        except ValueError:
            pass
        if (
            value.username is not None
            or value.password is not None
        ):
            raise ValueError("Source URLs cannot contain credentials")
        if (
            host in {"localhost", "0.0.0.0", "::1"}
            or host.endswith(".local")
            or is_private_address
        ):
            raise ValueError("Local source URLs are not supported")
        youtube_hosts = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtu.be",
            "www.youtu.be",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }
        if host not in youtube_hosts:
            raise ValueError("Only YouTube source URLs are currently supported")
        return value


class ProjectCreate(ApiModel):
    source: SourceInput


class ProjectCreated(ApiModel):
    id: str


class ProjectSource(ApiModel):
    kind: Literal["youtube"] = "youtube"
    url: str


class GenerationOptions(ApiModel):
    clip_count: int | None = Field(default=None, ge=1, le=MAX_CLIP_COUNT)
    content_type: ContentType = ContentType.AUTO
    video_layout: VideoLayout = VideoLayout.FILL_CROP
    min_clip_duration: float = Field(default=20, ge=5, le=60)
    max_clip_duration: float = Field(default=60, ge=5, le=60)
    highlight_montage: bool = False
    highlight_window_seconds: float = Field(default=4, ge=3, le=6)
    highlight_montage_max_duration: float = Field(default=60, ge=12, le=60)
    highlight_montage_max_moments: int = Field(default=12, ge=2, le=20)
    transcript_mode: TranscriptMode = TranscriptMode.AUTO
    analysis_max_concurrency: int = Field(default=2, ge=1, le=4)
    analysis_request_max_attempts: int = Field(default=3, ge=1, le=6)
    force: bool = False

    @field_validator("max_clip_duration")
    @classmethod
    def _finite_duration(cls, value: float) -> float:
        if not 5 <= value <= 60:
            raise ValueError("Clip duration must be between 5 and 60 seconds")
        return value


class GenerationCreate(ApiModel):
    provider_profile_id: str = Field(min_length=1, max_length=120)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class AssetLinks(ApiModel):
    video_url: str
    download_url: str | None = None
    thumbnail_url: str | None = None
    poster_url: str | None = None


class EditorLink(ApiModel):
    route: str
    available: bool = False


class SourceRange(ApiModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceRange":
        if self.end <= self.start:
            raise ValueError("source range end must be greater than start")
        return self


class ClipEditDecisionList(ApiModel):
    """Versioned, non-destructive source decisions reserved for the editor."""

    version: int = Field(default=1, ge=1)
    kind: Literal["continuous", "montage"]
    source_video_id: str = Field(min_length=1)
    source_ranges: list[SourceRange] = Field(min_length=1)
    video_layout: VideoLayout
    caption_preset: str = Field(min_length=1)


class ClipAsset(ApiModel):
    id: str
    title: str
    start: float
    end: float
    duration: float
    score: float | None = None
    hook: str | None = None
    reason: str | None = None
    assets: AssetLinks
    editor: EditorLink
    # Optional only for rehydrating pre-editor local state; all new exports set it.
    edit_decision_list: ClipEditDecisionList | None = None


class SourceSummary(ApiModel):
    title: str | None = None
    thumbnail_url: str | None = None
    duration: float | None = None


class JobResult(ApiModel):
    source: SourceSummary | None = None
    montage: ClipAsset | None = None
    clips: list[ClipAsset] = Field(default_factory=list)


class JobError(ApiModel):
    code: str
    message: str


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(ApiModel):
    id: str
    project_id: str
    status: JobStatus
    stage: str | None = None
    stage_progress: float | None = Field(default=None, ge=0, le=1)
    message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JobResult | None = None
    error: JobError | None = None


class ProjectResponse(ApiModel):
    id: str
    created_at: datetime
    source: ProjectSource
    jobs: list[JobResponse] = Field(default_factory=list)


class GenerationSummary(ApiModel):
    """Small generation projection used by the recent-projects index."""

    id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    title: str | None = None
    thumbnail_url: str | None = None
    export_count: int = Field(default=0, ge=0)


class ProjectSummary(ApiModel):
    """Project list item without the full generation results or event history."""

    id: str
    created_at: datetime
    source: ProjectSource
    generation_count: int = Field(ge=1)
    latest_generation: GenerationSummary


class ProjectListResponse(ApiModel):
    items: list[ProjectSummary] = Field(default_factory=list)
    next_cursor: str | None = None


class PublicPipelineEvent(ApiModel):
    stage: str
    message: str
    progress: float | None = None
    current: int | None = None
    total: int | None = None


class JobEventEnvelope(ApiModel):
    job_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime
    type: Literal["progress", "completed", "failed"]
    event: PublicPipelineEvent | None = None
    job: JobResponse | None = None


class DiagnosticCheck(ApiModel):
    status: Literal["healthy", "unhealthy"]
    message: str


class MediaToolDiagnostics(ApiModel):
    ffmpeg: DiagnosticCheck
    ffprobe: DiagnosticCheck


class ProviderDiagnostic(ApiModel):
    profile_id: str
    status: Literal["healthy", "unhealthy", "unconfigured"]
    message: str


class OutputDiagnostics(ApiModel):
    writable: bool
    free_bytes: int | None = Field(default=None, ge=0)
    message: str


class SystemDiagnostics(ApiModel):
    status: Literal["healthy", "degraded"]
    version: str = "0.1.0"
    media_tools: MediaToolDiagnostics
    providers: list[ProviderDiagnostic]
    output: OutputDiagnostics
    timestamp: datetime


class RetentionRequest(ApiModel):
    max_age_days: int | None = Field(default=None, ge=1, le=3650)
    max_jobs: int | None = Field(default=None, ge=1, le=10_000)


class RetentionResult(ApiModel):
    deleted_job_ids: list[str] = Field(default_factory=list)
    reclaimed_bytes: int = Field(default=0, ge=0)


def safe_error_code(error: BaseException) -> str:
    name = error.__class__.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower() or "pipeline_error"


def public_job_error(error: BaseException) -> JobError:
    """Return a useful failure without leaking URLs, paths, keys, or upstream bodies."""
    if isinstance(error, DependencyError):
        return JobError(
            code="dependency_error",
            message="Required local media tools are unavailable. Check the system setup.",
        )
    if isinstance(error, DurationLimitError):
        return JobError(
            code="duration_limit_error",
            message="The source video exceeds the configured duration limit.",
        )
    if isinstance(error, SourceTransferError):
        return JobError(
            code="source_transfer_error",
            message=(
                "YouTube temporarily rejected the media transfer. "
                "Please try this generation again."
            ),
        )
    if isinstance(error, DownloadError):
        return JobError(
            code="download_error",
            message="The source video could not be inspected or downloaded.",
        )
    if isinstance(error, TranscriptError):
        return JobError(
            code="transcript_error",
            message="A usable transcript could not be created for this video.",
        )
    if isinstance(error, AnalysisError):
        return JobError(
            code="analysis_error",
            message="The selected AI provider could not analyze this video.",
        )
    if isinstance(error, RenderError):
        return JobError(
            code="render_error",
            message="The selected clips could not be rendered.",
        )
    if isinstance(error, ClipperError):
        return JobError(
            code="pipeline_error",
            message="Clip generation could not be completed.",
        )
    return JobError(
        code=safe_error_code(error),
        message="Clip generation failed unexpectedly. Check the local service logs.",
    )


JsonObject = dict[str, Any]
