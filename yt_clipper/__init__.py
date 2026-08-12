"""Public, framework-neutral API for the YouTube clipper pipeline."""

from .application.pipeline import ClipPipeline
from .config import (
    DEFAULT_ANALYSIS_MODEL,
    ContentType,
    LLMProvider,
    resolve_analysis_model,
)
from .domain.errors import (
    AnalysisError,
    ClipperError,
    DependencyError,
    DownloadError,
    DurationLimitError,
    MediaExtractionError,
    MediaProbeError,
    RenderError,
    TranscriptError,
)
from .domain.models import (
    ClipCandidate,
    PipelineConfig,
    PipelineEvent,
    PipelineResult,
    PipelineStage,
    ProgressCallback,
    TranscriptDocument,
    TranscriptMode,
    TranscriptOrigin,
    VideoLayout,
    VideoMetadata,
    WhisperDevice,
)

__all__ = [
    "DEFAULT_ANALYSIS_MODEL",
    "AnalysisError",
    "ClipCandidate",
    "ClipPipeline",
    "ClipperError",
    "ContentType",
    "DependencyError",
    "DownloadError",
    "DurationLimitError",
    "MediaExtractionError",
    "MediaProbeError",
    "LLMProvider",
    "PipelineConfig",
    "PipelineEvent",
    "PipelineResult",
    "PipelineStage",
    "ProgressCallback",
    "RenderError",
    "TranscriptDocument",
    "TranscriptError",
    "TranscriptMode",
    "TranscriptOrigin",
    "VideoLayout",
    "VideoMetadata",
    "WhisperDevice",
    "resolve_analysis_model",
]
