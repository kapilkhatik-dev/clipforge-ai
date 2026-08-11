"""Application-specific exceptions with user-facing messages."""


class ClipperError(Exception):
    """Base class for expected pipeline failures."""


class DependencyError(ClipperError):
    """A required local executable or Python capability is unavailable."""


class DownloadError(ClipperError):
    """Video inspection or download failed."""


class DurationLimitError(DownloadError):
    """The source video exceeds the supported duration."""


class MediaProbeError(ClipperError):
    """FFprobe could not validate a media artifact."""


class MediaExtractionError(ClipperError):
    """FFmpeg could not extract a bounded audio artifact."""


class TranscriptError(ClipperError):
    """No usable transcript could be obtained."""


class AnalysisError(ClipperError):
    """The language-model analysis failed or returned unusable data."""


class RenderError(ClipperError):
    """FFmpeg failed to render a clip."""
