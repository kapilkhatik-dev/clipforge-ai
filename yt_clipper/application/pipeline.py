"""Framework-neutral orchestration for scripts and future UI adapters."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from ..domain.errors import AnalysisError, ClipperError, DownloadError
from ..domain.models import (
    ANALYSIS_PROMPT_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    AnalysisDocument,
    HighlightMontage,
    MontageAnalysisDocument,
    PipelineConfig,
    PipelineEvent,
    PipelineResult,
    PipelineStage,
    ProgressCallback,
    TranscriptDocument,
    VideoMetadata,
)
from ..infrastructure.artifacts import atomic_write_text
from ..infrastructure.media_tools import (
    MediaTools,
    resolve_media_tools,
    validate_ffmpeg_capabilities,
    validate_media_tools,
)
from ..services.analyzer import TranscriptAnalyzer, validate_clip_candidates
from ..services.downloader import VideoDownloader
from ..services.renderer import ClipRenderer
from ..services.transcript import TranscriptService

LOGGER = logging.getLogger(__name__)


class ClipPipeline:
    """Reusable application service with no console or web-framework coupling."""

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: ProgressCallback | None = None,
        media_tools: MediaTools | None = None,
        downloader: VideoDownloader | None = None,
        transcript_service: TranscriptService | None = None,
        analyzer: TranscriptAnalyzer | None = None,
        renderer: ClipRenderer | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback
        self._media_tools = media_tools or resolve_media_tools()
        self._downloader = downloader or VideoDownloader(self._media_tools)
        self._transcript_service = transcript_service or TranscriptService(
            self._media_tools
        )
        self._analyzer = analyzer or TranscriptAnalyzer(
            api_key=config.get_llm_api_key(),
            provider=config.llm_provider,
            codex_binary=config.codex_binary,
            codex_timeout_seconds=config.codex_timeout_seconds,
        )
        self._renderer = renderer or ClipRenderer(self._media_tools)

    def _emit(
        self,
        stage: PipelineStage,
        message: str,
        progress: float | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        if not self._progress_callback:
            return
        try:
            self._progress_callback(
                PipelineEvent(
                    stage=stage,
                    message=message,
                    progress=progress,
                    current=current,
                    total=total,
                )
            )
        except Exception:
            LOGGER.warning("Progress callback failed", exc_info=True)

    def _download_progress(self, status: dict[str, Any]) -> None:
        if status.get("status") != "downloading":
            return
        downloaded = float(status.get("downloaded_bytes") or 0)
        total = float(status.get("total_bytes") or status.get("total_bytes_estimate") or 0)
        progress = downloaded / total if total > 0 else None
        self._emit(PipelineStage.DOWNLOAD, "Downloading source video", progress=progress)

    def _transcript_progress(self, progress: float) -> None:
        self._emit(
            PipelineStage.TRANSCRIBE,
            "Transcribing local audio",
            progress=progress,
        )

    def _analysis_chunk_progress(self, current: int, total: int) -> None:
        progress = 0.8 * current / total if total else 0.0
        self._emit(
            PipelineStage.ANALYZE,
            f"Analyzed transcript chunk {current} of {total}",
            progress=progress,
            current=current,
            total=total,
        )

    def _montage_analysis_progress(self, current: int, total: int) -> None:
        progress = current / total if total else 0.0
        self._emit(
            PipelineStage.ANALYZE,
            f"Screened highlight batch {current} of {total}",
            progress=progress,
            current=current,
            total=total,
        )

    @staticmethod
    def _transcript_hash(transcript: TranscriptDocument) -> str:
        serialized = transcript.model_dump_json(exclude_none=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _analysis_backend_id(self) -> str:
        identity = getattr(self._analyzer, "analysis_backend_id", None)
        if callable(identity):
            return str(identity(self.config.model))
        return self.config.model

    def _load_cached_analysis(
        self,
        path: Path,
        transcript: TranscriptDocument,
    ) -> AnalysisDocument | None:
        if not path.is_file() or self.config.force:
            return None
        try:
            cached = AnalysisDocument.model_validate_json(path.read_text(encoding="utf-8"))
            expected_hash = self._transcript_hash(transcript)
            if not (
                cached.schema_version == ANALYSIS_SCHEMA_VERSION
                and cached.video_id == transcript.video_id
                and cached.model == self.config.model
                and (cached.analysis_backend or cached.model)
                == self._analysis_backend_id()
                and cached.clip_count == self.config.clip_count
                and cached.content_type == self.config.content_type
                and cached.min_clip_duration == self.config.min_clip_duration
                and cached.max_clip_duration == self.config.max_clip_duration
                and cached.analysis_prompt_version == ANALYSIS_PROMPT_VERSION
                and cached.chunk_max_characters
                == self.config.analysis_chunk_max_characters
                and cached.chunk_overlap_seconds
                == self.config.analysis_chunk_overlap_seconds
                and cached.transcript_origin == transcript.origin
                and cached.transcript_sha256 == expected_hash
            ):
                return None
            validated = validate_clip_candidates(
                [candidate.model_dump() for candidate in cached.candidates],
                video_duration=transcript.duration_seconds,
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
                transcript_segments=transcript.segments,
                require_segment_boundaries=True,
                require_standalone=True,
            )
            if (
                len(validated) != len(cached.candidates)
                or (
                    self.config.clip_count is not None
                    and len(validated) > self.config.clip_count
                )
            ):
                return None
            return cached.model_copy(update={"candidates": validated})
        except (AnalysisError, ValueError, OSError):
            LOGGER.warning("Ignoring invalid analysis cache at %s", path)
            return None

    def _load_cached_montage(
        self,
        path: Path,
        transcript: TranscriptDocument,
    ) -> HighlightMontage | None:
        if not path.is_file() or self.config.force:
            return None
        try:
            cached = MontageAnalysisDocument.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if not (
                cached.schema_version == 1
                and cached.video_id == transcript.video_id
                and cached.model == self.config.model
                and cached.analysis_backend == self._analysis_backend_id()
                and cached.content_type == self.config.content_type
                and cached.window_seconds == self.config.highlight_window_seconds
                and cached.max_duration
                == self.config.highlight_montage_max_duration
                and cached.max_moments == self.config.highlight_montage_max_moments
                and cached.batch_windows
                == self.config.highlight_analysis_batch_windows
                and cached.transcript_sha256 == self._transcript_hash(transcript)
                and cached.montage.duration
                <= self.config.highlight_montage_max_duration + 0.01
                and len(cached.montage.moments)
                <= self.config.highlight_montage_max_moments
                and all(
                    moment.end <= transcript.duration_seconds + 0.01
                    and 0 < moment.duration
                    <= self.config.highlight_window_seconds + 0.01
                    for moment in cached.montage.moments
                )
            ):
                return None
            return cached.montage
        except (ValueError, OSError):
            LOGGER.warning("Ignoring invalid montage analysis cache at %s", path)
            return None

    def run(self, url: str) -> PipelineResult:
        self._emit(PipelineStage.SETUP, "Checking local media tools", progress=0)
        validate_media_tools(self._media_tools)
        if not self.config.analyze_only:
            validate_ffmpeg_capabilities(self._media_tools)
        output_root = self.config.output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        self._emit(PipelineStage.INSPECT, "Inspecting video metadata", progress=0)
        metadata = self._downloader.inspect(
            url,
            cookies_from_browser=self.config.cookies_from_browser,
            maximum_duration_seconds=self.config.max_source_duration_seconds,
        )
        if metadata.duration_seconds < self.config.min_clip_duration:
            raise DownloadError(
                f"Video is only {metadata.duration_seconds:.1f} seconds long, shorter "
                f"than the {self.config.min_clip_duration:.1f}-second clip minimum."
            )
        self._emit(
            PipelineStage.INSPECT,
            f"Accepted {metadata.duration_seconds / 60:.1f}-minute video",
            progress=1,
        )

        work_dir = output_root / metadata.video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(work_dir / ".pipeline.lock"), timeout=0):
                return self._run_locked(metadata)
        except Timeout as exc:
            raise ClipperError(
                "Another clipper job is already processing this video."
            ) from exc

    def _run_locked(self, metadata: VideoMetadata) -> PipelineResult:
        self._emit(PipelineStage.DOWNLOAD, "Preparing source download", progress=0)
        video = self._downloader.download(
            metadata,
            output_root=self.config.output_dir,
            cookies_from_browser=self.config.cookies_from_browser,
            force=self.config.force,
            progress_hook=self._download_progress,
            maximum_duration_seconds=self.config.max_source_duration_seconds,
            maximum_download_bytes=self.config.max_source_download_bytes,
        )
        self._emit(PipelineStage.DOWNLOAD, "Source video ready", progress=1)

        self._emit(PipelineStage.TRANSCRIBE, "Obtaining timestamped transcript", progress=0)
        transcript, transcript_path = self._transcript_service.get_transcript(
            video,
            language=self.config.language,
            mode=self.config.transcript_mode,
            whisper_model=self.config.whisper_model,
            whisper_device=self.config.whisper_device,
            whisper_cpu_threads=self.config.whisper_cpu_threads,
            whisper_batch_size=self.config.whisper_batch_size,
            whisper_chunk_seconds=self.config.whisper_chunk_seconds,
            whisper_chunk_overlap_seconds=self.config.whisper_chunk_overlap_seconds,
            whisper_timeout_seconds=self.config.whisper_timeout_seconds,
            cookies_from_browser=self.config.cookies_from_browser,
            force=self.config.force,
            progress_callback=self._transcript_progress,
        )
        self._emit(
            PipelineStage.TRANSCRIBE,
            f"Transcript ready from {transcript.origin.value}",
            progress=1,
        )

        candidates_path = video.work_dir / "candidates.json"
        analysis = self._load_cached_analysis(candidates_path, transcript)
        if analysis:
            candidates = analysis.candidates
            self._emit(PipelineStage.ANALYZE, "Using cached clip analysis", progress=1)
        else:
            self._emit(
                PipelineStage.ANALYZE,
                "Asking the model to select clips "
                + f"({self.config.content_type.value} focus)",
                progress=0,
            )
            candidates = self._analyzer.find_clips(
                transcript=transcript,
                model=self.config.model,
                clip_count=self.config.clip_count,
                content_type=self.config.content_type,
                min_duration=self.config.min_clip_duration,
                max_duration=self.config.max_clip_duration,
                cache_dir=video.work_dir / "analysis_chunks",
                force=self.config.force,
                chunk_max_characters=self.config.analysis_chunk_max_characters,
                chunk_overlap_seconds=self.config.analysis_chunk_overlap_seconds,
                max_concurrency=self.config.analysis_max_concurrency,
                request_max_attempts=self.config.analysis_request_max_attempts,
                progress_callback=self._analysis_chunk_progress,
            )
            analysis = AnalysisDocument(
                schema_version=ANALYSIS_SCHEMA_VERSION,
                video_id=metadata.video_id,
                model=self.config.model,
                analysis_backend=self._analysis_backend_id(),
                clip_count=self.config.clip_count,
                content_type=self.config.content_type,
                min_clip_duration=self.config.min_clip_duration,
                max_clip_duration=self.config.max_clip_duration,
                analysis_prompt_version=ANALYSIS_PROMPT_VERSION,
                chunk_max_characters=self.config.analysis_chunk_max_characters,
                chunk_overlap_seconds=self.config.analysis_chunk_overlap_seconds,
                transcript_origin=transcript.origin,
                transcript_sha256=self._transcript_hash(transcript),
                candidates=candidates,
            )
            atomic_write_text(candidates_path, analysis.model_dump_json(indent=2))
            self._emit(PipelineStage.ANALYZE, "Clip selections ready", progress=1)

        montage: HighlightMontage | None = None
        montage_analysis_path: Path | None = None
        if self.config.highlight_montage:
            montage_analysis_path = video.work_dir / "montage.json"
            montage = self._load_cached_montage(montage_analysis_path, transcript)
            if montage is None:
                self._emit(
                    PipelineStage.ANALYZE,
                    "Screening short moments for a whole-video highlight montage",
                    progress=0,
                )
                montage = self._analyzer.find_highlight_montage(
                    transcript=transcript,
                    model=self.config.model,
                    content_type=self.config.content_type,
                    window_seconds=self.config.highlight_window_seconds,
                    max_duration=self.config.highlight_montage_max_duration,
                    max_moments=self.config.highlight_montage_max_moments,
                    batch_windows=self.config.highlight_analysis_batch_windows,
                    request_max_attempts=self.config.analysis_request_max_attempts,
                    progress_callback=self._montage_analysis_progress,
                )
                atomic_write_text(
                    montage_analysis_path,
                    MontageAnalysisDocument(
                        schema_version=1,
                        video_id=transcript.video_id,
                        model=self.config.model,
                        analysis_backend=self._analysis_backend_id(),
                        content_type=self.config.content_type,
                        window_seconds=self.config.highlight_window_seconds,
                        max_duration=self.config.highlight_montage_max_duration,
                        max_moments=self.config.highlight_montage_max_moments,
                        batch_windows=self.config.highlight_analysis_batch_windows,
                        transcript_sha256=self._transcript_hash(transcript),
                        montage=montage,
                    ).model_dump_json(indent=2),
                )
            else:
                self._emit(
                    PipelineStage.ANALYZE,
                    "Using cached highlight montage analysis",
                    progress=1,
                )

        clip_paths = []
        thumbnail_paths = []
        poster_paths = []
        montage_video_path: Path | None = None
        montage_thumbnail_path: Path | None = None
        montage_poster_path: Path | None = None
        if not self.config.analyze_only:
            clips_dir = video.work_dir / "clips"
            total = len(candidates)
            for index, candidate in enumerate(candidates, start=1):
                self._emit(
                    PipelineStage.RENDER,
                    f"Rendering clip {index} of {total}: {candidate.title}",
                    progress=(index - 1) / total,
                    current=index,
                    total=total,
                )
                rendered_path = self._renderer.render(
                    video=video,
                    transcript=transcript,
                    candidate=candidate,
                    output_dir=clips_dir,
                    index=index,
                    video_layout=self.config.video_layout,
                    force=self.config.force,
                )
                clip_paths.append(rendered_path)
                thumbnail_paths.append(rendered_path.with_suffix(".thumbnail.jpg"))
                poster_paths.append(rendered_path.with_suffix(".poster.jpg"))
            self._renderer.cleanup_stale(clips_dir, clip_paths)
            self._emit(
                PipelineStage.RENDER,
                f"Rendered {total} clip{'s' if total != 1 else ''}",
                progress=1,
                current=total,
                total=total,
            )
            if montage is not None:
                self._emit(
                    PipelineStage.RENDER,
                    f"Rendering highlight montage: {montage.title}",
                    progress=0,
                )
                montage_video_path = self._renderer.render_montage(
                    video=video,
                    transcript=transcript,
                    montage=montage,
                    output_dir=video.work_dir / "montage",
                    video_layout=self.config.video_layout,
                    force=self.config.force,
                )
                self._renderer.cleanup_stale(
                    montage_video_path.parent,
                    [montage_video_path],
                )
                montage_thumbnail_path = montage_video_path.with_suffix(
                    ".thumbnail.jpg"
                )
                montage_poster_path = montage_video_path.with_suffix(".poster.jpg")
                self._emit(
                    PipelineStage.RENDER,
                    "Highlight montage ready",
                    progress=1,
                )

        result = PipelineResult(
            video=video,
            transcript=transcript,
            candidates=candidates,
            transcript_path=transcript_path,
            candidates_path=candidates_path,
            clip_paths=clip_paths,
            thumbnail_paths=thumbnail_paths,
            poster_paths=poster_paths,
            montage=montage,
            montage_analysis_path=montage_analysis_path,
            montage_video_path=montage_video_path,
            montage_thumbnail_path=montage_thumbnail_path,
            montage_poster_path=montage_poster_path,
        )
        self._emit(PipelineStage.COMPLETE, "Pipeline complete", progress=1)
        return result
