from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import yt_clipper.application.pipeline as pipeline_module
from yt_clipper import ClipPipeline, ContentType, PipelineConfig, WhisperDevice
from yt_clipper.domain.errors import AnalysisError, InsufficientHighlightsError
from yt_clipper.domain.models import (
    MONTAGE_ANALYSIS_PROMPT_VERSION,
    MONTAGE_ANALYSIS_SCHEMA_VERSION,
    ClipCandidate,
    DownloadedVideo,
    HighlightMoment,
    HighlightMontage,
    MontageAnalysisDocument,
    TranscriptDocument,
    TranscriptMode,
    TranscriptOrigin,
    TranscriptSegment,
    VideoMetadata,
)
from yt_clipper.infrastructure.media_tools import MediaTools


def test_pipeline_propagates_one_hour_resource_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="One-hour test",
        duration_seconds=3599,
    )
    observed: dict[str, Any] = {}

    class FakeDownloader:
        def inspect(
            self,
            url: str,
            cookies_from_browser: str | None,
            maximum_duration_seconds: int,
        ) -> VideoMetadata:
            observed["inspect"] = (
                url,
                cookies_from_browser,
                maximum_duration_seconds,
            )
            return metadata

        def download(
            self,
            inspected_metadata: VideoMetadata,
            output_root: Path,
            cookies_from_browser: str | None,
            force: bool,
            progress_hook: object,
            maximum_duration_seconds: int,
            maximum_download_bytes: int,
        ) -> DownloadedVideo:
            work_dir = output_root / inspected_metadata.video_id
            work_dir.mkdir(parents=True, exist_ok=True)
            observed["download"] = (
                maximum_duration_seconds,
                maximum_download_bytes,
                cookies_from_browser,
                force,
                callable(progress_hook),
            )
            return DownloadedVideo(
                metadata=inspected_metadata,
                video_path=work_dir / "source.mp4",
                metadata_path=work_dir / "metadata.json",
                work_dir=work_dir,
            )

    class FakeTranscriptService:
        def get_transcript(
            self,
            video: DownloadedVideo,
            **settings: object,
        ) -> tuple[TranscriptDocument, Path]:
            observed["transcript"] = settings
            transcript = TranscriptDocument(
                video_id=video.metadata.video_id,
                language="en",
                requested_language="en",
                origin=TranscriptOrigin.MANUAL,
                source_fingerprint="a" * 64,
                duration_seconds=video.metadata.duration_seconds,
                segments=[
                    TranscriptSegment(
                        start=0,
                        end=60,
                        text="A complete opening topic with a clear resolution.",
                    )
                ],
            )
            return transcript, video.work_dir / "transcript.json"

    class FakeAnalyzer:
        def find_clips(
            self,
            transcript: TranscriptDocument,
            **settings: object,
        ) -> list[ClipCandidate]:
            observed["analysis"] = settings
            assert transcript.duration_seconds == 3599
            return [
                ClipCandidate(
                    title="Complete topic",
                    start=0,
                    end=60,
                    score=0.9,
                    hook="A complete opening topic",
                    reason="The topic includes setup and resolution",
                    standalone=True,
                    topic="A complete topic",
                    opening_context="The subject is introduced immediately",
                    closing_resolution="The thought reaches a clear conclusion",
                )
            ]

    monkeypatch.setattr(pipeline_module, "validate_media_tools", lambda _tools: None)
    config = PipelineConfig(
        output_dir=tmp_path,
        analyze_only=True,
        max_source_duration_seconds=3599,
        max_source_download_bytes=3 * 1024**3,
        language="hi",
        transcript_mode=TranscriptMode.WHISPER,
        whisper_model="medium",
        whisper_device=WhisperDevice.CPU,
        whisper_cpu_threads=3,
        whisper_batch_size=2,
        whisper_chunk_seconds=240,
        whisper_chunk_overlap_seconds=7,
        whisper_timeout_seconds=1800,
        analysis_chunk_max_characters=30_000,
        analysis_chunk_overlap_seconds=75,
        analysis_max_concurrency=3,
        analysis_request_max_attempts=4,
        content_type=ContentType.COMEDY,
    )
    pipeline = ClipPipeline(
        config,
        media_tools=MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        downloader=FakeDownloader(),  # pyright: ignore[reportArgumentType]
        transcript_service=FakeTranscriptService(),  # pyright: ignore[reportArgumentType]
        analyzer=FakeAnalyzer(),  # pyright: ignore[reportArgumentType]
    )

    result = pipeline.run(metadata.source_url)

    assert observed["inspect"] == (metadata.source_url, None, 3599)
    assert observed["download"] == (3599, 3 * 1024**3, None, False, True)

    transcript_settings = observed["transcript"]
    assert isinstance(transcript_settings, dict)
    assert transcript_settings["language"] == "hi"
    assert transcript_settings["mode"] == TranscriptMode.WHISPER
    assert transcript_settings["whisper_model"] == "medium"
    assert transcript_settings["whisper_device"] == WhisperDevice.CPU
    assert transcript_settings["whisper_cpu_threads"] == 3
    assert transcript_settings["whisper_batch_size"] == 2
    assert transcript_settings["whisper_chunk_seconds"] == 240
    assert transcript_settings["whisper_chunk_overlap_seconds"] == 7
    assert transcript_settings["whisper_timeout_seconds"] == 1800

    analysis_settings = observed["analysis"]
    assert isinstance(analysis_settings, dict)
    assert analysis_settings["cache_dir"] == tmp_path / metadata.video_id / "analysis_chunks"
    assert analysis_settings["chunk_max_characters"] == 30_000
    assert analysis_settings["chunk_overlap_seconds"] == 75
    assert analysis_settings["max_concurrency"] == 3
    assert analysis_settings["request_max_attempts"] == 4
    assert analysis_settings["clip_count"] is None
    assert analysis_settings["content_type"] == ContentType.COMEDY

    saved_analysis = result.candidates_path.read_text(encoding="utf-8")
    assert '"content_type": "comedy"' in saved_analysis

    assert result.video.metadata.duration_seconds == 3599
    assert result.candidates[0].duration == 60
    assert result.clip_paths == []
    assert result.candidates_path.is_file()


def test_pipeline_passes_common_api_key_to_default_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeConfiguredAnalyzer:
        def __init__(
            self,
            *,
            api_key: str | None = None,
            provider: object = None,
            codex_binary: str = "codex",
            codex_timeout_seconds: int = 300,
        ) -> None:
            observed["api_key"] = api_key
            observed["provider"] = provider
            observed["codex_binary"] = codex_binary
            observed["codex_timeout_seconds"] = codex_timeout_seconds

    monkeypatch.setattr(pipeline_module, "TranscriptAnalyzer", FakeConfiguredAnalyzer)
    config = PipelineConfig.model_validate({"llm_api_key": "shared-test-key"})

    _ = ClipPipeline(
        config,
        media_tools=MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        downloader=object(),  # pyright: ignore[reportArgumentType]
        transcript_service=object(),  # pyright: ignore[reportArgumentType]
        renderer=object(),  # pyright: ignore[reportArgumentType]
    )

    assert observed["api_key"] == "shared-test-key"
    assert observed["provider"] == config.llm_provider
    assert observed["codex_binary"] == config.codex_binary
    assert observed["codex_timeout_seconds"] == config.codex_timeout_seconds


def test_pipeline_renders_every_candidate_in_automatic_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Automatic clip count",
        duration_seconds=90,
    )
    work_dir = tmp_path / metadata.video_id
    video = DownloadedVideo(
        metadata=metadata,
        video_path=work_dir / "source.mp4",
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )
    transcript = TranscriptDocument(
        video_id=metadata.video_id,
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=metadata.duration_seconds,
        segments=[
            TranscriptSegment(
                start=float(index * 30),
                end=float((index + 1) * 30),
                text=f"Complete high-quality topic {index}.",
            )
            for index in range(3)
        ],
    )
    candidates = [
        ClipCandidate(
            title=f"Clip {index}",
            start=float(index * 30),
            end=float((index + 1) * 30),
            score=0.9,
            hook=f"Topic {index}",
            reason="A complete high-quality topic",
            standalone=True,
            topic=f"Topic {index}",
            opening_context="The topic is introduced immediately.",
            closing_resolution="The topic reaches a clear conclusion.",
        )
        for index in range(3)
    ]
    rendered: list[tuple[int, str]] = []

    class FakeDownloader:
        def inspect(self, _url: str, **_settings: object) -> VideoMetadata:
            return metadata

        def download(self, _metadata: VideoMetadata, **_settings: object) -> DownloadedVideo:
            work_dir.mkdir(parents=True, exist_ok=True)
            return video

    class FakeTranscriptService:
        def get_transcript(
            self, _video: DownloadedVideo, **_settings: object
        ) -> tuple[TranscriptDocument, Path]:
            return transcript, work_dir / "transcript.json"

    class FakeAnalyzer:
        def find_clips(
            self, transcript: TranscriptDocument, **settings: object
        ) -> list[ClipCandidate]:
            assert transcript.video_id == metadata.video_id
            assert settings["clip_count"] is None
            return candidates

    class FakeRenderer:
        def render(
            self,
            *,
            candidate: ClipCandidate,
            output_dir: Path,
            index: int,
            **_settings: object,
        ) -> Path:
            rendered.append((index, candidate.title))
            return output_dir / f"{index:02d}.mp4"

        def cleanup_stale(
            self, _output_dir: Path, _retained_paths: list[Path]
        ) -> None:
            return None

    monkeypatch.setattr(pipeline_module, "validate_media_tools", lambda _tools: None)
    monkeypatch.setattr(
        pipeline_module,
        "validate_ffmpeg_capabilities",
        lambda _tools: None,
    )
    pipeline = ClipPipeline(
        PipelineConfig(output_dir=tmp_path),
        media_tools=MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        downloader=FakeDownloader(),  # pyright: ignore[reportArgumentType]
        transcript_service=FakeTranscriptService(),  # pyright: ignore[reportArgumentType]
        analyzer=FakeAnalyzer(),  # pyright: ignore[reportArgumentType]
        renderer=FakeRenderer(),  # pyright: ignore[reportArgumentType]
    )

    result = pipeline.run(metadata.source_url)

    assert rendered == [(1, "Clip 0"), (2, "Clip 1"), (3, "Clip 2")]
    assert len(result.clip_paths) == 3
    assert result.thumbnail_paths == [
        tmp_path / metadata.video_id / "clips" / f"{index:02d}.thumbnail.jpg"
        for index in range(1, 4)
    ]
    assert result.poster_paths == [
        tmp_path / metadata.video_id / "clips" / f"{index:02d}.poster.jpg"
        for index in range(1, 4)
    ]


def test_pipeline_generates_opt_in_whole_video_highlight_montage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Montage test",
        duration_seconds=60,
    )
    work_dir = tmp_path / metadata.video_id
    video = DownloadedVideo(
        metadata=metadata,
        video_path=work_dir / "source.mp4",
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )
    transcript = TranscriptDocument(
        video_id=metadata.video_id,
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=60,
        segments=[TranscriptSegment(start=0, end=60, text="All moments")],
    )
    candidate = ClipCandidate(
        title="Continuous",
        start=0,
        end=60,
        score=0.9,
        hook="Hook",
        reason="Reason",
        standalone=True,
        topic="Topic",
        opening_context="Opening",
        closing_resolution="Closing",
    )
    montage = HighlightMontage(
        title="Best of everything",
        summary="The strongest moments.",
        moments=[
            HighlightMoment(start=0, end=4, score=0.9, hook="A", reason="A"),
            HighlightMoment(start=20, end=24, score=0.8, hook="B", reason="B"),
        ],
    )
    observed: dict[str, object] = {}

    class FakeDownloader:
        def inspect(self, _url: str, **_settings: object) -> VideoMetadata:
            return metadata

        def download(self, _metadata: VideoMetadata, **_settings: object) -> DownloadedVideo:
            work_dir.mkdir(parents=True, exist_ok=True)
            return video

    class FakeTranscriptService:
        def get_transcript(
            self, _video: DownloadedVideo, **_settings: object
        ) -> tuple[TranscriptDocument, Path]:
            return transcript, work_dir / "transcript.json"

    class FakeAnalyzer:
        def find_clips(self, **_settings: object) -> list[ClipCandidate]:
            return [candidate]

        def find_highlight_montage(self, **settings: object) -> HighlightMontage:
            observed["montage_analysis"] = settings
            return montage

    class FakeRenderer:
        def render(self, *, output_dir: Path, **_settings: object) -> Path:
            return output_dir / "01-continuous.mp4"

        def render_montage(self, *, output_dir: Path, **settings: object) -> Path:
            observed["montage_render"] = settings
            return output_dir / "highlight-montage-best.mp4"

        def cleanup_stale(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(pipeline_module, "validate_media_tools", lambda _tools: None)
    monkeypatch.setattr(
        pipeline_module,
        "validate_ffmpeg_capabilities",
        lambda _tools: None,
    )
    pipeline = ClipPipeline(
        PipelineConfig(output_dir=tmp_path, highlight_montage=True),
        media_tools=MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        downloader=FakeDownloader(),  # pyright: ignore[reportArgumentType]
        transcript_service=FakeTranscriptService(),  # pyright: ignore[reportArgumentType]
        analyzer=FakeAnalyzer(),  # pyright: ignore[reportArgumentType]
        renderer=FakeRenderer(),  # pyright: ignore[reportArgumentType]
    )
    result = pipeline.run(metadata.source_url)

    analysis_settings = observed["montage_analysis"]
    assert isinstance(analysis_settings, dict)
    assert analysis_settings["window_seconds"] == 4
    assert analysis_settings["max_duration"] == 60
    assert result.montage == montage
    assert result.montage_analysis_path == work_dir / "montage.json"
    assert result.montage_analysis_path is not None
    assert result.montage_analysis_path.is_file()
    assert result.montage_video_path == work_dir / "montage" / "highlight-montage-best.mp4"
    assert result.montage_video_path is not None
    assert result.montage_thumbnail_path == result.montage_video_path.with_suffix(
        ".thumbnail.jpg"
    )


def test_montage_cache_requires_current_version_and_exact_highlight_windows(
    tmp_path: Path,
) -> None:
    transcript = TranscriptDocument(
        video_id="cache-video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=12,
        segments=[
            TranscriptSegment(
                start=0,
                end=12,
                text="one two three four five six seven eight nine ten eleven twelve",
            )
        ],
    )
    config = PipelineConfig(output_dir=tmp_path, highlight_montage=True)
    pipeline = object.__new__(ClipPipeline)
    pipeline.config = config

    class FakeAnalyzer:
        @staticmethod
        def analysis_backend_id(model: str) -> str:
            return model

    pipeline._analyzer = FakeAnalyzer()  # type: ignore[attr-defined]
    cache_path = tmp_path / "montage.json"

    def write_cache(
        moments: list[HighlightMoment],
        *,
        prompt_version: int = MONTAGE_ANALYSIS_PROMPT_VERSION,
    ) -> None:
        document = MontageAnalysisDocument(
            schema_version=MONTAGE_ANALYSIS_SCHEMA_VERSION,
            analysis_prompt_version=prompt_version,
            video_id=transcript.video_id,
            model=config.model,
            analysis_backend=config.model,
            content_type=config.content_type,
            window_seconds=config.highlight_window_seconds,
            max_duration=config.highlight_montage_max_duration,
            max_moments=config.highlight_montage_max_moments,
            batch_windows=config.highlight_analysis_batch_windows,
            transcript_sha256=pipeline._transcript_hash(transcript),
            montage=HighlightMontage(
                title="Cached montage",
                summary="Two cached moments.",
                moments=moments,
            ),
        )
        cache_path.write_text(document.model_dump_json(), encoding="utf-8")

    exact_moments = [
        HighlightMoment(start=0, end=4, score=0.9, hook="A", reason="A"),
        HighlightMoment(start=8, end=12, score=0.8, hook="B", reason="B"),
    ]
    write_cache(exact_moments)
    assert pipeline._load_cached_montage(cache_path, transcript) is not None

    arbitrary_moments = [
        HighlightMoment(start=1, end=4, score=0.9, hook="A", reason="A"),
        exact_moments[1],
    ]
    write_cache(arbitrary_moments)
    assert pipeline._load_cached_montage(cache_path, transcript) is None

    write_cache(
        exact_moments,
        prompt_version=MONTAGE_ANALYSIS_PROMPT_VERSION + 1,
    )
    assert pipeline._load_cached_montage(cache_path, transcript) is None


@pytest.mark.parametrize(
    ("montage_error", "is_expected_skip"),
    [
        (InsufficientHighlightsError("not enough strong moments"), True),
        (AnalysisError("provider response failed"), False),
    ],
)
def test_pipeline_only_skips_expected_insufficient_highlights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    montage_error: AnalysisError,
    is_expected_skip: bool,
) -> None:
    metadata = VideoMetadata(
        video_id="montage-skip",
        source_url="https://www.youtube.com/watch?v=montage-skip",
        title="Montage skip test",
        duration_seconds=60,
    )
    work_dir = tmp_path / metadata.video_id
    video = DownloadedVideo(
        metadata=metadata,
        video_path=work_dir / "source.mp4",
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )
    transcript = TranscriptDocument(
        video_id=metadata.video_id,
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=60,
        segments=[TranscriptSegment(start=0, end=60, text="Continuous transcript")],
    )
    candidate = ClipCandidate(
        title="Continuous clip",
        start=0,
        end=60,
        score=0.9,
        hook="Hook",
        reason="Reason",
        standalone=True,
        topic="Topic",
        opening_context="Opening",
        closing_resolution="Closing",
    )
    rendered: list[str] = []
    cleanup_calls: list[tuple[Path, list[Path]]] = []

    class FakeDownloader:
        def inspect(self, _url: str, **_settings: object) -> VideoMetadata:
            return metadata

        def download(self, _metadata: VideoMetadata, **_settings: object) -> DownloadedVideo:
            work_dir.mkdir(parents=True, exist_ok=True)
            return video

    class FakeTranscriptService:
        def get_transcript(
            self, _video: DownloadedVideo, **_settings: object
        ) -> tuple[TranscriptDocument, Path]:
            return transcript, work_dir / "transcript.json"

    class FakeAnalyzer:
        def find_clips(self, **_settings: object) -> list[ClipCandidate]:
            return [candidate]

        def find_highlight_montage(self, **_settings: object) -> HighlightMontage:
            raise montage_error

    class FakeRenderer:
        def render(self, *, output_dir: Path, **_settings: object) -> Path:
            rendered.append("continuous")
            return output_dir / "01-continuous.mp4"

        def cleanup_stale(
            self, output_dir: Path, retained_paths: list[Path]
        ) -> None:
            cleanup_calls.append((output_dir, retained_paths))

    monkeypatch.setattr(pipeline_module, "validate_media_tools", lambda _tools: None)
    monkeypatch.setattr(
        pipeline_module,
        "validate_ffmpeg_capabilities",
        lambda _tools: None,
    )
    pipeline = ClipPipeline(
        PipelineConfig(output_dir=tmp_path, highlight_montage=True),
        media_tools=MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        downloader=FakeDownloader(),  # pyright: ignore[reportArgumentType]
        transcript_service=FakeTranscriptService(),  # pyright: ignore[reportArgumentType]
        analyzer=FakeAnalyzer(),  # pyright: ignore[reportArgumentType]
        renderer=FakeRenderer(),  # pyright: ignore[reportArgumentType]
    )
    stale_montage_cache = work_dir / "montage.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    stale_montage_cache.write_text("not valid montage JSON", encoding="utf-8")

    if not is_expected_skip:
        with pytest.raises(AnalysisError, match="provider response failed"):
            pipeline.run(metadata.source_url)
        assert rendered == []
        return

    result = pipeline.run(metadata.source_url)
    assert rendered == ["continuous"]
    assert len(result.clip_paths) == 1
    assert result.montage is None
    assert result.montage_analysis_path is None
    assert result.montage_video_path is None
    assert not stale_montage_cache.exists()
    assert cleanup_calls[-1] == (work_dir / "montage", [])
