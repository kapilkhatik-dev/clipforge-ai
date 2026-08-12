from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import yt_clipper.application.pipeline as pipeline_module
from yt_clipper import ClipPipeline, ContentType, PipelineConfig, WhisperDevice
from yt_clipper.domain.models import (
    ClipCandidate,
    DownloadedVideo,
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
