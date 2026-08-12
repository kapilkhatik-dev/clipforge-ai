from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from yt_clipper.domain.errors import TranscriptError
from yt_clipper.domain.models import (
    DownloadedVideo,
    TranscriptOrigin,
    TranscriptSegment,
    TranscriptMode,
    VideoMetadata,
    WhisperDevice,
)
from yt_clipper.infrastructure.media_tools import MediaTools
from yt_clipper.services import transcript as transcript_module
from yt_clipper.services.transcript import (
    TranscriptService,
    _language_candidates,
    audio_window_bounds,
    normalize_segments,
    parse_json3_captions,
    parse_vtt_or_srt_captions,
)


def _downloaded_video(tmp_path: Path, duration_seconds: float = 10.0) -> DownloadedVideo:
    work_dir = tmp_path / "output" / "video"
    work_dir.mkdir(parents=True)
    return DownloadedVideo(
        metadata=VideoMetadata(
            video_id="video",
            source_url="https://www.youtube.com/watch?v=video",
            title="Video",
            duration_seconds=duration_seconds,
        ),
        video_path=work_dir / "source.mp4",
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )


def test_one_hour_audio_is_planned_in_bounded_windows() -> None:
    core_start = 0.0
    windows = []
    while core_start < 3600:
        core_end, window_start, window_end = audio_window_bounds(
            core_start,
            audio_duration=3600,
            chunk_seconds=300,
            overlap_seconds=5,
        )
        windows.append((core_start, core_end, window_start, window_end))
        core_start = core_end

    assert len(windows) == 12
    assert max(window_end - window_start for _, _, window_start, window_end in windows) <= 310
    assert windows[-1][1] == 3600


def test_auto_language_candidates_are_deterministic_real_tracks() -> None:
    tracks = {"fr": [], "EN": [], "es-orig": []}

    assert _language_candidates(tracks, "AUTO") == ["es-orig", "EN", "fr"]


def test_auto_language_prefers_manual_captions_and_bounds_response_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened_urls: list[str] = []
    read_sizes: list[int] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nManual caption\n"

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            _ = options

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            _ = (url, download)
            return {
                "subtitles": {
                    "fr": [{"ext": "vtt", "url": "manual-fr"}],
                },
                "automatic_captions": {
                    "en-orig": [{"ext": "vtt", "url": "automatic-en"}],
                },
            }

        def urlopen(self, url: str) -> FakeResponse:
            opened_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr(transcript_module, "YoutubeDL", FakeYoutubeDL)
    service = TranscriptService(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    transcript = service._from_youtube_captions(
        _downloaded_video(tmp_path),
        language="auto",
        source_fingerprint="a" * 64,
        cookies_from_browser=None,
    )

    assert transcript is not None
    assert transcript.origin == TranscriptOrigin.MANUAL
    assert transcript.language == "fr"
    assert opened_urls == ["manual-fr"]
    assert read_sizes == [transcript_module._MAX_CAPTION_RESPONSE_BYTES + 1]


def test_oversized_caption_response_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    limit = 1024 * 1024
    read_sizes: list[int] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return b"x" * size

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, object]) -> None:
            _ = options

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            _ = (url, download)
            return {
                "subtitles": {
                    "en": [{"ext": "vtt", "url": "oversized"}],
                }
            }

        def urlopen(self, url: str) -> FakeResponse:
            assert url == "oversized"
            return FakeResponse()

    monkeypatch.setattr(transcript_module, "_MAX_CAPTION_RESPONSE_BYTES", limit)
    monkeypatch.setattr(transcript_module, "YoutubeDL", FakeYoutubeDL)
    service = TranscriptService(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    with pytest.raises(TranscriptError, match="use local Whisper transcription"):
        service._from_youtube_captions(
            _downloaded_video(tmp_path),
            language="en",
            source_fingerprint="a" * 64,
            cookies_from_browser=None,
        )

    assert read_sizes == [limit + 1]


def test_local_video_uses_matching_automatic_sidecar_captions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_source = tmp_path / "Comedy [local].mp4"
    local_source.write_bytes(b"original-local-video")
    sidecar = tmp_path / "Comedy [local].en-auto.srt"
    sidecar.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nA complete funny setup.\n\n"
        "2\n00:00:04,000 --> 00:00:08,000\nAnd this is the punchline.\n",
        encoding="utf-8",
    )
    work_dir = tmp_path / "output" / "local-video"
    work_dir.mkdir(parents=True)
    staged_source = work_dir / "source.mp4"
    staged_source.write_bytes(local_source.read_bytes())
    video = DownloadedVideo(
        metadata=VideoMetadata(
            video_id="local-video",
            source_url=str(local_source),
            title="Comedy",
            duration_seconds=10,
        ),
        video_path=staged_source,
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )
    service = TranscriptService(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )
    monkeypatch.setattr(
        service,
        "_from_whisper",
        lambda **_kwargs: pytest.fail("A valid sidecar must avoid Whisper"),
    )

    transcript, transcript_path = service.get_transcript(
        video,
        language="en",
        mode=TranscriptMode.CAPTIONS,
    )

    assert transcript.origin == TranscriptOrigin.AUTOMATIC
    assert transcript.language == "en-auto"
    assert [segment.text for segment in transcript.segments] == [
        "A complete funny setup.",
        "And this is the punchline.",
    ]
    assert transcript_path.is_file()


def test_whisper_lock_covers_pcm_extraction_and_transcription(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    lock_state = {"held": False}
    video = _downloaded_video(tmp_path, duration_seconds=1.0)

    class FakeFileLock:
        def __init__(self, path: str, timeout: float) -> None:
            assert Path(path) == video.work_dir.parent / ".whisper.lock"
            assert timeout > 0

        def __enter__(self) -> "FakeFileLock":
            lock_state["held"] = True
            events.append("lock-enter")
            return self

        def __exit__(self, *args: object) -> None:
            _ = args
            events.append("lock-exit")
            lock_state["held"] = False

    def fake_extract_mono_pcm(
        tools: MediaTools,
        source_path: Path,
        destination_path: Path,
        timeout_seconds: float,
        max_duration_seconds: float,
    ) -> int:
        _ = (tools, source_path)
        assert lock_state["held"]
        assert timeout_seconds > 0
        assert max_duration_seconds == video.metadata.duration_seconds
        events.append("extract")
        _ = destination_path.write_bytes(b"\x00\x00" * 16_000)
        return 16_000

    class FakeWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = (args, kwargs)
            assert lock_state["held"]
            events.append("model")

        def transcribe(self, audio: object, **kwargs: object) -> tuple[list[object], object]:
            _ = (audio, kwargs)
            assert lock_state["held"]
            events.append("transcribe")
            segment = SimpleNamespace(start=0.0, end=0.5, text="Hello", words=[])
            return [segment], SimpleNamespace(language="en")

    class UnexpectedBatchedPipeline:
        def __init__(self, model: object) -> None:
            _ = model
            raise AssertionError("Batch pipeline should not be used")

    fake_ctranslate2 = ModuleType("ctranslate2")
    setattr(fake_ctranslate2, "get_cuda_device_count", lambda: 0)
    fake_faster_whisper = ModuleType("faster_whisper")
    setattr(fake_faster_whisper, "WhisperModel", FakeWhisperModel)
    setattr(
        fake_faster_whisper,
        "BatchedInferencePipeline",
        UnexpectedBatchedPipeline,
    )

    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster_whisper)
    monkeypatch.setattr(transcript_module, "FileLock", FakeFileLock)
    monkeypatch.setattr(transcript_module, "extract_mono_pcm", fake_extract_mono_pcm)

    service = TranscriptService(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )
    transcript = service._from_whisper(
        video=video,
        language="auto",
        model_name="small",
        configured_device=WhisperDevice.CPU,
        cpu_threads=1,
        batch_size=1,
        chunk_seconds=300,
        overlap_seconds=5,
        timeout_seconds=30,
        source_fingerprint="a" * 64,
        options_fingerprint="b" * 64,
        progress_callback=None,
    )

    assert transcript.origin == TranscriptOrigin.WHISPER
    assert events == ["lock-enter", "extract", "model", "transcribe", "lock-exit"]
    assert not lock_state["held"]
    assert not list((video.work_dir / "temp").glob("whisper-*"))


def test_parses_json3_segments_and_word_offsets() -> None:
    payload = json.dumps(
        {
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 2000,
                    "segs": [
                        {"utf8": "Hello ", "tOffsetMs": 0},
                        {"utf8": "world", "tOffsetMs": 900},
                    ],
                }
            ]
        }
    )

    segments = parse_json3_captions(payload)

    assert len(segments) == 1
    assert segments[0].start == 1.0
    assert segments[0].end == 3.0
    assert segments[0].text == "Hello world"
    assert [(word.start, word.end) for word in segments[0].words] == [
        (1.0, 1.9),
        (1.9, 3.0),
    ]


def test_parses_webvtt_and_strips_markup() -> None:
    payload = """WEBVTT

00:00:01.000 --> 00:00:03.500 align:start
<c.yellow>Hello &amp; welcome</c>

00:03.500 --> 00:05.000
Next line
"""

    segments = parse_vtt_or_srt_captions(payload)

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (1.0, 3.5, "Hello & welcome"),
        (3.5, 5.0, "Next line"),
    ]


def test_normalization_clamps_and_deduplicates() -> None:
    duplicate = TranscriptSegment(start=1, end=4, text=" Same   text ")
    segments = normalize_segments([duplicate, duplicate, TranscriptSegment(start=4, end=8, text="End")], 6)

    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (1.0, 4.0, "Same text"),
        (4.0, 6.0, "End"),
    ]
