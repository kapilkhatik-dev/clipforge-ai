from __future__ import annotations

import math
from pathlib import Path

import pytest
from yt_dlp.utils import RejectedVideoReached

import yt_clipper.services.downloader as downloader_module
from yt_clipper.domain.errors import DownloadError, DurationLimitError
from yt_clipper.domain.models import VideoMetadata
from yt_clipper.infrastructure.media_tools import MediaProbe, MediaTools
from yt_clipper.services.downloader import (
    VideoDownloader,
    parse_cookies_from_browser,
    validate_source_duration,
)


def test_accepts_video_at_one_hour_limit() -> None:
    assert validate_source_duration(3600) == 3600.0


def test_rejects_video_over_one_hour() -> None:
    with pytest.raises(DurationLimitError, match="current limit is 60 minutes"):
        validate_source_duration(3600.01)


@pytest.mark.parametrize("duration", [None, math.nan, math.inf, -math.inf])
def test_rejects_invalid_duration(duration: float | None) -> None:
    with pytest.raises(DownloadError, match="valid video duration"):
        validate_source_duration(duration)


def test_parses_browser_cookie_spec() -> None:
    assert parse_cookies_from_browser("Chrome+basictext:Profile 1::work") == (
        "chrome",
        "Profile 1",
        "BASICTEXT",
        "work",
    )


def test_rejects_unsupported_cookie_browser() -> None:
    with pytest.raises(DownloadError, match="Unsupported cookie browser"):
        parse_cookies_from_browser("unknown-browser")


def test_inspects_and_stages_local_mp4_without_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "Funny local video.mp4"
    source.write_bytes(b"local-video-content")
    probe = MediaProbe(duration_seconds=120, has_video=True, has_audio=True)
    monkeypatch.setattr(downloader_module, "probe_media", lambda *_args: probe)

    class FailYoutubeDL:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("Local files must not call yt-dlp")

    monkeypatch.setattr(downloader_module, "YoutubeDL", FailYoutubeDL)
    downloader = VideoDownloader(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    metadata = downloader.inspect(str(source))
    staged = downloader.download(metadata, tmp_path / "output")

    assert metadata.video_id.startswith("local-")
    assert metadata.source_url == str(source.resolve())
    assert metadata.title == "Funny local video"
    assert staged.video_path.read_bytes() == source.read_bytes()
    assert staged.video_path.name == "source.mp4"
    assert staged.metadata_path.is_file()


def test_download_filter_rejects_changed_over_limit_video_before_media_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeYoutubeDL:
        def __init__(self, params):
            self.params = params

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, _url: str, download: bool):
            assert download is True
            assert self.params["max_filesize"] == 4 * 1024**3
            assert self.params["concurrent_fragment_downloads"] == 2
            rejection = self.params["match_filter"](
                {"duration": 3601}, incomplete=False
            )
            assert rejection
            raise RejectedVideoReached(rejection)

    monkeypatch.setattr(downloader_module, "YoutubeDL", FakeYoutubeDL)
    downloader = VideoDownloader(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Test",
        duration_seconds=100,
    )

    with pytest.raises(DurationLimitError, match="Nothing was downloaded"):
        downloader.download(metadata, tmp_path)
    assert not (tmp_path / metadata.video_id / "source.mp4").exists()


def test_aggregate_stream_limit_cancels_download_and_removes_partials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Test",
        duration_seconds=100,
    )
    partial_path = tmp_path / metadata.video_id / "source.video.part"

    class FakeYoutubeDL:
        def __init__(self, params):
            self.params = params

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, _url: str, download: bool):
            assert download is True
            partial_path.write_bytes(b"partial")
            progress = self.params["progress_hooks"][0]
            progress(
                {
                    "status": "downloading",
                    "filename": "source.mp4",
                    "downloaded_bytes": 10,
                    "total_bytes": 60,
                    "info_dict": {"format_id": "video"},
                }
            )
            progress(
                {
                    "status": "downloading",
                    "filename": "source.mp4",
                    "downloaded_bytes": 10,
                    "total_bytes": 60,
                    "info_dict": {"format_id": "audio"},
                }
            )
            raise AssertionError("The aggregate limit should cancel the download")

    monkeypatch.setattr(downloader_module, "YoutubeDL", FakeYoutubeDL)
    downloader = VideoDownloader(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    with pytest.raises(DownloadError, match="Downloaded streams exceed"):
        downloader.download(
            metadata,
            tmp_path,
            maximum_download_bytes=100,
        )

    assert not partial_path.exists()


def test_caches_normalized_original_youtube_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeResponse:
        headers = {"Content-Length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://i.ytimg.com/vi/abcdefghijk/maxresdefault.jpg"

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_read"):
                return b""
            self._read = True
            return b"rawimage"

    monkeypatch.setattr(downloader_module, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    def fake_run(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"normalized-jpeg")

    monkeypatch.setattr(downloader_module.subprocess, "run", fake_run)
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Test",
        duration_seconds=100,
        thumbnail_url="https://i.ytimg.com/vi/abcdefghijk/maxresdefault.jpg",
    )
    downloader = VideoDownloader(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    thumbnail = downloader._ensure_original_thumbnail(metadata, tmp_path, force=False)

    assert thumbnail == tmp_path / "source-thumbnail.jpg"
    assert thumbnail.read_bytes() == b"normalized-jpeg"


def test_rejects_untrusted_thumbnail_host_without_requesting_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        downloader_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("untrusted URL must not be requested"),
    )
    metadata = VideoMetadata(
        video_id="abcdefghijk",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        title="Test",
        duration_seconds=100,
        thumbnail_url="https://example.com/untrusted.jpg",
    )
    downloader = VideoDownloader(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    )

    assert downloader._ensure_original_thumbnail(metadata, tmp_path, force=False) is None
