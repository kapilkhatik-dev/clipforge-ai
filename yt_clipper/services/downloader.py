"""Bounded local-file staging and yt-dlp video downloads."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from yt_dlp import YoutubeDL
from yt_dlp.cookies import SUPPORTED_BROWSERS, SUPPORTED_KEYRINGS
from yt_dlp.utils import YoutubeDLError as YtDlpError

from ..domain.errors import (
    DownloadError,
    DurationLimitError,
    MediaProbeError,
    SourceTransferError,
)
from ..domain.models import (
    MAX_SOURCE_DURATION_SECONDS,
    DownloadedVideo,
    VideoMetadata,
)
from ..infrastructure.artifacts import atomic_write_text
from ..infrastructure.media_tools import MediaTools, probe_media

DownloadProgressHook = Callable[[dict[str, Any]], None]
_YOUTUBE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_LOCAL_VIDEO_EXTENSIONS = frozenset({".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_MAX_THUMBNAIL_DOWNLOAD_BYTES = 10 * 1024**2
_ADAPTIVE_MP4_FORMAT = (
    "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
)
_PROGRESSIVE_MP4_FORMAT = (
    "best[height<=1080][ext=mp4][vcodec^=avc1][acodec!=none]/"
    "best[height<=1080][ext=mp4][vcodec!=none][acodec!=none]"
)
_TRANSIENT_TRANSFER_ERROR = re.compile(
    r"(?:"
    r"http error\s+(?:403|408|425|429|5\d\d)\b|"
    r"\b(?:connection (?:aborted|reset)|network is unreachable|remote end closed)\b|"
    r"\b(?:temporary failure|temporarily unavailable|timed?\s*out)\b"
    r")",
    re.IGNORECASE,
)
LOGGER = logging.getLogger(__name__)


def validate_source_duration(
    duration_seconds: float | int | None,
    maximum_seconds: int = MAX_SOURCE_DURATION_SECONDS,
) -> float:
    if duration_seconds is None:
        raise DownloadError("YouTube did not report a valid video duration.")

    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DownloadError("YouTube did not report a valid video duration.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise DownloadError("YouTube did not report a valid video duration.")
    if duration > maximum_seconds:
        actual_minutes = duration / 60
        maximum_minutes = maximum_seconds / 60
        raise DurationLimitError(
            f"Video is {actual_minutes:.1f} minutes long; the current limit is "
            f"{maximum_minutes:.0f} minutes. Nothing was downloaded."
        )
    return duration


def validate_youtube_video_id(video_id: str) -> str:
    if not _YOUTUBE_VIDEO_ID.fullmatch(video_id):
        raise DownloadError("YouTube returned an invalid video identifier.")
    return video_id


def _is_transient_transfer_error(error: BaseException) -> bool:
    """Classify retryable transport failures without exposing upstream details."""
    return _TRANSIENT_TRANSFER_ERROR.search(str(error)) is not None


def resolve_local_video_path(value: str) -> Path | None:
    """Resolve an existing supported local video path without treating URLs as files."""
    if urlparse(value).scheme.casefold() in {"http", "https"}:
        return None
    candidate = Path(value).expanduser()
    try:
        if candidate.is_file() and candidate.suffix.casefold() in _LOCAL_VIDEO_EXTENSIONS:
            return candidate.resolve()
    except OSError:
        return None
    return None


def local_video_id(path: Path) -> str:
    """Create a safe cache key that changes when the local source changes."""
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return f"local-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def parse_cookies_from_browser(value: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    """Parse yt-dlp's common BROWSER[+KEYRING][:PROFILE][::CONTAINER] form."""
    if not value:
        return None

    browser_and_profile, separator, container = value.partition("::")
    browser_and_keyring, profile_separator, profile = browser_and_profile.partition(":")
    browser, keyring_separator, keyring = browser_and_keyring.partition("+")
    browser = browser.strip().casefold()
    if not browser:
        raise DownloadError("--cookies-from-browser requires a browser name.")
    if browser not in SUPPORTED_BROWSERS:
        raise DownloadError(f"Unsupported cookie browser '{browser}'.")

    normalized_keyring = (
        keyring.strip().upper() if keyring_separator and keyring.strip() else None
    )
    if normalized_keyring and normalized_keyring not in SUPPORTED_KEYRINGS:
        raise DownloadError(f"Unsupported browser keyring '{keyring.strip()}'.")

    return (
        browser,
        profile.strip() if profile_separator and profile.strip() else None,
        normalized_keyring,
        container.strip() if separator and container.strip() else None,
    )


class VideoDownloader:
    def __init__(self, media_tools: MediaTools) -> None:
        self._media_tools = media_tools

    def _ensure_original_thumbnail(
        self,
        metadata: VideoMetadata,
        work_dir: Path,
        force: bool,
    ) -> Path | None:
        """Cache a normalized copy of YouTube's original thumbnail when available."""
        destination = work_dir / "source-thumbnail.jpg"
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            return destination

        thumbnail_url = (metadata.thumbnail_url or "").strip()
        parsed = urlparse(thumbnail_url)
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme.casefold() != "https"
            or not hostname
            or not (hostname == "ytimg.com" or hostname.endswith(".ytimg.com"))
        ):
            return destination if destination.is_file() else None

        raw_path: Path | None = None
        converted_path: Path | None = None
        try:
            request = Request(
                thumbnail_url,
                headers={"User-Agent": "Mozilla/5.0 Clipper/1.0"},
            )
            with urlopen(request, timeout=30) as response:  # noqa: S310 - host is restricted above
                final_hostname = (
                    urlparse(response.geturl()).hostname or ""
                ).casefold()
                if not (
                    final_hostname == "ytimg.com"
                    or final_hostname.endswith(".ytimg.com")
                ):
                    raise ValueError("thumbnail redirect left the trusted YouTube image host")
                declared_size = int(response.headers.get("Content-Length") or 0)
                if declared_size > _MAX_THUMBNAIL_DOWNLOAD_BYTES:
                    raise ValueError("thumbnail exceeds the 10 MiB download limit")
                with tempfile.NamedTemporaryFile(
                    prefix="source-thumbnail-",
                    suffix=".download",
                    dir=work_dir,
                    delete=False,
                ) as temporary:
                    raw_path = Path(temporary.name)
                    downloaded = 0
                    while chunk := response.read(64 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_THUMBNAIL_DOWNLOAD_BYTES:
                            raise ValueError("thumbnail exceeds the 10 MiB download limit")
                        temporary.write(chunk)
            if not raw_path or raw_path.stat().st_size == 0:
                raise ValueError("thumbnail download was empty")

            with tempfile.NamedTemporaryFile(
                prefix="source-thumbnail-",
                suffix=".jpg",
                dir=work_dir,
                delete=False,
            ) as converted:
                converted_path = Path(converted.name)
            converted_path.unlink(missing_ok=True)
            subprocess.run(
                [
                    str(self._media_tools.ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(raw_path),
                    "-vf",
                    (
                        "scale=1280:720:force_original_aspect_ratio=increase:"
                        "flags=lanczos,crop=1280:720,setsar=1"
                    ),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(converted_path),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            if not converted_path.is_file() or converted_path.stat().st_size == 0:
                raise ValueError("FFmpeg did not create a normalized thumbnail")
            os.replace(converted_path, destination)
            return destination
        except (
            OSError,
            ValueError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            LOGGER.warning("Could not cache the original video thumbnail: %s", exc)
            return destination if destination.is_file() else None
        finally:
            if raw_path:
                raw_path.unlink(missing_ok=True)
            if converted_path:
                converted_path.unlink(missing_ok=True)

    @staticmethod
    def _common_options(cookies_from_browser: str | None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 30,
        }
        parsed_cookies = parse_cookies_from_browser(cookies_from_browser)
        if parsed_cookies:
            options["cookiesfrombrowser"] = parsed_cookies
        return options

    def inspect(
        self,
        url: str,
        cookies_from_browser: str | None = None,
        maximum_duration_seconds: int = MAX_SOURCE_DURATION_SECONDS,
    ) -> VideoMetadata:
        if local_path := resolve_local_video_path(url):
            try:
                probe = probe_media(self._media_tools, local_path)
            except MediaProbeError as exc:
                raise DownloadError(f"Could not inspect local video: {exc}") from exc
            if not probe.has_video or not probe.has_audio:
                raise DownloadError(
                    "Local input must contain both video and audio streams."
                )
            duration = validate_source_duration(
                probe.duration_seconds,
                maximum_seconds=maximum_duration_seconds,
            )
            return VideoMetadata(
                video_id=local_video_id(local_path),
                source_url=str(local_path),
                title=local_path.stem,
                duration_seconds=duration,
            )

        options = self._common_options(cookies_from_browser)
        options["skip_download"] = True

        try:
            with YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
                info = ydl.extract_info(url, download=False)
        except YtDlpError as exc:
            raise DownloadError(f"Could not inspect the YouTube video: {exc}") from exc

        if not info or info.get("_type") in {"playlist", "multi_video"} or info.get("entries"):
            raise DownloadError("Only a single YouTube video URL is supported.")

        extractor_key = str(info.get("extractor_key") or "")
        if not extractor_key.casefold().startswith("youtube"):
            raise DownloadError("Only YouTube video URLs are supported.")

        duration = validate_source_duration(
            info.get("duration"),
            maximum_seconds=maximum_duration_seconds,
        )
        video_id = validate_youtube_video_id(str(info.get("id") or "").strip())
        title = str(info.get("title") or "").strip()
        if not video_id or not title:
            raise DownloadError("YouTube returned incomplete video metadata.")

        return VideoMetadata(
            video_id=video_id,
            source_url=str(info.get("webpage_url") or url),
            title=title,
            duration_seconds=duration,
            uploader=info.get("uploader") or info.get("channel"),
            upload_date=info.get("upload_date"),
            thumbnail_url=info.get("thumbnail"),
        )

    def download(
        self,
        metadata: VideoMetadata,
        output_root: Path,
        cookies_from_browser: str | None = None,
        force: bool = False,
        progress_hook: DownloadProgressHook | None = None,
        maximum_duration_seconds: int = MAX_SOURCE_DURATION_SECONDS,
        maximum_download_bytes: int = 4 * 1024**3,
    ) -> DownloadedVideo:
        validate_source_duration(
            metadata.duration_seconds,
            maximum_seconds=maximum_duration_seconds,
        )
        local_path = resolve_local_video_path(metadata.source_url)
        video_id = (
            metadata.video_id
            if local_path is not None
            else validate_youtube_video_id(metadata.video_id)
        )
        output_root = output_root.expanduser().resolve()
        work_dir = (output_root / video_id).resolve()
        if work_dir.parent != output_root:
            raise DownloadError("Unsafe output directory derived from video metadata.")
        work_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = work_dir / "metadata.json"
        video_path = work_dir / "source.mp4"
        thumbnail_path = self._ensure_original_thumbnail(metadata, work_dir, force)

        if video_path.is_file() and not force:
            try:
                probe = probe_media(self._media_tools, video_path)
                duration_matches = abs(probe.duration_seconds - metadata.duration_seconds) <= 5
                within_mux_tolerance = (
                    probe.duration_seconds <= maximum_duration_seconds + 5
                )
                size_is_safe = video_path.stat().st_size <= maximum_download_bytes
                if (
                    probe.has_video
                    and probe.has_audio
                    and duration_matches
                    and within_mux_tolerance
                    and size_is_safe
                ):
                    atomic_write_text(
                        metadata_path, metadata.model_dump_json(indent=2)
                    )
                    return DownloadedVideo(
                        metadata=metadata,
                        video_path=video_path,
                        metadata_path=metadata_path,
                        work_dir=work_dir,
                        thumbnail_path=thumbnail_path,
                    )
            except (DurationLimitError, MediaProbeError, OSError):
                pass
            video_path.unlink(missing_ok=True)

        if force:
            for existing in work_dir.glob("source.*"):
                if existing.is_file():
                    existing.unlink()

        if local_path is not None:
            try:
                source_size = local_path.stat().st_size
            except OSError as exc:
                raise DownloadError(f"Could not inspect local video size: {exc}") from exc
            if source_size > maximum_download_bytes:
                raise DownloadError(
                    "Local video exceeds the configured source-size limit."
                )

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".source-local-",
                    suffix=".mp4",
                    dir=work_dir,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                temporary_path.unlink(missing_ok=True)

                if local_path.suffix.casefold() == ".mp4":
                    try:
                        os.link(local_path, temporary_path)
                    except OSError:
                        shutil.copy2(local_path, temporary_path)
                else:
                    subprocess.run(
                        [
                            str(self._media_tools.ffmpeg),
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            "-y",
                            "-i",
                            str(local_path),
                            "-map",
                            "0:v:0",
                            "-map",
                            "0:a:0",
                            "-c",
                            "copy",
                            "-movflags",
                            "+faststart",
                            str(temporary_path),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=600,
                    )
                os.replace(temporary_path, video_path)
                temporary_path = None
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise DownloadError(f"Could not stage local video: {exc}") from exc
            finally:
                if temporary_path:
                    temporary_path.unlink(missing_ok=True)

            try:
                if video_path.stat().st_size > maximum_download_bytes:
                    video_path.unlink(missing_ok=True)
                    raise DownloadError(
                        "Staged local video exceeds the configured source-size limit."
                    )
                staged_probe = probe_media(self._media_tools, video_path)
            except OSError as exc:
                video_path.unlink(missing_ok=True)
                raise DownloadError(
                    f"Could not inspect staged local video size: {exc}"
                ) from exc
            except MediaProbeError as exc:
                video_path.unlink(missing_ok=True)
                raise DownloadError(f"Local media failed validation: {exc}") from exc
            if (
                not staged_probe.has_video
                or not staged_probe.has_audio
                or staged_probe.duration_seconds > maximum_duration_seconds + 5
                or abs(staged_probe.duration_seconds - metadata.duration_seconds) > 5
            ):
                video_path.unlink(missing_ok=True)
                raise DownloadError("Staged local media does not match its metadata.")

            if progress_hook:
                progress_hook(
                    {
                        "status": "downloading",
                        "downloaded_bytes": source_size,
                        "total_bytes": source_size,
                    }
                )
            atomic_write_text(metadata_path, metadata.model_dump_json(indent=2))
            return DownloadedVideo(
                metadata=metadata,
                video_path=video_path,
                metadata_path=metadata_path,
                work_dir=work_dir,
                thumbnail_path=thumbnail_path,
            )

        def clean_source_files() -> None:
            for partial in work_dir.glob("source.*"):
                if partial.is_file():
                    partial.unlink(missing_ok=True)

        def download_attempt(format_selector: str) -> dict[str, Any] | None:
            # These collections are deliberately per-attempt. A failed transfer
            # must not make a clean subsequent retry look oversized.
            duration_rejection: list[DownloadError] = []
            size_rejection: list[DownloadError] = []
            downloaded_by_file: dict[str, int] = {}

            def duration_filter(
                info: dict[str, Any], *, incomplete: bool
            ) -> str | None:
                if incomplete and info.get("duration") is None:
                    return None
                try:
                    validate_source_duration(
                        info.get("duration"),
                        maximum_seconds=maximum_duration_seconds,
                    )
                except DownloadError as exc:
                    duration_rejection.append(exc)
                    return str(exc)
                return None

            def bounded_progress(status: dict[str, Any]) -> None:
                if status.get("status") in {"downloading", "finished"}:
                    info = status.get("info_dict")
                    format_id = (
                        info.get("format_id") if isinstance(info, dict) else None
                    )
                    transfer_key = str(
                        format_id
                        or status.get("filename")
                        or status.get("tmpfilename")
                        or "download"
                    )
                    transferred = max(
                        int(status.get("downloaded_bytes") or 0),
                        int(status.get("total_bytes") or 0),
                        int(status.get("total_bytes_estimate") or 0),
                    )
                    downloaded_by_file[transfer_key] = max(
                        downloaded_by_file.get(transfer_key, 0),
                        transferred,
                    )
                    if sum(downloaded_by_file.values()) > maximum_download_bytes:
                        error = DownloadError(
                            "Downloaded streams exceed the configured source-size limit."
                        )
                        size_rejection.append(error)
                        raise error
                if progress_hook:
                    progress_hook(status)

            options = self._common_options(cookies_from_browser)
            options.update(
                {
                    "format": format_selector,
                    "outtmpl": str(work_dir / "source.%(ext)s"),
                    "merge_output_format": "mp4",
                    "ffmpeg_location": self._media_tools.yt_dlp_location,
                    "overwrites": force,
                    "continuedl": True,
                    "noprogress": True,
                    "match_filter": duration_filter,
                    "break_on_reject": True,
                    "max_filesize": maximum_download_bytes,
                    "concurrent_fragment_downloads": 2,
                }
            )
            options["progress_hooks"] = [bounded_progress]

            try:
                with YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
                    downloaded_info = ydl.extract_info(
                        metadata.source_url, download=True
                    )
            except (YtDlpError, DownloadError) as exc:
                clean_source_files()
                if duration_rejection:
                    raise duration_rejection[-1] from exc
                if size_rejection:
                    raise size_rejection[-1] from exc
                raise

            if duration_rejection:
                clean_source_files()
                raise duration_rejection[-1]
            if size_rejection:
                clean_source_files()
                raise size_rejection[-1]
            return downloaded_info

        attempt_formats = (
            _ADAPTIVE_MP4_FORMAT,
            _ADAPTIVE_MP4_FORMAT,
            _PROGRESSIVE_MP4_FORMAT,
        )
        downloaded_info: dict[str, Any] | None = None
        for attempt_index, format_selector in enumerate(attempt_formats):
            try:
                downloaded_info = download_attempt(format_selector)
                break
            except DownloadError:
                # Duration and aggregate-size failures are policy decisions, not
                # transport failures, so another format must not bypass them.
                raise
            except YtDlpError as exc:
                # Discard partial streams and create a new YoutubeDL instance so
                # extraction cannot reuse an expired signed media URL.
                clean_source_files()
                if attempt_index == 0:
                    LOGGER.warning(
                        "Adaptive MP4 download failed; retrying with fresh media URLs."
                    )
                    continue
                if attempt_index == 1:
                    LOGGER.warning(
                        "Adaptive MP4 retry failed; trying a progressive MP4 format."
                    )
                    continue
                if _is_transient_transfer_error(exc):
                    raise SourceTransferError(
                        "Video download failed with both adaptive and progressive MP4 formats."
                    ) from exc
                raise DownloadError(
                    "Video download failed after all supported MP4 formats were attempted."
                ) from exc

        if downloaded_info:
            try:
                validate_source_duration(
                    downloaded_info.get("duration"),
                    maximum_seconds=maximum_duration_seconds,
                )
            except DownloadError:
                for partial in work_dir.glob("source.*"):
                    if partial.is_file():
                        partial.unlink(missing_ok=True)
                raise

        if not video_path.is_file():
            possible_outputs = [
                path
                for path in work_dir.glob("source.*")
                if path.is_file() and path.suffix.lower() not in {".part", ".ytdl"}
            ]
            if len(possible_outputs) == 1 and possible_outputs[0].suffix.lower() == ".mp4":
                video_path = possible_outputs[0]
            else:
                raise DownloadError(
                    "yt-dlp completed but did not produce the expected source.mp4 file."
                )

        try:
            probe = probe_media(self._media_tools, video_path)
        except MediaProbeError as exc:
            video_path.unlink(missing_ok=True)
            raise DownloadError(f"Downloaded media failed validation: {exc}") from exc
        if not probe.has_video or not probe.has_audio:
            video_path.unlink(missing_ok=True)
            raise DownloadError(
                "Downloaded media must contain both video and audio streams."
            )
        if (
            probe.duration_seconds > maximum_duration_seconds + 5
            or abs(probe.duration_seconds - metadata.duration_seconds) > 5
        ):
            video_path.unlink(missing_ok=True)
            raise DownloadError("Downloaded media duration does not match its metadata.")
        try:
            source_size = video_path.stat().st_size
        except OSError as exc:
            video_path.unlink(missing_ok=True)
            raise DownloadError(f"Could not inspect downloaded media size: {exc}") from exc
        if source_size > maximum_download_bytes:
            video_path.unlink(missing_ok=True)
            raise DownloadError(
                "Downloaded media exceeds the configured source-size limit."
            )

        atomic_write_text(metadata_path, metadata.model_dump_json(indent=2))
        return DownloadedVideo(
            metadata=metadata,
            video_path=video_path,
            metadata_path=metadata_path,
            work_dir=work_dir,
            thumbnail_path=thumbnail_path,
        )
