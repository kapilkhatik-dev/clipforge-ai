"""Discovery and capability checks for the external FFmpeg toolchain."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..domain.errors import DependencyError, MediaExtractionError, MediaProbeError


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float
    has_video: bool
    has_audio: bool
    has_attached_picture: bool = False


@dataclass(frozen=True)
class MediaTools:
    ffmpeg: Path
    ffprobe: Path

    @property
    def yt_dlp_location(self) -> str:
        return str(self.ffmpeg.parent)


def _find_winget_executable(command: str) -> Path | None:
    if os.name != "nt" or not (local_app_data := os.getenv("LOCALAPPDATA")):
        return None
    packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(packages.glob(f"Gyan.FFmpeg_*/ffmpeg-*/bin/{command}.exe"))
    return matches[-1].resolve() if matches else None


def _resolve_executable(configured: str | None, command: str) -> Path | None:
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            executable_name = f"{command}.exe" if os.name == "nt" else command
            configured_path = configured_path / executable_name
        if configured_path.is_file():
            return configured_path.resolve()
        discovered = shutil.which(configured)
        if discovered:
            return Path(discovered).resolve()

    discovered = shutil.which(command)
    if discovered:
        return Path(discovered).resolve()
    return _find_winget_executable(command)


def resolve_media_tools() -> MediaTools:
    ffmpeg_home = os.getenv("FFMPEG_HOME")
    ffmpeg = _resolve_executable(os.getenv("FFMPEG_BINARY") or ffmpeg_home, "ffmpeg")
    ffprobe = _resolve_executable(os.getenv("FFPROBE_BINARY") or ffmpeg_home, "ffprobe")

    missing = []
    if not ffmpeg:
        missing.append("ffmpeg")
    if not ffprobe:
        missing.append("ffprobe")
    if missing:
        names = " and ".join(missing)
        raise DependencyError(
            f"Missing {names}. Install FFmpeg and ensure its bin directory is on PATH, "
            "or set FFMPEG_HOME in .env. On Windows: "
            "winget install --id Gyan.FFmpeg --exact"
        )

    assert ffmpeg is not None and ffprobe is not None
    if ffmpeg.parent != ffprobe.parent:
        raise DependencyError(
            "FFmpeg and FFprobe must be installed in the same bin directory so yt-dlp "
            "can use both. Set FFMPEG_HOME to that directory."
        )
    return MediaTools(ffmpeg=ffmpeg, ffprobe=ffprobe)


def _run_capability_command(executable: Path, argument: str) -> str:
    try:
        result = subprocess.run(
            [str(executable), "-hide_banner", argument],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DependencyError(f"Could not inspect FFmpeg capabilities: {exc}") from exc
    return f"{result.stdout}\n{result.stderr}"


def validate_media_tools(tools: MediaTools) -> None:
    _run_capability_command(tools.ffmpeg, "-version")
    _run_capability_command(tools.ffprobe, "-version")


def probe_media(tools: MediaTools, path: Path) -> MediaProbe:
    try:
        result = subprocess.run(
            [
                str(tools.ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type:stream_disposition=attached_pic",
                "-of",
                "json",
                str(path.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        payload = json.loads(result.stdout)
        duration = float(payload.get("format", {}).get("duration", 0))
        streams = payload.get("streams", [])
        has_attached_picture = any(
            stream.get("codec_type") == "video"
            and bool(stream.get("disposition", {}).get("attached_pic"))
            for stream in streams
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        TypeError,
        OverflowError,
        json.JSONDecodeError,
    ) as exc:
        raise MediaProbeError(f"FFprobe could not read '{path}': {exc}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MediaProbeError(f"FFprobe reported an invalid duration for '{path}'.")
    return MediaProbe(
        duration_seconds=duration,
        has_video=any(
            stream.get("codec_type") == "video"
            and not bool(stream.get("disposition", {}).get("attached_pic"))
            for stream in streams
        ),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        has_attached_picture=has_attached_picture,
    )


def extract_mono_pcm(
    tools: MediaTools,
    source_path: Path,
    destination_path: Path,
    timeout_seconds: float,
    max_duration_seconds: float,
) -> int:
    """Extract bounded 16 kHz mono signed-16 PCM and return its sample count."""
    samples_per_second = 16_000
    max_samples_float = max_duration_seconds * samples_per_second
    if (
        not math.isfinite(max_duration_seconds)
        or max_duration_seconds <= 0
        or not math.isfinite(max_samples_float)
    ):
        raise MediaExtractionError("PCM extraction requires a finite positive duration limit.")
    max_sample_count = math.ceil(max_samples_float)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                str(tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path.resolve()),
                "-t",
                str(max_duration_seconds),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(samples_per_second),
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "-threads",
                "1",
                str(destination_path.resolve()),
            ],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        destination_path.unlink(missing_ok=True)
        raise MediaExtractionError(f"Could not extract source audio: {exc}") from exc

    try:
        size = destination_path.stat().st_size
    except OSError as exc:
        raise MediaExtractionError(f"Could not inspect extracted audio: {exc}") from exc
    if size <= 0 or size % 2:
        destination_path.unlink(missing_ok=True)
        raise MediaExtractionError("FFmpeg produced invalid PCM audio.")

    sample_count = size // 2
    if sample_count > max_sample_count:
        destination_path.unlink(missing_ok=True)
        raise MediaExtractionError(
            "FFmpeg produced PCM audio longer than the configured media duration."
        )
    return sample_count


def validate_ffmpeg_capabilities(tools: MediaTools) -> None:
    filters = _run_capability_command(tools.ffmpeg, "-filters")
    encoders = _run_capability_command(tools.ffmpeg, "-encoders")

    required_filters = (
        "subtitles",
        "gblur",
        "overlay",
        "scale",
        "crop",
        "split",
        "eq",
        "setsar",
    )
    missing_filters = [
        name for name in required_filters if not re.search(rf"\b{re.escape(name)}\b", filters)
    ]
    required_encoders = ("libx264", "aac")
    missing_encoders = [
        name for name in required_encoders if not re.search(rf"\b{re.escape(name)}\b", encoders)
    ]

    if missing_filters or missing_encoders:
        details = []
        if missing_filters:
            details.append(f"filters: {', '.join(missing_filters)}")
        if missing_encoders:
            details.append(f"encoders: {', '.join(missing_encoders)}")
        raise DependencyError(
            "The installed FFmpeg build lacks required " + "; ".join(details) + "."
        )
