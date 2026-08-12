"""Caption-first transcript acquisition with a local faster-whisper fallback."""

from __future__ import annotations

import html
import json
import logging
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock, Timeout
from yt_dlp import YoutubeDL
from yt_dlp.utils import YoutubeDLError as YtDlpError

from ..domain.errors import MediaExtractionError, TranscriptError
from ..domain.models import (
    DownloadedVideo,
    TranscriptDocument,
    TranscriptMode,
    TranscriptOrigin,
    TranscriptSegment,
    TranscriptWord,
    WhisperDevice,
)
from ..infrastructure.artifacts import (
    atomic_write_text,
    fingerprint_file,
    fingerprint_payload,
)
from ..infrastructure.media_tools import MediaTools, extract_mono_pcm
from .downloader import parse_cookies_from_browser, resolve_local_video_path

LOGGER = logging.getLogger(__name__)
_WHISPER_PIPELINE_VERSION = 2
_AUDIO_SAMPLE_RATE = 16_000
_MAX_CAPTION_RESPONSE_BYTES = 32 * 1024 * 1024
_LOCAL_CAPTION_EXTENSIONS = (".srt", ".vtt")
TranscriptProgressCallback = Callable[[float], None]
_TIMESTAMP_LINE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})"
)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _clean_caption_text(text: str) -> str:
    text = html.unescape(_TAG.sub("", text))
    return _WHITESPACE.sub(" ", text.replace("\u200b", " ")).strip()


def _timestamp_to_seconds(value: str) -> float:
    pieces = value.replace(",", ".").split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        raise ValueError(f"Invalid subtitle timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_json3_captions(payload: str | bytes) -> list[TranscriptSegment]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"Invalid JSON3 caption data: {exc}") from exc

    events = document.get("events") or []
    segments: list[TranscriptSegment] = []
    for index, event in enumerate(events):
        raw_parts = event.get("segs") or []
        text = _clean_caption_text("".join(str(part.get("utf8") or "") for part in raw_parts))
        if not text:
            continue

        start = max(0.0, float(event.get("tStartMs") or 0) / 1000)
        duration_ms = float(event.get("dDurationMs") or 0)
        if duration_ms <= 0 and index + 1 < len(events):
            next_start = float(events[index + 1].get("tStartMs") or 0)
            duration_ms = max(0.0, next_start - start * 1000)
        end = start + (duration_ms / 1000 if duration_ms > 0 else 2.0)

        words: list[TranscriptWord] = []
        for part_index, part in enumerate(raw_parts):
            word_text = _clean_caption_text(str(part.get("utf8") or ""))
            if not word_text:
                continue
            word_start = start + float(part.get("tOffsetMs") or 0) / 1000
            if part_index + 1 < len(raw_parts):
                next_offset = float(raw_parts[part_index + 1].get("tOffsetMs") or 0) / 1000
                word_end = start + next_offset
            else:
                word_end = end
            if word_end <= word_start:
                word_end = min(end, word_start + 0.2)
            if word_end > word_start:
                words.append(
                    TranscriptWord(start=word_start, end=word_end, text=word_text)
                )

        segments.append(
            TranscriptSegment(start=start, end=end, text=text, words=words)
        )
    return segments


def parse_vtt_or_srt_captions(payload: str | bytes) -> list[TranscriptSegment]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8-sig", errors="replace")
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", normalized)
    segments: list[TranscriptSegment] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timestamp_index = next(
            (index for index, line in enumerate(lines) if "-->" in line), None
        )
        if timestamp_index is None:
            continue
        match = _TIMESTAMP_LINE.search(lines[timestamp_index])
        if not match:
            continue
        text = _clean_caption_text(" ".join(lines[timestamp_index + 1 :]))
        if not text:
            continue
        start = _timestamp_to_seconds(match.group("start"))
        end = _timestamp_to_seconds(match.group("end"))
        if end > start:
            segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def normalize_segments(
    segments: Iterable[TranscriptSegment], duration_seconds: float
) -> list[TranscriptSegment]:
    normalized: list[TranscriptSegment] = []
    seen: set[tuple[int, int, str]] = set()

    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = max(0.0, min(float(segment.start), duration_seconds))
        end = max(0.0, min(float(segment.end), duration_seconds))
        text = _clean_caption_text(segment.text)
        if not text or end <= start:
            continue

        key = (round(start * 100), round(end * 100), text.casefold())
        if key in seen:
            continue
        seen.add(key)

        words = []
        for word in segment.words:
            word_start = max(start, min(word.start, end))
            word_end = max(start, min(word.end, end))
            if word_end > word_start and word.text.strip():
                words.append(
                    TranscriptWord(
                        start=word_start,
                        end=word_end,
                        text=word.text,
                        probability=word.probability,
                    )
                )
        normalized.append(
            TranscriptSegment(start=start, end=end, text=text, words=words)
        )
    return normalized


def _language_candidates(tracks: dict[str, Any], requested: str) -> list[str]:
    requested_lower = requested.casefold()
    if requested_lower == "auto":
        available = [code for code in tracks if code.casefold() != "live_chat"]
        return sorted(
            available,
            key=lambda code: (
                0 if code.casefold().endswith("-orig") else 1,
                code.casefold(),
                code,
            ),
        )

    base = requested_lower.split("-")[0]

    def priority(code: str) -> tuple[int, str]:
        lowered = code.casefold()
        if lowered == requested_lower:
            return (0, lowered)
        if lowered == f"{base}-orig":
            return (1, lowered)
        if lowered == base:
            return (2, lowered)
        if lowered.startswith(f"{requested_lower}-"):
            return (3, lowered)
        if lowered.startswith(f"{base}-"):
            return (4, lowered)
        return (10, lowered)

    return [code for code in sorted(tracks, key=priority) if priority(code)[0] < 10]


class _CaptionResponseTooLargeError(TranscriptError):
    pass


def _read_caption_payload(response: Any) -> str | bytes:
    payload = response.read(_MAX_CAPTION_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_CAPTION_RESPONSE_BYTES:
        limit_mib = _MAX_CAPTION_RESPONSE_BYTES // (1024 * 1024)
        raise _CaptionResponseTooLargeError(
            f"YouTube caption data exceeded the {limit_mib} MiB safety limit. "
            "Try another caption language or use local Whisper transcription."
        )
    return payload


def _ordered_caption_formats(formats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preference = {"json3": 0, "vtt": 1, "srt": 2}
    supported = [item for item in formats if item.get("ext") in preference and item.get("url")]
    return sorted(supported, key=lambda item: preference[str(item["ext"])])


def audio_window_bounds(
    core_start: float,
    audio_duration: float,
    chunk_seconds: int,
    overlap_seconds: int,
) -> tuple[float, float, float]:
    core_end = min(audio_duration, core_start + chunk_seconds)
    window_start = max(0.0, core_start - overlap_seconds)
    window_end = min(audio_duration, core_end + overlap_seconds)
    return core_end, window_start, window_end


def _whisper_options_fingerprint(
    model_name: str,
    device: WhisperDevice,
    cpu_threads: int,
    batch_size: int,
    chunk_seconds: int,
    overlap_seconds: int,
) -> str:
    return fingerprint_payload(
        {
            "pipeline_version": _WHISPER_PIPELINE_VERSION,
            "model": model_name,
            "device": device.value,
            "cpu_threads": cpu_threads,
            "batch_size": batch_size,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "sample_rate": _AUDIO_SAMPLE_RATE,
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
            "word_timestamps": True,
            "condition_on_previous_text": False,
        }
    )


class TranscriptService:
    def __init__(self, media_tools: MediaTools) -> None:
        self._media_tools = media_tools

    def get_transcript(
        self,
        video: DownloadedVideo,
        language: str = "en",
        mode: TranscriptMode = TranscriptMode.AUTO,
        whisper_model: str = "small",
        whisper_device: WhisperDevice = WhisperDevice.AUTO,
        whisper_cpu_threads: int = 4,
        whisper_batch_size: int = 1,
        whisper_chunk_seconds: int = 300,
        whisper_chunk_overlap_seconds: int = 5,
        whisper_timeout_seconds: int = 3600,
        cookies_from_browser: str | None = None,
        force: bool = False,
        progress_callback: TranscriptProgressCallback | None = None,
    ) -> tuple[TranscriptDocument, Path]:
        cache_path = video.work_dir / "transcript.json"
        try:
            source_fingerprint = fingerprint_file(video.video_path)
        except OSError as exc:
            raise TranscriptError(f"Could not fingerprint source video: {exc}") from exc
        whisper_fingerprint = _whisper_options_fingerprint(
            whisper_model,
            whisper_device,
            whisper_cpu_threads,
            whisper_batch_size,
            whisper_chunk_seconds,
            whisper_chunk_overlap_seconds,
        )
        if cache_path.is_file() and not force:
            try:
                cached = TranscriptDocument.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
                if self._cache_matches(
                    cached,
                    video,
                    language,
                    mode,
                    whisper_model,
                    whisper_fingerprint,
                    source_fingerprint,
                ):
                    return cached, cache_path
            except (ValueError, OSError):
                LOGGER.warning("Ignoring invalid transcript cache at %s", cache_path)

        transcript: TranscriptDocument | None = None
        local_source = resolve_local_video_path(video.metadata.source_url)
        if mode in {TranscriptMode.AUTO, TranscriptMode.CAPTIONS}:
            try:
                transcript = (
                    self._from_local_sidecar(
                        video,
                        local_source,
                        language,
                        source_fingerprint,
                    )
                    if local_source is not None
                    else self._from_youtube_captions(
                        video,
                        language,
                        source_fingerprint,
                        cookies_from_browser,
                    )
                )
            except TranscriptError:
                if mode == TranscriptMode.CAPTIONS:
                    raise
                LOGGER.warning(
                    "Captions were unavailable; falling back to local Whisper.",
                    exc_info=True,
                )

        if transcript is None:
            if mode == TranscriptMode.CAPTIONS:
                raise TranscriptError(
                    f"No manual or automatic captions were found for language '{language}'."
                )
            transcript = self._from_whisper(
                video=video,
                language=language,
                model_name=whisper_model,
                configured_device=whisper_device,
                cpu_threads=whisper_cpu_threads,
                batch_size=whisper_batch_size,
                chunk_seconds=whisper_chunk_seconds,
                overlap_seconds=whisper_chunk_overlap_seconds,
                timeout_seconds=whisper_timeout_seconds,
                source_fingerprint=source_fingerprint,
                options_fingerprint=whisper_fingerprint,
                progress_callback=progress_callback,
            )

        atomic_write_text(cache_path, transcript.model_dump_json(indent=2))
        return transcript, cache_path

    def _from_local_sidecar(
        self,
        video: DownloadedVideo,
        source_path: Path,
        language: str,
        source_fingerprint: str,
    ) -> TranscriptDocument | None:
        source_stem = source_path.stem.casefold()
        candidates = [
            path
            for path in sorted(
                source_path.parent.iterdir(),
                key=lambda item: item.name.casefold(),
            )
            if path.is_file()
            and path.suffix.casefold() in _LOCAL_CAPTION_EXTENSIONS
            and (
                path.stem.casefold() == source_stem
                or path.name.casefold().startswith(f"{source_stem}.")
            )
        ]

        requested = language.casefold()
        if requested != "auto":
            requested_base = requested.split("-")[0]
            candidates.sort(
                key=lambda path: (
                    0
                    if f".{requested}." in path.name.casefold()
                    else 1
                    if f".{requested_base}-" in path.name.casefold()
                    else 2,
                    path.name.casefold(),
                )
            )

        for caption_path in candidates:
            try:
                if caption_path.stat().st_size > _MAX_CAPTION_RESPONSE_BYTES:
                    continue
                payload = caption_path.read_bytes()
                segments = normalize_segments(
                    parse_vtt_or_srt_captions(payload),
                    video.metadata.duration_seconds,
                )
            except OSError as exc:
                LOGGER.debug("Could not read local captions %s: %s", caption_path, exc)
                continue
            if not segments:
                continue

            name_without_extension = caption_path.name[: -len(caption_path.suffix)]
            language_suffix = name_without_extension.removeprefix(
                f"{source_path.stem}."
            )
            detected_language = (
                language_suffix if language_suffix != name_without_extension else language
            )
            origin = (
                TranscriptOrigin.AUTOMATIC
                if "auto" in language_suffix.casefold()
                else TranscriptOrigin.MANUAL
            )
            return TranscriptDocument(
                video_id=video.metadata.video_id,
                language=detected_language or language,
                requested_language=language,
                origin=origin,
                source_fingerprint=source_fingerprint,
                duration_seconds=video.metadata.duration_seconds,
                segments=segments,
            )
        return None

    @staticmethod
    def _cache_matches(
        cached: TranscriptDocument,
        video: DownloadedVideo,
        language: str,
        mode: TranscriptMode,
        whisper_model: str,
        whisper_fingerprint: str,
        source_fingerprint: str,
    ) -> bool:
        if not (
            cached.video_id == video.metadata.video_id
            and cached.requested_language.casefold() == language.casefold()
            and cached.source_fingerprint == source_fingerprint
            and abs(cached.duration_seconds - video.metadata.duration_seconds) <= 0.01
        ):
            return False
        if cached.origin == TranscriptOrigin.WHISPER and (
            cached.whisper_model != whisper_model
            or cached.whisper_options_fingerprint != whisper_fingerprint
        ):
            return False
        if mode == TranscriptMode.AUTO:
            return True
        if mode == TranscriptMode.WHISPER:
            return cached.origin == TranscriptOrigin.WHISPER
        return cached.origin in {TranscriptOrigin.MANUAL, TranscriptOrigin.AUTOMATIC}

    def _from_youtube_captions(
        self,
        video: DownloadedVideo,
        language: str,
        source_fingerprint: str,
        cookies_from_browser: str | None,
    ) -> TranscriptDocument | None:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": 30,
            "retries": 3,
        }
        parsed_cookies = parse_cookies_from_browser(cookies_from_browser)
        if parsed_cookies:
            options["cookiesfrombrowser"] = parsed_cookies

        caption_limit_error: _CaptionResponseTooLargeError | None = None
        try:
            with YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
                info = ydl.extract_info(video.metadata.source_url, download=False)
                if not info:
                    raise TranscriptError("YouTube returned no caption metadata.")

                repositories = (
                    (TranscriptOrigin.MANUAL, info.get("subtitles") or {}),
                    (TranscriptOrigin.AUTOMATIC, info.get("automatic_captions") or {}),
                )
                for origin, tracks in repositories:
                    for track_language in _language_candidates(tracks, language):
                        for caption_format in _ordered_caption_formats(tracks[track_language]):
                            try:
                                with ydl.urlopen(caption_format["url"]) as response:
                                    payload = _read_caption_payload(response)
                                if caption_format["ext"] == "json3":
                                    parsed = parse_json3_captions(payload)
                                else:
                                    parsed = parse_vtt_or_srt_captions(payload)
                                segments = normalize_segments(
                                    parsed, video.metadata.duration_seconds
                                )
                                if segments:
                                    return TranscriptDocument(
                                        video_id=video.metadata.video_id,
                                        language=track_language,
                                        requested_language=language,
                                        origin=origin,
                                        source_fingerprint=source_fingerprint,
                                        duration_seconds=video.metadata.duration_seconds,
                                        segments=segments,
                                    )
                            except _CaptionResponseTooLargeError as exc:
                                caption_limit_error = exc
                                LOGGER.debug(
                                    "Caption format %s exceeded the response limit: %s",
                                    caption_format.get("ext"),
                                    exc,
                                )
                            except Exception as exc:  # Try the next offered subtitle format.
                                LOGGER.debug(
                                    "Caption format %s failed: %s",
                                    caption_format.get("ext"),
                                    exc,
                                )
        except YtDlpError as exc:
            raise TranscriptError(f"Could not inspect YouTube captions: {exc}") from exc

        if caption_limit_error is not None:
            raise caption_limit_error
        return None

    @staticmethod
    def _resolve_whisper_device(
        ctranslate2_module: Any,
        configured_device: WhisperDevice,
    ) -> tuple[str, str]:
        cuda_available = ctranslate2_module.get_cuda_device_count() > 0
        if configured_device == WhisperDevice.CUDA and not cuda_available:
            raise TranscriptError("CUDA was requested for Whisper but no CUDA device is available.")
        device = (
            "cuda"
            if configured_device == WhisperDevice.CUDA
            or (configured_device == WhisperDevice.AUTO and cuda_available)
            else "cpu"
        )
        if device == "cpu":
            return device, "int8"
        try:
            supported = set(ctranslate2_module.get_supported_compute_types("cuda"))
        except Exception:
            supported = set()
        compute_type = "int8_float16" if "int8_float16" in supported else "float16"
        return device, compute_type

    @staticmethod
    def _notify_progress(
        callback: TranscriptProgressCallback | None,
        progress: float,
    ) -> None:
        if not callback:
            return
        try:
            callback(max(0.0, min(1.0, progress)))
        except Exception:
            LOGGER.warning("Transcript progress callback failed", exc_info=True)

    def _from_whisper(
        self,
        *,
        video: DownloadedVideo,
        language: str,
        model_name: str,
        configured_device: WhisperDevice,
        cpu_threads: int,
        batch_size: int,
        chunk_seconds: int,
        overlap_seconds: int,
        timeout_seconds: int,
        source_fingerprint: str,
        options_fingerprint: str,
        progress_callback: TranscriptProgressCallback | None,
    ) -> TranscriptDocument:
        try:
            import ctranslate2
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise TranscriptError(
                "Local transcription requires faster-whisper. Install requirements.txt."
            ) from exc

        deadline = time.monotonic() + timeout_seconds

        def remaining_time() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptError("Local Whisper transcription exceeded its time limit.")
            return remaining

        temp_root = video.work_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        segments: list[TranscriptSegment] = []
        detected_language: str | None = None

        lock_path = video.work_dir.parent / ".whisper.lock"
        try:
            with FileLock(str(lock_path), timeout=remaining_time()):
                with tempfile.TemporaryDirectory(prefix="whisper-", dir=temp_root) as temp:
                    pcm_path = Path(temp) / "audio.pcm"
                    sample_count = extract_mono_pcm(
                        self._media_tools,
                        video.video_path,
                        pcm_path,
                        timeout_seconds=remaining_time(),
                        max_duration_seconds=video.metadata.duration_seconds,
                    )
                    audio_duration = min(
                        video.metadata.duration_seconds,
                        sample_count / _AUDIO_SAMPLE_RATE,
                    )
                    pcm = np.memmap(pcm_path, mode="r", dtype="<i2")
                    try:
                        device, compute_type = self._resolve_whisper_device(
                            ctranslate2,
                            configured_device,
                        )
                        model = WhisperModel(
                            model_name,
                            device=device,
                            compute_type=compute_type,
                            cpu_threads=cpu_threads,
                            num_workers=1,
                        )
                        engine = (
                            BatchedInferencePipeline(model)
                            if batch_size > 1
                            else model
                        )
                        core_start = 0.0
                        while core_start < audio_duration:
                            _ = remaining_time()
                            core_end, window_start, window_end = audio_window_bounds(
                                core_start,
                                audio_duration,
                                chunk_seconds,
                                overlap_seconds,
                            )
                            first_sample = int(window_start * _AUDIO_SAMPLE_RATE)
                            last_sample = min(
                                sample_count,
                                int(window_end * _AUDIO_SAMPLE_RATE),
                            )
                            audio_window = np.asarray(
                                pcm[first_sample:last_sample],
                                dtype=np.float32,
                            )
                            audio_window *= 1.0 / 32768.0

                            requested_language = detected_language
                            if not requested_language and language.casefold() != "auto":
                                requested_language = language
                            transcribe_options: dict[str, Any] = {
                                "language": requested_language,
                                "beam_size": 5,
                                "temperature": 0.0,
                                "vad_filter": True,
                                "vad_parameters": {
                                    "min_silence_duration_ms": 500,
                                    "speech_pad_ms": 200,
                                },
                                "word_timestamps": True,
                                "without_timestamps": False,
                                "condition_on_previous_text": False,
                            }
                            if batch_size > 1:
                                transcribe_options["batch_size"] = batch_size
                            raw_segments, info = engine.transcribe(
                                audio_window,
                                **transcribe_options,
                            )
                            if not detected_language:
                                reported_language = getattr(info, "language", None)
                                if reported_language:
                                    detected_language = str(reported_language)
                                elif language.casefold() != "auto":
                                    detected_language = language

                            is_last_core = core_end >= audio_duration - 0.001
                            for raw_segment in raw_segments:
                                _ = remaining_time()
                                absolute_start = window_start + float(raw_segment.start)
                                absolute_end = window_start + float(raw_segment.end)
                                midpoint = (absolute_start + absolute_end) / 2
                                if midpoint < core_start or (
                                    not is_last_core and midpoint >= core_end
                                ):
                                    continue
                                absolute_start = max(0.0, absolute_start)
                                absolute_end = min(
                                    video.metadata.duration_seconds,
                                    absolute_end,
                                )
                                text = _clean_caption_text(raw_segment.text)
                                if not text or absolute_end <= absolute_start:
                                    continue

                                words: list[TranscriptWord] = []
                                for raw_word in raw_segment.words or []:
                                    word_text = _clean_caption_text(raw_word.word)
                                    word_start = max(
                                        absolute_start,
                                        window_start + float(raw_word.start),
                                    )
                                    word_end = min(
                                        absolute_end,
                                        window_start + float(raw_word.end),
                                    )
                                    if word_text and word_end > word_start:
                                        words.append(
                                            TranscriptWord(
                                                start=word_start,
                                                end=word_end,
                                                text=word_text,
                                                probability=float(raw_word.probability),
                                            )
                                        )
                                segments.append(
                                    TranscriptSegment(
                                        start=absolute_start,
                                        end=absolute_end,
                                        text=text,
                                        words=words,
                                    )
                                )

                            del audio_window
                            core_start = core_end
                            self._notify_progress(
                                progress_callback,
                                core_end / audio_duration,
                            )
                    finally:
                        del pcm
        except Timeout as exc:
            raise TranscriptError(
                "Another local Whisper job did not finish before the timeout."
            ) from exc
        except TranscriptError:
            raise
        except MediaExtractionError as exc:
            raise TranscriptError(str(exc)) from exc
        except Exception as exc:
            raise TranscriptError(f"Local Whisper transcription failed: {exc}") from exc

        normalized = normalize_segments(segments, video.metadata.duration_seconds)
        if not normalized:
            raise TranscriptError("Whisper completed but returned no spoken transcript.")

        return TranscriptDocument(
            video_id=video.metadata.video_id,
            language=detected_language or language,
            requested_language=language,
            origin=TranscriptOrigin.WHISPER,
            source_fingerprint=source_fingerprint,
            whisper_model=model_name,
            whisper_options_fingerprint=options_fingerprint,
            duration_seconds=video.metadata.duration_seconds,
            segments=normalized,
        )
