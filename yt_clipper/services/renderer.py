"""FFmpeg rendering and ASS caption generation."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..domain.errors import MediaProbeError, RenderError
from ..domain.models import (
    ClipCandidate,
    DownloadedVideo,
    HighlightMoment,
    HighlightMontage,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
    VideoLayout,
)
from ..infrastructure.artifacts import (
    atomic_write_text,
    fingerprint_file,
    fingerprint_payload,
)
from ..infrastructure.media_tools import MediaTools, probe_media


_RENDER_SCHEMA_VERSION = 5
_CAPTION_OVERLAP_TOLERANCE_SECONDS = 0.01
_MIN_CAPTION_DURATION_SECONDS = 0.05
_THUMBNAIL_WIDTH = 1280
_THUMBNAIL_HEIGHT = 720
_POSTER_WIDTH = 1080
_POSTER_HEIGHT = 1920


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    text: str


def _slice_untimed_text(
    segment: TranscriptSegment, visible_start: float, visible_end: float
) -> str:
    words = segment.text.split()
    if not words or (visible_start <= segment.start and visible_end >= segment.end):
        return segment.text
    duration = segment.end - segment.start
    first = math.floor((visible_start - segment.start) / duration * len(words))
    last = math.ceil((visible_end - segment.start) / duration * len(words))
    return " ".join(words[max(0, first) : min(len(words), max(first + 1, last))])


def clip_transcript_segments(
    segments: list[TranscriptSegment], clip_start: float, clip_end: float
) -> list[TranscriptSegment]:
    clipped: list[TranscriptSegment] = []
    for segment in segments:
        if segment.start >= clip_end or segment.end <= clip_start:
            continue
        start = max(segment.start, clip_start)
        end = min(segment.end, clip_end)
        if end <= start:
            continue

        words = []
        for word in segment.words:
            word_start = max(word.start, clip_start)
            word_end = min(word.end, clip_end)
            if word_end > word_start:
                words.append(
                    TranscriptWord(
                        start=word_start - clip_start,
                        end=word_end - clip_start,
                        text=word.text,
                        probability=word.probability,
                    )
                )
        text = (
            " ".join(word.text for word in words)
            if words
            else _slice_untimed_text(segment, start, end)
        )
        if text:
            clipped.append(
                TranscriptSegment(
                    start=start - clip_start,
                    end=end - clip_start,
                    text=text,
                    words=words,
                )
            )
    return clipped


def montage_transcript_segments(
    segments: list[TranscriptSegment],
    moments: list[HighlightMoment],
) -> list[TranscriptSegment]:
    """Rebase source transcript cues onto the montage's concatenated timeline."""
    rebased: list[TranscriptSegment] = []
    cursor = 0.0
    for moment in moments:
        for segment in clip_transcript_segments(segments, moment.start, moment.end):
            rebased.append(
                segment.model_copy(
                    update={
                        "start": segment.start + cursor,
                        "end": segment.end + cursor,
                        "words": [
                            word.model_copy(
                                update={
                                    "start": word.start + cursor,
                                    "end": word.end + cursor,
                                }
                            )
                            for word in segment.words
                        ],
                    }
                )
            )
        cursor += moment.duration
    return rebased


def _split_words(words: list[str], maximum_words: int = 7) -> list[list[str]]:
    return [words[index : index + maximum_words] for index in range(0, len(words), maximum_words)]


def _caption_text_key(text: str) -> str:
    return " ".join(text.casefold().split())


def _single_active_caption_timeline(cues: list[CaptionCue]) -> list[CaptionCue]:
    """Merge duplicate cues and prevent libass from stacking overlapping events."""
    valid = [
        CaptionCue(cue.start, cue.end, cue.text.strip())
        for cue in cues
        if cue.text.strip() and cue.end - cue.start >= _MIN_CAPTION_DURATION_SECONDS
    ]
    valid.sort(key=lambda cue: (cue.start, cue.end, _caption_text_key(cue.text)))

    intervals_by_text: dict[str, list[CaptionCue]] = {}
    for cue in valid:
        key = _caption_text_key(cue.text)
        matching = intervals_by_text.setdefault(key, [])
        if matching and cue.start <= (
            matching[-1].end + _CAPTION_OVERLAP_TOLERANCE_SECONDS
        ):
            previous = matching[-1]
            matching[-1] = CaptionCue(
                start=min(previous.start, cue.start),
                end=max(previous.end, cue.end),
                text=previous.text,
            )
        else:
            matching.append(cue)

    deduplicated = sorted(
        (cue for matching in intervals_by_text.values() for cue in matching),
        key=lambda cue: (cue.start, cue.end, _caption_text_key(cue.text)),
    )
    timeline: list[CaptionCue] = []
    for cue in deduplicated:
        if not timeline:
            timeline.append(cue)
            continue

        previous = timeline[-1]
        if cue.start >= previous.end:
            timeline.append(cue)
            continue

        if cue.start <= previous.start + _CAPTION_OVERLAP_TOLERANCE_SECONDS:
            richer = max(
                (previous, cue),
                key=lambda item: (
                    len(item.text.split()),
                    len(item.text),
                    item.end - item.start,
                ),
            )
            timeline[-1] = CaptionCue(
                start=min(previous.start, cue.start),
                end=richer.end,
                text=richer.text,
            )
            continue

        shortened = CaptionCue(previous.start, cue.start, previous.text)
        if shortened.end - shortened.start >= _MIN_CAPTION_DURATION_SECONDS:
            timeline[-1] = shortened
        else:
            timeline.pop()
        timeline.append(cue)

    return timeline


def build_caption_cues(segments: list[TranscriptSegment]) -> list[CaptionCue]:
    raw_cues: list[CaptionCue] = []
    for segment in segments:
        if segment.words:
            word_groups = [
                segment.words[index : index + 7]
                for index in range(0, len(segment.words), 7)
            ]
            for group in word_groups:
                text = " ".join(word.text for word in group).strip()
                if text and group[-1].end > group[0].start:
                    raw_cues.append(
                        CaptionCue(group[0].start, group[-1].end, text)
                    )
            continue

        words = segment.text.split()
        groups = _split_words(words)
        if not groups:
            continue
        duration = segment.end - segment.start
        total_words = max(1, len(words))
        consumed = 0
        for group in groups:
            cue_start = segment.start + duration * consumed / total_words
            consumed += len(group)
            cue_end = segment.start + duration * consumed / total_words
            if cue_end > cue_start:
                raw_cues.append(
                    CaptionCue(cue_start, cue_end, " ".join(group))
                )
    return _single_active_caption_timeline(raw_cues)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, hundredths = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{hundredths:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _wrap_ass_text(text: str, target_width: int = 28) -> str:
    words = text.split()
    if len(text) <= target_width or len(words) < 2:
        return text

    best_index = min(
        range(1, len(words)),
        key=lambda index: abs(
            len(" ".join(words[:index])) - len(" ".join(words[index:]))
        ),
    )
    return f"{' '.join(words[:best_index])}\\N{' '.join(words[best_index:])}"


def build_ass_document(
    segments: list[TranscriptSegment],
    width: int = 1080,
    height: int = 1920,
    font_name: str = "Arial",
) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
Kerning: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Clip,{font_name},80,&H0017DFFF,&H0017DFFF,&H00141414,&H50000000,-1,0,0,0,100,100,0.4,0,1,5,2.5,2,84,84,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for cue in build_caption_cues(segments):
        text = _wrap_ass_text(_escape_ass_text(cue.text))
        if text and cue.end > cue.start:
            events.append(
                f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},"
                f"Clip,,0,0,0,,{text}"
            )
    return header + "\n".join(events) + ("\n" if events else "")


def _balanced_title_lines(
    title: str,
    maximum_lines: int = 3,
    one_line_limit: int = 26,
    target_line_length: int = 31,
) -> list[str]:
    words = title.split()
    if not words:
        return []
    preferred_lines = (
        1
        if len(title) <= one_line_limit
        else max(2, math.ceil(len(title) / target_line_length))
    )
    line_count = min(maximum_lines, preferred_lines)
    line_count = min(line_count, len(words))
    lines: list[str] = []
    start = 0
    for line_index in range(line_count - 1):
        remaining_lines = line_count - line_index
        maximum_end = len(words) - (remaining_lines - 1)
        remaining_length = len(" ".join(words[start:]))
        target = remaining_length / remaining_lines
        end = min(
            range(start + 1, maximum_end + 1),
            key=lambda candidate: abs(len(" ".join(words[start:candidate])) - target),
        )
        lines.append(" ".join(words[start:end]))
        start = end
    lines.append(" ".join(words[start:]))
    return lines


def build_thumbnail_ass_document(
    title: str,
    width: int = _THUMBNAIL_WIDTH,
    height: int = _THUMBNAIL_HEIGHT,
    font_name: str = "Arial",
) -> str:
    lines = _balanced_title_lines(_escape_ass_text(title))
    wrapped_title = r"\N".join(lines)
    longest_line = max((len(line) for line in lines), default=0)
    if longest_line <= 26:
        font_size = 78
    elif longest_line <= 34:
        font_size = 70
    elif longest_line <= 42:
        font_size = 62
    else:
        font_size = 54
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
Kerning: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H70000000,-1,0,0,0,100,100,0.3,0,1,3,2,1,72,72,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Title,,0,0,0,,{wrapped_title}
"""


def build_thumbnail_filter_graph(
    width: int = _THUMBNAIL_WIDTH,
    height: int = _THUMBNAIL_HEIGHT,
) -> str:
    panel_top = round(height * 0.54)
    panel_height = height - panel_top
    accent_y = panel_top + 38
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,eq=contrast=1.04:saturation=0.92,"
        f"drawbox=x=0:y={panel_top}:w=iw:h={panel_height}:color=black@0.72:t=fill,"
        f"drawbox=x=72:y={accent_y}:w=118:h=7:color=0xFFDF17@1:t=fill,"
        "subtitles=filename='thumbnail-title.ass'"
    )


def build_vertical_poster_ass_document(
    title: str,
    width: int = _POSTER_WIDTH,
    height: int = _POSTER_HEIGHT,
    font_name: str = "Arial",
) -> str:
    lines = _balanced_title_lines(
        _escape_ass_text(title),
        maximum_lines=5,
        one_line_limit=17,
        target_line_length=18,
    )
    wrapped_title = r"\N".join(lines)
    longest_line = max((len(line) for line in lines), default=0)
    if longest_line <= 16:
        font_size = 108
    elif longest_line <= 20:
        font_size = 96
    elif longest_line <= 24:
        font_size = 84
    elif longest_line <= 28:
        font_size = 72
    else:
        font_size = 62
    title_top = round(height * 0.54)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
Kerning: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: PosterTitle,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H70000000,-1,0,0,0,100,100,0.4,0,1,4,2.5,7,84,84,{title_top},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,PosterTitle,,0,0,0,,{wrapped_title}
"""


def build_vertical_poster_filter_graph(
    width: int = _POSTER_WIDTH,
    height: int = _POSTER_HEIGHT,
) -> str:
    panel_top = round(height * 0.42)
    accent_y = panel_top + 92
    artwork_y = 0
    artwork_box_height = round(height * 0.40)
    return (
        "[0:v]split=2[poster_background][poster_artwork];"
        f"[poster_background]scale={width}:{height}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},gblur=sigma=42,"
        "eq=brightness=-0.32:saturation=0.78,setsar=1[background];"
        f"[poster_artwork]scale={width}:{artwork_box_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[artwork];"
        f"[background][artwork]overlay=(W-w)/2:{artwork_y}[canvas];"
        f"[canvas]drawbox=x=0:y={panel_top}:w=iw:h={height - panel_top}:"
        "color=black@0.76:t=fill,"
        f"drawbox=x=84:y={accent_y}:w=150:h=9:color=0xFFDF17@1:t=fill,"
        "subtitles=filename='poster-title.ass'[poster]"
    )


def slugify(value: str, maximum_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (slug[:maximum_length].rstrip("-") or "clip")


def build_video_filter_graph(
    video_layout: VideoLayout,
    width: int,
    height: int,
) -> str:
    if video_layout == VideoLayout.FILL_CROP:
        return (
            f"[0:v:0]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,"
            f"subtitles=filename='captions.ass'[video]"
        )

    return (
        f"[0:v:0]split=2[background][foreground];"
        f"[background]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=24,eq=brightness=-0.22,setsar=1[bg];"
        f"[foreground]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[composed];"
        f"[composed]subtitles=filename='captions.ass'[video]"
    )


def build_montage_filter_graph(
    moments: list[HighlightMoment],
    video_layout: VideoLayout,
    width: int,
    height: int,
) -> str:
    """Build exact per-moment trims followed by one captioned A/V concat."""
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, moment in enumerate(moments):
        trim = f"trim=start={moment.start:.3f}:end={moment.end:.3f},setpts=PTS-STARTPTS"
        if video_layout == VideoLayout.FILL_CROP:
            filters.append(
                f"[0:v:0]{trim},scale={width}:{height}:"
                "force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height},setsar=1[v{index}]"
            )
        else:
            filters.extend(
                [
                    f"[0:v:0]{trim},split=2[background{index}][foreground{index}]",
                    f"[background{index}]scale={width}:{height}:"
                    "force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={width}:{height},gblur=sigma=24,"
                    f"eq=brightness=-0.22,setsar=1[bg{index}]",
                    f"[foreground{index}]scale={width}:{height}:"
                    "force_original_aspect_ratio=decrease:flags=lanczos,"
                    f"setsar=1[fg{index}]",
                    f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2[v{index}]",
                ]
            )
        filters.append(
            f"[0:a:0]atrim=start={moment.start:.3f}:end={moment.end:.3f},"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(moments)}:v=1:a=1[montage_video][audio]"
    )
    filters.append(
        "[montage_video]subtitles=filename='captions.ass'[video]"
    )
    return ";".join(filters)


class ClipRenderer:
    def __init__(self, media_tools: MediaTools) -> None:
        self._media_tools = media_tools

    @staticmethod
    def _render_fingerprint(
        video: DownloadedVideo,
        transcript: TranscriptDocument,
        candidate: ClipCandidate,
        video_layout: VideoLayout,
        width: int,
        height: int,
    ) -> str:
        return fingerprint_payload(
            {
                "schema_version": _RENDER_SCHEMA_VERSION,
                "source": fingerprint_file(video.video_path),
                "source_thumbnail": (
                    fingerprint_file(video.thumbnail_path)
                    if video.thumbnail_path and video.thumbnail_path.is_file()
                    else "fallback-source-frame"
                ),
                "transcript": transcript.model_dump(mode="json"),
                "candidate": candidate.model_dump(mode="json"),
                "video_layout": video_layout.value,
                "width": width,
                "height": height,
                "caption_style": "bold-yellow-charcoal-outline-v2",
                "caption_timeline": "single-active-v1",
                "thumbnail_style": "source-artwork-title-panel-v1",
                "video_codec": "libx264-crf18-medium",
                "audio_codec": "aac-192k",
            }
        )

    def _cached_output_is_valid(
        self,
        output_path: Path,
        thumbnail_path: Path,
        manifest_path: Path,
        expected_fingerprint: str,
        expected_duration: float,
    ) -> bool:
        if (
            not output_path.is_file()
            or not thumbnail_path.is_file()
            or not manifest_path.is_file()
        ):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("render_fingerprint") != expected_fingerprint:
                return False
            if manifest.get("output_fingerprint") != fingerprint_file(output_path):
                return False
            if manifest.get("thumbnail_fingerprint") != fingerprint_file(thumbnail_path):
                return False
            probe = probe_media(self._media_tools, output_path)
            return (
                probe.has_video
                and probe.has_attached_picture
                and abs(probe.duration_seconds - expected_duration) <= 0.75
            )
        except (OSError, ValueError, MediaProbeError, json.JSONDecodeError):
            return False

    @staticmethod
    def _poster_fingerprint(
        video: DownloadedVideo,
        candidate: ClipCandidate,
    ) -> str:
        source = (
            video.thumbnail_path
            if video.thumbnail_path and video.thumbnail_path.is_file()
            else video.video_path
        )
        return fingerprint_payload(
            {
                "schema_version": 1,
                "source": fingerprint_file(source),
                "source_kind": (
                    "original-thumbnail"
                    if source != video.video_path
                    else "fallback-source-frame"
                ),
                "candidate": candidate.model_dump(mode="json"),
                "poster_style": "vertical-original-artwork-title-panel-v2",
                "width": _POSTER_WIDTH,
                "height": _POSTER_HEIGHT,
            }
        )

    def _ensure_vertical_poster(
        self,
        video: DownloadedVideo,
        candidate: ClipCandidate,
        output_path: Path,
        manifest_path: Path,
        force: bool,
    ) -> Path:
        poster_path = output_path.with_suffix(".poster.jpg")
        try:
            poster_fingerprint = self._poster_fingerprint(video, candidate)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("render manifest must be a JSON object")
            if (
                not force
                and poster_path.is_file()
                and manifest.get("poster_fingerprint") == poster_fingerprint
                and manifest.get("poster_output_fingerprint")
                == fingerprint_file(poster_path)
            ):
                return poster_path
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RenderError(f"Could not initialize vertical poster: {exc}") from exc

        temp_root = video.work_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="poster-", dir=temp_root) as temp:
            temp_dir = Path(temp)
            poster_title_path = temp_dir / "poster-title.ass"
            temporary_poster = temp_dir / "poster.jpg"
            poster_title_path.write_text(
                build_vertical_poster_ass_document(candidate.title),
                encoding="utf-8-sig",
            )
            poster_source = (
                video.thumbnail_path
                if video.thumbnail_path and video.thumbnail_path.is_file()
                else video.video_path
            )
            command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if poster_source == video.video_path:
                command.extend(["-ss", f"{candidate.start + candidate.duration / 2:.3f}"])
            command.extend(
                [
                    "-i",
                    str(poster_source.resolve()),
                    "-filter_complex",
                    build_vertical_poster_filter_graph(),
                    "-map",
                    "[poster]",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(temporary_poster),
                ]
            )
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                relevant_error = "\n".join(stderr.strip().splitlines()[-12:])
                raise RenderError(
                    "FFmpeg failed while creating the vertical poster: "
                    f"{relevant_error or exc}"
                ) from exc
            if not temporary_poster.is_file():
                raise RenderError("FFmpeg did not create the vertical poster.")
            os.replace(temporary_poster, poster_path)

        manifest["poster_fingerprint"] = poster_fingerprint
        manifest["poster_output_fingerprint"] = fingerprint_file(poster_path)
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        return poster_path

    def render(
        self,
        video: DownloadedVideo,
        transcript: TranscriptDocument,
        candidate: ClipCandidate,
        output_dir: Path,
        index: int,
        video_layout: VideoLayout = VideoLayout.FILL_CROP,
        force: bool = False,
        width: int = 1080,
        height: int = 1920,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (output_dir / f"{index:02d}-{slugify(candidate.title)}.mp4").resolve()
        thumbnail_path = output_path.with_suffix(".thumbnail.jpg")
        manifest_path = output_path.with_suffix(".render.json")
        try:
            render_fingerprint = self._render_fingerprint(
                video, transcript, candidate, video_layout, width, height
            )
        except OSError as exc:
            raise RenderError(f"Could not fingerprint render inputs: {exc}") from exc
        if not force and self._cached_output_is_valid(
            output_path,
            thumbnail_path,
            manifest_path,
            render_fingerprint,
            candidate.duration,
        ):
            self._ensure_vertical_poster(
                video,
                candidate,
                output_path,
                manifest_path,
                force=False,
            )
            return output_path
        output_path.unlink(missing_ok=True)
        thumbnail_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

        relative_segments = clip_transcript_segments(
            transcript.segments, candidate.start, candidate.end
        )
        temp_root = video.work_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"clip-{index:02d}-", dir=temp_root) as temp:
            temp_dir = Path(temp)
            caption_path = temp_dir / "captions.ass"
            thumbnail_title_path = temp_dir / "thumbnail-title.ass"
            temporary_thumbnail = temp_dir / "thumbnail.jpg"
            encoded_output = temp_dir / "encoded.mp4"
            temporary_output = temp_dir / "render.mp4"
            caption_path.write_text(
                build_ass_document(relative_segments, width=width, height=height),
                encoding="utf-8-sig",
            )
            thumbnail_title_path.write_text(
                build_thumbnail_ass_document(candidate.title),
                encoding="utf-8-sig",
            )

            thumbnail_source = (
                video.thumbnail_path
                if video.thumbnail_path and video.thumbnail_path.is_file()
                else video.video_path
            )
            thumbnail_command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if thumbnail_source == video.video_path:
                thumbnail_command.extend(
                    ["-ss", f"{candidate.start + candidate.duration / 2:.3f}"]
                )
            thumbnail_command.extend(
                [
                    "-i",
                    str(thumbnail_source.resolve()),
                    "-vf",
                    build_thumbnail_filter_graph(),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(temporary_thumbnail),
                ]
            )
            try:
                subprocess.run(
                    thumbnail_command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                relevant_error = "\n".join(stderr.strip().splitlines()[-12:])
                raise RenderError(
                    f"FFmpeg failed while creating thumbnail {index}: "
                    f"{relevant_error or exc}"
                ) from exc
            if not temporary_thumbnail.is_file():
                raise RenderError(f"FFmpeg did not create thumbnail {index}.")

            filter_graph = build_video_filter_graph(video_layout, width, height)
            command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{candidate.start:.3f}",
                "-t",
                f"{candidate.duration:.3f}",
                "-i",
                str(video.video_path.resolve()),
                "-filter_complex",
                filter_graph,
                "-map",
                "[video]",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                "-shortest",
                str(encoded_output),
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                encoded_output.unlink(missing_ok=True)
                stderr = getattr(exc, "stderr", "") or ""
                relevant_error = "\n".join(stderr.strip().splitlines()[-12:])
                raise RenderError(
                    f"FFmpeg failed while rendering clip {index}: "
                    f"{relevant_error or exc}"
                ) from exc

            if result.returncode != 0 or not encoded_output.is_file():
                raise RenderError(f"FFmpeg did not create clip {index}.")

            mux_command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(encoded_output),
                "-i",
                str(temporary_thumbnail),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-map",
                "1:v:0",
                "-c",
                "copy",
                "-disposition:v:1",
                "attached_pic",
                "-metadata",
                f"title={candidate.title}",
                "-metadata:s:v:1",
                f"title={candidate.title}",
                "-metadata:s:v:1",
                "comment=Cover (front)",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            try:
                subprocess.run(
                    mux_command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                relevant_error = "\n".join(stderr.strip().splitlines()[-12:])
                raise RenderError(
                    f"FFmpeg failed while embedding thumbnail {index}: "
                    f"{relevant_error or exc}"
                ) from exc
            try:
                probe = probe_media(self._media_tools, temporary_output)
            except MediaProbeError as exc:
                raise RenderError(f"Rendered clip {index} failed validation: {exc}") from exc
            if (
                not probe.has_video
                or not probe.has_attached_picture
                or abs(probe.duration_seconds - candidate.duration) > 0.75
            ):
                raise RenderError(
                    f"Rendered clip {index} has invalid streams, cover art, or duration."
                )

            os.replace(temporary_output, output_path)
            os.replace(temporary_thumbnail, thumbnail_path)
            manifest = {
                "schema_version": _RENDER_SCHEMA_VERSION,
                "video_layout": video_layout.value,
                "render_fingerprint": render_fingerprint,
                "output_fingerprint": fingerprint_file(output_path),
                "thumbnail_fingerprint": fingerprint_file(thumbnail_path),
            }
            atomic_write_text(
                manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True),
            )

        self._ensure_vertical_poster(
            video,
            candidate,
            output_path,
            manifest_path,
            force=force,
        )
        return output_path

    def render_montage(
        self,
        video: DownloadedVideo,
        transcript: TranscriptDocument,
        montage: HighlightMontage,
        output_dir: Path,
        video_layout: VideoLayout = VideoLayout.FILL_CROP,
        force: bool = False,
        width: int = 1080,
        height: int = 1920,
    ) -> Path:
        """Render non-contiguous AI-selected moments into one polished short video."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"highlight-montage-{slugify(montage.title)}.mp4"
        ).resolve()
        thumbnail_path = output_path.with_suffix(".thumbnail.jpg")
        manifest_path = output_path.with_suffix(".render.json")
        representative = montage.moments[0]
        artwork_candidate = ClipCandidate(
            title=montage.title,
            start=representative.start,
            end=representative.end,
            score=max(moment.score for moment in montage.moments),
            hook=representative.hook,
            reason=montage.summary,
        )
        try:
            render_fingerprint = fingerprint_payload(
                {
                    "schema_version": 1,
                    "source": fingerprint_file(video.video_path),
                    "source_thumbnail": (
                        fingerprint_file(video.thumbnail_path)
                        if video.thumbnail_path and video.thumbnail_path.is_file()
                        else "fallback-source-frame"
                    ),
                    "transcript": transcript.model_dump(mode="json"),
                    "montage": montage.model_dump(mode="json"),
                    "video_layout": video_layout.value,
                    "width": width,
                    "height": height,
                    "caption_style": "bold-yellow-charcoal-outline-v2",
                    "montage_style": "exact-windows-hard-cut-v1",
                }
            )
        except OSError as exc:
            raise RenderError(f"Could not fingerprint montage inputs: {exc}") from exc

        if not force and self._cached_output_is_valid(
            output_path,
            thumbnail_path,
            manifest_path,
            render_fingerprint,
            montage.duration,
        ):
            self._ensure_vertical_poster(
                video,
                artwork_candidate,
                output_path,
                manifest_path,
                force=False,
            )
            return output_path

        for path in (output_path, thumbnail_path, manifest_path):
            path.unlink(missing_ok=True)
        relative_segments = montage_transcript_segments(
            transcript.segments,
            montage.moments,
        )
        temp_root = video.work_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="montage-", dir=temp_root) as temp:
            temp_dir = Path(temp)
            caption_path = temp_dir / "captions.ass"
            thumbnail_title_path = temp_dir / "thumbnail-title.ass"
            temporary_thumbnail = temp_dir / "thumbnail.jpg"
            encoded_output = temp_dir / "encoded.mp4"
            temporary_output = temp_dir / "render.mp4"
            caption_path.write_text(
                build_ass_document(relative_segments, width=width, height=height),
                encoding="utf-8-sig",
            )
            thumbnail_title_path.write_text(
                build_thumbnail_ass_document(montage.title),
                encoding="utf-8-sig",
            )

            thumbnail_source = (
                video.thumbnail_path
                if video.thumbnail_path and video.thumbnail_path.is_file()
                else video.video_path
            )
            thumbnail_command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
            ]
            if thumbnail_source == video.video_path:
                thumbnail_command.extend(
                    [
                        "-ss",
                        f"{representative.start + representative.duration / 2:.3f}",
                    ]
                )
            thumbnail_command.extend(
                [
                    "-i",
                    str(thumbnail_source.resolve()),
                    "-vf",
                    build_thumbnail_filter_graph(),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(temporary_thumbnail),
                ]
            )
            try:
                subprocess.run(
                    thumbnail_command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                raise RenderError(
                    "FFmpeg failed while creating the montage thumbnail: "
                    + ("\n".join(stderr.strip().splitlines()[-12:]) or str(exc))
                ) from exc

            command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video.video_path.resolve()),
                "-filter_complex",
                build_montage_filter_graph(
                    montage.moments,
                    video_layout,
                    width,
                    height,
                ),
                "-map",
                "[video]",
                "-map",
                "[audio]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(encoded_output),
            ]
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                raise RenderError(
                    "FFmpeg failed while rendering the highlight montage: "
                    + ("\n".join(stderr.strip().splitlines()[-12:]) or str(exc))
                ) from exc

            mux_command = [
                str(self._media_tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(encoded_output),
                "-i",
                str(temporary_thumbnail),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-map",
                "1:v:0",
                "-c",
                "copy",
                "-disposition:v:1",
                "attached_pic",
                "-metadata",
                f"title={montage.title}",
                "-metadata:s:v:1",
                f"title={montage.title}",
                "-metadata:s:v:1",
                "comment=Cover (front)",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            try:
                subprocess.run(
                    mux_command,
                    cwd=temp_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
                probe = probe_media(self._media_tools, temporary_output)
            except (
                OSError,
                MediaProbeError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise RenderError(
                    f"Could not finalize the highlight montage: {exc}"
                ) from exc
            if (
                not probe.has_video
                or not probe.has_attached_picture
                or abs(probe.duration_seconds - montage.duration) > 0.75
            ):
                raise RenderError(
                    "Rendered highlight montage has invalid streams, artwork, or duration."
                )
            os.replace(temporary_output, output_path)
            os.replace(temporary_thumbnail, thumbnail_path)
            atomic_write_text(
                manifest_path,
                json.dumps(
                    {
                        "schema_version": _RENDER_SCHEMA_VERSION,
                        "video_layout": video_layout.value,
                        "render_fingerprint": render_fingerprint,
                        "output_fingerprint": fingerprint_file(output_path),
                        "thumbnail_fingerprint": fingerprint_file(thumbnail_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )

        self._ensure_vertical_poster(
            video,
            artwork_candidate,
            output_path,
            manifest_path,
            force=force,
        )
        return output_path

    @staticmethod
    def cleanup_stale(output_dir: Path, retained_paths: list[Path]) -> None:
        retained = {path.resolve() for path in retained_paths}
        retained.update(path.with_suffix(".render.json") for path in retained_paths)
        retained.update(path.with_suffix(".thumbnail.jpg") for path in retained_paths)
        retained.update(path.with_suffix(".poster.jpg") for path in retained_paths)
        for pattern in (
            "*.mp4",
            "*.render.json",
            "*.thumbnail.jpg",
            "*.poster.jpg",
        ):
            for path in output_dir.glob(pattern):
                if path.resolve() not in retained:
                    path.unlink(missing_ok=True)
