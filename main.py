"""Debugger-friendly development runner for the video clipper pipeline.

Set CLIPPER_VIDEO_URL in .env, place breakpoints anywhere in ``yt_clipper``,
and launch the "Debug YouTube Clipper" target in Zed. Future frontends should
import the package directly instead of depending on this module.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from yt_clipper import (
    ClipPipeline,
    ContentType,
    PipelineConfig,
    PipelineEvent,
    PipelineResult,
    ProgressCallback,
    VideoLayout,
)


class ConsoleProgress:
    """Minimal progress adapter used only by the development runner."""

    def __init__(self) -> None:
        self._last_bucket: dict[str, int] = {}
        self._last_message: dict[str, str] = {}

    def __call__(self, event: PipelineEvent) -> None:
        stage = event.stage.value
        if event.progress is None:
            if self._last_message.get(stage) != event.message:
                print(f"[{stage}] {event.message}")
                self._last_message[stage] = event.message
            return

        bucket = min(10, int(event.progress * 10))
        if (
            self._last_bucket.get(stage) == bucket
            and self._last_message.get(stage) == event.message
        ):
            return
        self._last_bucket[stage] = bucket
        self._last_message[stage] = event.message
        print(f"[{stage}] {event.message} ({event.progress * 100:.0f}%)")


def run(
    video_url: str | None = None,
    *,
    config: PipelineConfig | None = None,
    video_layout: VideoLayout | None = None,
    clip_count: int | None = None,
    content_type: ContentType | str | None = None,
    highlight_montage: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run one video with values suitable for scripts, tests, or a debugger."""
    _ = load_dotenv()
    resolved_url = (video_url or os.getenv("CLIPPER_VIDEO_URL", "")).strip()
    if not resolved_url:
        raise ValueError(
            "Set CLIPPER_VIDEO_URL to a URL or local path, or pass video_url "
            "directly to main.run()."
        )

    if config is not None and (
        video_layout is not None
        or clip_count is not None
        or content_type is not None
        or highlight_montage is not None
    ):
        raise ValueError("Pass either config or individual runner settings, not both.")
    if config is not None:
        pipeline_config = config
    else:
        config_values: dict[str, object] = {}
        if video_layout is not None:
            config_values["video_layout"] = video_layout
        if clip_count is not None:
            config_values["clip_count"] = clip_count
        if content_type is not None:
            config_values["content_type"] = content_type
        if highlight_montage is not None:
            config_values["highlight_montage"] = highlight_montage
        pipeline_config = PipelineConfig.model_validate(config_values)
    callback = progress_callback if progress_callback is not None else ConsoleProgress()
    return ClipPipeline(
        pipeline_config,
        progress_callback=callback,
    ).run(resolved_url)


def print_result(result: PipelineResult) -> None:
    """Display development-runner output without coupling the core package to I/O."""
    print(f"\nTranscript: {result.transcript_path}")
    print(f"Candidates: {result.candidates_path}")
    if result.clip_paths:
        print("Clips:")
        for candidate, clip_path, thumbnail_path, poster_path in zip(
            result.candidates,
            result.clip_paths,
            result.thumbnail_paths,
            result.poster_paths,
            strict=True,
        ):
            print(f"  {candidate.title}")
            print(f"    Video: {clip_path}")
            print(f"    Thumbnail: {thumbnail_path}")
            print(f"    Vertical poster: {poster_path}")
    else:
        print("No clips were rendered.")
    if result.montage_video_path:
        print("Highlight montage:")
        print(f"  {result.montage.title if result.montage else 'Best moments'}")
        print(f"    Video: {result.montage_video_path}")
        print(f"    Thumbnail: {result.montage_thumbnail_path}")
        print(f"    Vertical poster: {result.montage_poster_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print_result(
        run(
            video_url="https://www.youtube.com/watch?v=aSR1tndcaLE",
            video_layout=VideoLayout.FILL_CROP,
        )
    )
