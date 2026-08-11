from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from yt_clipper.domain.errors import MediaExtractionError, MediaProbeError
from yt_clipper.infrastructure.media_tools import (
    MediaTools,
    extract_mono_pcm,
    probe_media,
)


def test_extracts_bounded_mono_pcm_with_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_command: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        captured_kwargs.update(kwargs)
        _ = Path(command[-1]).write_bytes(b"\x00\x00" * 16_000)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tools = MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    destination = tmp_path / "audio.pcm"

    sample_count = extract_mono_pcm(
        tools,
        tmp_path / "source.mp4",
        destination,
        timeout_seconds=123,
        max_duration_seconds=1.0,
    )

    assert sample_count == 16_000
    assert ["-ac", "1"] == captured_command[
        captured_command.index("-ac") : captured_command.index("-ac") + 2
    ]
    assert ["-ar", "16000"] == captured_command[
        captured_command.index("-ar") : captured_command.index("-ar") + 2
    ]
    assert "pcm_s16le" in captured_command
    assert ["-t", "1.0"] == captured_command[
        captured_command.index("-t") : captured_command.index("-t") + 2
    ]
    assert captured_kwargs["timeout"] == 123


def test_rejects_pcm_output_longer_than_duration_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        _ = Path(command[-1]).write_bytes(b"\x00\x00" * 16_001)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tools = MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
    destination = tmp_path / "audio.pcm"

    with pytest.raises(MediaExtractionError, match="longer than"):
        extract_mono_pcm(
            tools,
            tmp_path / "source.mp4",
            destination,
            timeout_seconds=123,
            max_duration_seconds=1.0,
        )

    assert not destination.exists()


@pytest.mark.parametrize("duration", ["nan", "inf", "-inf"])
def test_probe_rejects_non_finite_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    duration: str,
) -> None:
    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        stdout = (
            '{"format": {"duration": "'
            + duration
            + '"}, "streams": [{"codec_type": "audio"}]}'
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tools = MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))

    with pytest.raises(MediaProbeError, match="invalid duration"):
        probe_media(tools, tmp_path / "source.mp4")


def test_probe_distinguishes_main_video_from_attached_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        stdout = (
            '{"format":{"duration":"30"},"streams":['
            '{"codec_type":"video","disposition":{"attached_pic":0}},'
            '{"codec_type":"audio","disposition":{"attached_pic":0}},'
            '{"codec_type":"video","disposition":{"attached_pic":1}}]}'
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = probe_media(
        MediaTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
        tmp_path / "clip.mp4",
    )

    assert probe.has_video
    assert probe.has_audio
    assert probe.has_attached_picture
