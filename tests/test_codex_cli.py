from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import yt_clipper.services.codex_cli as codex_cli_module
from yt_clipper.services.codex_cli import CodexCLIClient, CodexCLIError


@pytest.mark.skipif(os.name != "nt", reason="Windows VS Code discovery only")
def test_codex_cli_discovers_latest_vscode_extension_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    older = (
        tmp_path
        / ".vscode"
        / "extensions"
        / "openai.chatgpt-26.721-win32-x64"
        / "bin"
        / "windows-x86_64"
        / "codex.exe"
    )
    latest = (
        tmp_path
        / ".vscode"
        / "extensions"
        / "openai.chatgpt-26.803-win32-x64"
        / "bin"
        / "windows-x86_64"
        / "codex.exe"
    )
    older.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    older.write_bytes(b"older")
    latest.write_bytes(b"latest")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="codex-cli test\n",
            stderr="",
        )

    monkeypatch.setattr(codex_cli_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(codex_cli_module.Path, "home", lambda: tmp_path)

    client = CodexCLIClient("codex", run_fn=fake_run)

    assert client.cache_identity("codex/default") == "codex-cli test:default"
    assert commands[0][0] == str(latest.resolve())


def test_codex_cli_uses_ephemeral_toolless_structured_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"test executable")
    calls: list[tuple[list[str], dict[str, object]]] = []
    requested_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "standalone": {"type": "boolean", "default": False},
                        "topic": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clips"],
        "additionalProperties": False,
    }

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="codex-cli 1.2.3\n",
                stderr="",
            )

        schema_path = Path(command[command.index("--output-schema") + 1])
        written_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        item_schema = written_schema["properties"]["clips"]["items"]
        assert item_schema["required"] == ["title", "standalone", "topic"]
        assert item_schema["additionalProperties"] is False
        assert "default" not in item_schema["properties"]["standalone"]
        assert "default" not in item_schema["properties"]["topic"]
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"clips": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-codex")
    monkeypatch.setenv("SAFE_TEST_VALUE", "preserved")
    client = CodexCLIClient(
        str(executable),
        timeout_seconds=456,
        run_fn=fake_run,
    )

    assert client.cache_identity("codex/default") == "codex-cli 1.2.3:default"
    assert client.cache_identity("codex/default") == "codex-cli 1.2.3:default"
    result = client.request(
        messages=[
            {"role": "system", "content": "Select complete clips."},
            {"role": "user", "content": "[0.000 - 20.000] Transcript"},
        ],
        model="codex/default",
        output_schema=requested_schema,
        description="Submit clip candidates.",
        max_attempts=1,
    )

    assert result.payload == {"clips": []}
    original_items = requested_schema["properties"]["clips"]["items"]
    assert original_items["required"] == ["title"]
    assert original_items["properties"]["standalone"]["default"] is False
    assert len(calls) == 2
    command, kwargs = calls[1]
    assert command[:2] == [str(executable.resolve()), "exec"]
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert "features.shell_tool=false" in command
    assert "agents.enabled=false" in command
    assert 'web_search="disabled"' in command
    assert 'approval_policy="never"' in command
    assert "--model" not in command
    assert command[-1] == "-"
    assert kwargs["timeout"] == 456
    assert "Transcript" in str(kwargs["input"])
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert "OPENROUTER_API_KEY" not in environment
    assert environment["SAFE_TEST_VALUE"] == "preserved"


def test_codex_cli_passes_an_explicit_model_without_provider_prefix(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"test executable")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"clips": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    client = CodexCLIClient(str(executable), run_fn=fake_run)
    _ = client.request(
        messages=[{"role": "user", "content": "Transcript"}],
        model="codex/gpt-test",
        output_schema={"type": "object"},
        description="Submit clips.",
        max_attempts=1,
    )

    command = commands[0]
    assert command[command.index("--model") + 1] == "gpt-test"


def test_codex_cli_reports_missing_chatgpt_login_without_retrying(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"test executable")
    call_count = 0

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Not logged in",
        )

    client = CodexCLIClient(str(executable), run_fn=fake_run)

    with pytest.raises(CodexCLIError, match=r"codex login"):
        client.request(
            messages=[{"role": "user", "content": "Transcript"}],
            model="codex/default",
            output_schema={"type": "object"},
            description="Submit clips.",
            max_attempts=3,
        )

    assert call_count == 1
