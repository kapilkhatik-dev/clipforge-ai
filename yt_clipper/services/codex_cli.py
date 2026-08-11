"""Structured, read-only transcript analysis through the local Codex CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from copy import deepcopy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEX_DEFAULT_MODEL = "codex/default"
_VERSION_TIMEOUT_SECONDS = 30


class CodexCLIError(Exception):
    """The local Codex executable was unavailable or returned no usable result."""


@dataclass(frozen=True)
class CodexCLIResult:
    payload: dict[str, Any] | list[Any]
    content: str


RunFunction = Callable[..., subprocess.CompletedProcess[str]]


def _strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the strict JSON Schema shape required by Codex structured output."""
    strict = deepcopy(schema)

    def normalize(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                normalize(item)
            return
        if not isinstance(node, dict):
            return

        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)

        for value in node.values():
            normalize(value)

    normalize(strict)
    return strict


class CodexCLIClient:
    """Invoke ``codex exec`` without exposing the project or persisting sessions."""

    def __init__(
        self,
        executable: str = "codex",
        *,
        timeout_seconds: int = 300,
        run_fn: RunFunction = subprocess.run,
    ) -> None:
        self._configured_executable = executable.strip() or "codex"
        self._timeout_seconds = timeout_seconds
        self._run = run_fn
        self._version: str | None = None
        self._version_lock = threading.Lock()

    def _executable(self) -> str:
        configured_path = Path(self._configured_executable).expanduser()
        if configured_path.is_file():
            return str(configured_path.resolve())
        discovered = shutil.which(self._configured_executable)
        if discovered:
            return str(Path(discovered).resolve())

        configured_name = configured_path.name.casefold()
        if os.name == "nt" and configured_name in {"codex", "codex.exe"}:
            bundled = self._windows_vscode_codex()
            if bundled is not None:
                return str(bundled)
        raise CodexCLIError(
            "Could not find the Codex CLI on PATH or in the installed VS Code "
            "OpenAI extension. Install Codex or set CLIPPER_CODEX_BINARY to the "
            "executable path."
        )

    @staticmethod
    def _windows_vscode_codex() -> Path | None:
        try:
            home = Path.home()
        except (OSError, RuntimeError):
            return None

        candidates: list[Path] = []
        for extensions_dir in (
            home / ".vscode" / "extensions",
            home / ".vscode-insiders" / "extensions",
        ):
            try:
                candidates.extend(
                    path.resolve()
                    for path in extensions_dir.glob(
                        "openai.chatgpt-*/bin/windows-*/codex.exe"
                    )
                    if path.is_file()
                )
            except OSError:
                continue

        if not candidates:
            return None
        return max(
            candidates,
            key=lambda path: (
                path.parents[2].name.casefold(),
                str(path).casefold(),
            ),
        )

    def _cli_version(self) -> str:
        with self._version_lock:
            if self._version is not None:
                return self._version
            executable = self._executable()
            try:
                result = self._run(
                    [executable, "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_VERSION_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CodexCLIError(f"Could not inspect the Codex CLI: {exc}") from exc
            if result.returncode != 0:
                raise CodexCLIError(
                    f"Could not inspect the Codex CLI: {self._error_text(result)}"
                )
            version = result.stdout.strip()
            if not version:
                raise CodexCLIError("The Codex CLI returned no version information.")
            self._version = version
            return version

    @staticmethod
    def _resolved_model(model: str) -> str | None:
        normalized = model.strip()
        if not normalized or normalized == CODEX_DEFAULT_MODEL:
            return None
        if normalized.startswith("codex/"):
            return normalized.removeprefix("codex/") or None
        return normalized

    def cache_identity(self, model: str) -> str:
        selected_model = self._resolved_model(model) or "default"
        return f"{self._cli_version()}:{selected_model}"

    @staticmethod
    def _error_text(result: subprocess.CompletedProcess[str]) -> str:
        output = (result.stderr or result.stdout or "").strip()
        if not output:
            return f"process exited with status {result.returncode}"
        return "\n".join(output.splitlines()[-12:])

    @staticmethod
    def _prompt(messages: Sequence[dict[str, str]], description: str) -> str:
        sections = [
            "Analyze only the supplied transcript task. Do not inspect files, run "
            "commands, browse, or use tools. Return only JSON matching the supplied "
            "output schema.",
            f"OUTPUT PURPOSE:\n{description}",
        ]
        for message in messages:
            role = message.get("role", "user").upper()
            sections.append(f"{role}:\n{message.get('content', '')}")
        return "\n\n".join(sections)

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        sensitive_markers = (
            "API_KEY",
            "ACCESS_TOKEN",
            "AUTH_TOKEN",
            "CREDENTIAL",
            "PASSWORD",
            "PRIVATE_KEY",
            "SECRET",
        )
        return {
            name: value
            for name, value in os.environ.items()
            if not any(marker in name.upper() for marker in sensitive_markers)
        }

    def request(
        self,
        *,
        messages: Sequence[dict[str, str]],
        model: str,
        output_schema: dict[str, Any],
        description: str,
        max_attempts: int,
    ) -> CodexCLIResult:
        executable = self._executable()
        prompt = self._prompt(messages, description)
        strict_schema = _strict_output_schema(output_schema)
        last_error = "Codex CLI request failed."

        for _attempt in range(max(1, max_attempts)):
            with tempfile.TemporaryDirectory(prefix="clipper-codex-") as temporary:
                temp_dir = Path(temporary)
                schema_path = temp_dir / "clip-schema.json"
                output_path = temp_dir / "clip-result.json"
                schema_path.write_text(
                    json.dumps(strict_schema, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                command = [
                    executable,
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--config",
                    "features.shell_tool=false",
                    "--config",
                    "agents.enabled=false",
                    "--config",
                    'web_search="disabled"',
                    "--config",
                    'approval_policy="never"',
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--cd",
                    str(temp_dir),
                ]
                selected_model = self._resolved_model(model)
                if selected_model:
                    command.extend(["--model", selected_model])
                command.append("-")

                try:
                    result = self._run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=self._sanitized_environment(),
                        timeout=self._timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    last_error = (
                        "Codex CLI analysis exceeded "
                        f"{self._timeout_seconds} seconds."
                    )
                    continue
                except OSError as exc:
                    raise CodexCLIError(f"Could not start the Codex CLI: {exc}") from exc

                if result.returncode != 0:
                    last_error = self._error_text(result)
                    if "not logged in" in last_error.casefold():
                        raise CodexCLIError(
                            "Codex CLI is not logged in. Run 'codex login', complete "
                            "the ChatGPT sign-in flow, then retry."
                        )
                    continue
                if not output_path.is_file():
                    last_error = "Codex CLI completed without writing structured output."
                    continue

                content = output_path.read_text(encoding="utf-8").strip()
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise CodexCLIError(
                        f"Codex CLI returned invalid structured JSON: {exc}"
                    ) from exc
                if not isinstance(payload, (dict, list)):
                    raise CodexCLIError(
                        "Codex CLI structured output must be a JSON object or array."
                    )
                return CodexCLIResult(payload=payload, content=content)

        raise CodexCLIError(last_error)
