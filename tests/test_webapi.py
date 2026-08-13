from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from yt_clipper import PipelineEvent, PipelineStage
from yt_clipper.domain.errors import AnalysisError, DownloadError, SourceTransferError
from yt_clipper.domain.models import (
    ClipCandidate,
    DownloadedVideo,
    HighlightMoment,
    HighlightMontage,
    PipelineConfig,
    PipelineResult,
    TranscriptDocument,
    TranscriptOrigin,
    TranscriptSegment,
    VideoMetadata,
)
from yt_clipper.webapi.app import create_app
from yt_clipper.webapi.models import ProviderPatch, public_job_error
from yt_clipper.webapi.providers import ProviderRegistry


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLIPPER_LLM_PROVIDER",
        "CLIPPER_LLM_MODEL",
        "CLIPPER_LLM_API_KEY",
        "CLIPPER_CODEX_MODEL",
        "CLIPPER_NVIDIA_MODEL",
        "CLIPPER_OPENROUTER_MODEL",
        "CLIPPER_OPENAI_MODEL",
        "CLIPPER_ANTHROPIC_MODEL",
        "CLIPPER_MODEL",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _wait_for_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _configure_and_activate(
    client: TestClient,
    provider: str,
    *,
    api_key: str = "test-key",
) -> None:
    profile_id = f"{provider}-default"
    assert client.patch(
        f"/api/v1/provider-profiles/{profile_id}",
        json={"apiKey": api_key},
    ).status_code == 200
    assert client.post(
        f"/api/v1/provider-profiles/{profile_id}/test"
    ).json()["status"] == "healthy"
    assert client.patch(
        "/api/v1/settings/default-provider",
        json={"providerProfileId": profile_id},
    ).status_code == 200


def _successful_pipeline_factory(
    observed: dict[str, object], output_root: Path
):
    class FakePipeline:
        def __init__(self, config: PipelineConfig, callback: object) -> None:
            observed["config"] = config
            self._callback = callback

        def run(self, url: str) -> PipelineResult:
            observed["url"] = url
            callback = self._callback
            assert callable(callback)
            callback(PipelineEvent(stage=PipelineStage.INSPECT, message="Inspected", progress=1))
            callback(PipelineEvent(stage=PipelineStage.COMPLETE, message="Done", progress=1))
            return _sample_pipeline_result(output_root, url, b"video-data")

    return FakePipeline


def _sample_pipeline_result(
    output_root: Path,
    url: str,
    video_payload: bytes,
) -> PipelineResult:
    work_dir = output_root / "sample"
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clips_dir / "01-funny.mp4"
    thumb_path = clips_dir / "01-funny.thumbnail.jpg"
    poster_path = clips_dir / "01-funny.poster.jpg"
    clip_path.write_bytes(video_payload)
    thumb_path.write_bytes(b"thumbnail")
    poster_path.write_bytes(b"poster")
    metadata = VideoMetadata(
        video_id="sample",
        source_url=url,
        title="A funny source",
        duration_seconds=120,
        thumbnail_url="https://example.com/source.jpg",
    )
    downloaded = DownloadedVideo(
        metadata=metadata,
        video_path=work_dir / "source.mp4",
        metadata_path=work_dir / "metadata.json",
        work_dir=work_dir,
    )
    transcript = TranscriptDocument(
        video_id="sample",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=120,
        segments=[TranscriptSegment(start=10, end=40, text="A complete joke")],
    )
    candidate = ClipCandidate(
        title="The perfect punchline",
        start=10,
        end=40,
        score=0.95,
        hook="An immediate setup",
        reason="A strong punchline",
        standalone=True,
    )
    return PipelineResult(
        video=downloaded,
        transcript=transcript,
        candidates=[candidate],
        transcript_path=work_dir / "transcript.json",
        candidates_path=work_dir / "candidates.json",
        clip_paths=[clip_path],
        thumbnail_paths=[thumb_path],
        poster_paths=[poster_path],
    )


def test_provider_catalog_profiles_and_runtime_switching(tmp_path: Path) -> None:
    tested: list[PipelineConfig] = []

    def tester(config: PipelineConfig) -> tuple[bool, str]:
        tested.append(config)
        return bool(config.get_llm_api_key()), "Configuration checked"

    app = create_app(output_root=tmp_path, provider_tester=tester)
    with TestClient(app) as client:
        bootstrap = client.get("/api/v1/bootstrap").json()
        assert bootstrap["defaultProviderProfileId"] == "nvidia-default"
        assert bootstrap["capabilities"] == {
            "localUpload": False,
            "clipEditor": False,
        }

        providers = client.get("/api/v1/providers").json()
        assert {item["id"] for item in providers} == {
            "codex",
            "nvidia",
            "openrouter",
            "openai",
            "anthropic",
        }
        openrouter = next(item for item in providers if item["id"] == "openrouter")
        assert openrouter["requiresCredential"] is True
        assert openrouter["allowCustomModel"] is True
        assert "structured-output" in openrouter["capabilities"]
        assert openrouter["configurationFields"] == [
            {
                "key": "apiKey",
                "label": "API key",
                "inputType": "secret",
                "section": "Authentication",
                "sectionDescription": (
                    "Paste a new key only when you want to replace the existing credential."
                ),
                "helpText": (
                    "The key is write-only and remains in this local service session. "
                    "Use an environment variable for restart-persistent configuration."
                ),
                "placeholder": "Paste provider API key",
                "required": True,
                "writeOnly": True,
                "clearable": True,
                "minLength": 1,
                "maxLength": 8192,
                "minimum": None,
                "maximum": None,
                "step": None,
                "suffix": None,
            }
        ]

        response = client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"model": "openrouter/test/model", "apiKey": "private-key"},
        )
        assert response.status_code == 200
        profile = response.json()
        assert profile["credential"] == {"configured": True, "source": "runtime"}
        assert profile["configuration"]["apiKey"] == {
            "value": None,
            "configured": True,
            "source": "runtime",
        }
        assert "private-key" not in response.text

        generic = client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"configuration": {"apiKey": "generic-private-key"}},
        )
        assert generic.status_code == 200
        assert "generic-private-key" not in generic.text
        assert generic.json()["configuration"]["apiKey"]["value"] is None

        result = client.post(
            "/api/v1/provider-profiles/openrouter-default/test"
        ).json()
        assert result["status"] == "healthy"
        assert tested[-1].model == "openrouter/test/model"
        assert tested[-1].get_llm_api_key() == "generic-private-key"

        activated = client.patch(
            "/api/v1/settings/default-provider",
            json={"providerProfileId": "openrouter-default"},
        )
        assert activated.json() == {
            "defaultProviderProfileId": "openrouter-default"
        }
        profiles = client.get("/api/v1/provider-profiles").json()
        assert next(item for item in profiles if item["id"] == "openrouter-default")[
            "active"
        ] is True


def test_common_provider_overrides_do_not_leak_when_switching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("CLIPPER_LLM_MODEL", "openrouter/shared-model")
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "openrouter-shared-key")
    monkeypatch.setenv("CLIPPER_OPENAI_MODEL", "openai/provider-model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-provider-key")
    observed: list[PipelineConfig] = []

    def tester(config: PipelineConfig) -> tuple[bool, str]:
        observed.append(config)
        return bool(config.get_llm_api_key()), "Configuration checked"

    app = create_app(output_root=tmp_path, provider_tester=tester)
    with TestClient(app) as client:
        profiles = client.get("/api/v1/provider-profiles").json()
        openrouter = next(item for item in profiles if item["providerId"] == "openrouter")
        openai = next(item for item in profiles if item["providerId"] == "openai")
        assert openrouter["model"] == "openrouter/shared-model"
        assert openai["model"] == "openai/provider-model"

        result = client.post("/api/v1/provider-profiles/openai-default/test")
        assert result.status_code == 200
        assert observed[-1].model == "openai/provider-model"
        assert observed[-1].get_llm_api_key() == "openai-provider-key"

        activated = client.patch(
            "/api/v1/settings/default-provider",
            json={"providerProfileId": "openai-default"},
        )
        assert activated.status_code == 200


def test_provider_activation_requires_successful_setup_check(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        missing = client.patch(
            "/api/v1/settings/default-provider",
            json={"providerProfileId": "openai-default"},
        )
        assert missing.status_code == 409
        assert "Configure" in missing.json()["detail"]

        configured = client.patch(
            "/api/v1/provider-profiles/openai-default",
            json={"apiKey": "configured-key"},
        )
        assert configured.status_code == 200
        untested = client.patch(
            "/api/v1/settings/default-provider",
            json={"providerProfileId": "openai-default"},
        )
        assert untested.status_code == 409
        assert "setup check" in untested.json()["detail"]


def test_provider_model_isolation_and_atomic_patch_validation(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        wrong_provider = client.patch(
            "/api/v1/provider-profiles/openai-default",
            json={"model": "openrouter/not-an-openai-model"},
        )
        assert wrong_provider.status_code == 422
        assert "openai/" in wrong_provider.json()["detail"]

        partial_patch = client.patch(
            "/api/v1/provider-profiles/openai-default",
            json={
                "model": "openai/replacement",
                "codexBinary": "should-not-apply",
            },
        )
        assert partial_patch.status_code == 422
        profile = next(
            item
            for item in client.get("/api/v1/provider-profiles").json()
            if item["id"] == "openai-default"
        )
        assert profile["model"] == "openai/gpt-4.1-mini"

        unknown_field = client.patch(
            "/api/v1/provider-profiles/openai-default",
            json={"configuration": {"codexBinary": "should-not-apply"}},
        )
        assert unknown_field.status_code == 422
        assert "codexBinary" in unknown_field.json()["detail"]

        invalid_number = client.patch(
            "/api/v1/provider-profiles/codex-default",
            json={"configuration": {"codexTimeoutSeconds": 29}},
        )
        assert invalid_number.status_code == 422


def test_environment_model_mismatch_cannot_cross_provider_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CLIPPER_LLM_MODEL", "openrouter/wrong-provider")
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "openai-key")
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        check = client.post("/api/v1/provider-profiles/openai-default/test")
        assert check.status_code == 200
        assert check.json()["status"] == "unhealthy"

        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        generation = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openai-default", "options": {}},
        )
        assert generation.status_code == 422
        assert "openai/" in generation.json()["detail"]


def test_provider_setup_errors_and_validation_never_echo_secrets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "DO_NOT_ECHO_PROVIDER_SECRET"

    def tester(config: PipelineConfig) -> tuple[bool, str]:
        raise RuntimeError(f"upstream rejected {sentinel} at C:\\private\\provider.json")

    caplog.set_level("WARNING", logger="yt_clipper.webapi.providers")
    app = create_app(output_root=tmp_path, provider_tester=tester)
    with TestClient(app) as client:
        assert client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"apiKey": "configured-key"},
        ).status_code == 200
        checked = client.post("/api/v1/provider-profiles/openrouter-default/test")
        assert checked.status_code == 200
        assert checked.json()["status"] == "unhealthy"
        assert sentinel not in checked.text
        assert "provider.json" not in checked.text
        assert sentinel not in caplog.text
        assert "provider.json" not in caplog.text

        oversized = sentinel + ("x" * 8192)
        invalid = client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"apiKey": oversized},
        )
        assert invalid.status_code == 422
        assert sentinel not in invalid.text
        assert "input" not in invalid.json()["detail"]["errors"][0]

        generic_sentinel = "DO_NOT_ECHO_GENERIC_SECRET"
        generic_patch = ProviderPatch(
            configuration={"apiKey": generic_sentinel},
        )
        assert generic_sentinel not in repr(generic_patch)
        generic_invalid = client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"configuration": {"apiKey": generic_sentinel + ("x" * 8192)}},
        )
        assert generic_invalid.status_code == 422
        assert generic_sentinel not in generic_invalid.text
        assert generic_sentinel not in caplog.text


def test_stale_provider_check_cannot_mark_changed_settings_healthy() -> None:
    started = Event()
    release = Event()

    def tester(config: PipelineConfig) -> tuple[bool, str]:
        started.set()
        assert release.wait(2)
        return True, "ready"

    registry = ProviderRegistry(tester)
    registry.update(
        "openrouter-default",
        ProviderPatch(model="openrouter/old-model", api_key=SecretStr("old-key")),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(registry.test, "openrouter-default")
        assert started.wait(1)
        registry.update(
            "openrouter-default",
            ProviderPatch(model="openrouter/new-model", api_key=SecretStr("new-key")),
        )
        release.set()
        result = future.result(timeout=2)

    assert result.status == "unhealthy"
    assert registry.profile("openrouter-default").last_test is None
    with pytest.raises(ValueError, match="setup check"):
        registry.activate("openrouter-default")


def test_generation_requires_hosted_provider_credentials(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        generation = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        )
        assert generation.status_code == 422
        assert "Configure" in generation.json()["detail"]


def test_cross_site_mutations_are_blocked_without_web_client_header(
    tmp_path: Path,
) -> None:
    app = create_app(output_root=tmp_path)
    payload = {
        "source": {
            "kind": "youtube",
            "url": "https://youtu.be/abcdefghijk",
        }
    }
    with TestClient(app) as client:
        blocked = client.post(
            "/api/v1/projects",
            json=payload,
            headers={"Origin": "https://attacker.example"},
        )
        assert blocked.status_code == 403
        assert blocked.json()["detail"]["code"] == "cross_site_request_blocked"

        allowed = client.post(
            "/api/v1/projects",
            json=payload,
            headers={
                "Origin": "https://attacker.example",
                "X-ClipForge-Client": "web",
            },
        )
        assert allowed.status_code == 201

        same_origin = client.post(
            "/api/v1/projects",
            json=payload,
            headers={"Origin": "http://testserver"},
        )
        assert same_origin.status_code == 201


def test_project_generation_job_sse_and_opaque_assets(tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    factory = _successful_pipeline_factory(observed, tmp_path)
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=factory,
        max_workers=1,
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter", api_key="test-openrouter-key")
        project_response = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                }
            },
        )
        assert project_response.status_code == 201
        project_id = project_response.json()["id"]

        generation = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={
                "providerProfileId": "openrouter-default",
                "options": {
                    "clipCount": None,
                    "contentType": "comedy",
                    "videoLayout": "fill-crop",
                    "minClipDuration": 20,
                    "maxClipDuration": 60,
                    "highlightMontage": False,
                    "highlightWindowSeconds": 4,
                    "highlightMontageMaxDuration": 60,
                    "highlightMontageMaxMoments": 12,
                    "transcriptMode": "auto",
                    "analysisMaxConcurrency": 2,
                    "analysisRequestMaxAttempts": 3,
                    "force": False,
                },
            },
        )
        assert generation.status_code == 202
        job_id = generation.json()["id"]
        job = _wait_for_terminal(client, job_id)
        assert job["status"] == "completed"
        assert app.state.jobs._jobs[job_id].config is None
        assert job["stageProgress"] == 1
        assert job["result"]["clips"][0]["title"] == "The perfect punchline"
        assert job["result"]["source"]["thumbnailUrl"] is None
        assert "example.com/source.jpg" not in json.dumps(job)
        assert job["result"]["clips"][0]["editor"]["route"].startswith(
            f"/projects/{project_id}/"
        )
        config = observed["config"]
        assert isinstance(config, PipelineConfig)
        assert config.clip_count is None
        assert config.content_type.value == "comedy"
        assert config.llm_provider.value == "openrouter"

        asset_url = job["result"]["clips"][0]["assets"]["videoUrl"]
        assert asset_url.startswith("/api/v1/assets/")
        assert str(tmp_path) not in json.dumps(job)
        asset_response = client.get(asset_url)
        assert asset_response.status_code == 200
        assert asset_response.content == b"video-data"
        range_response = client.get(asset_url, headers={"Range": "bytes=0-4"})
        assert range_response.status_code == 206
        assert range_response.headers["content-range"] == "bytes 0-4/10"
        assert range_response.content == b"video"
        # Public asset IDs point at per-job snapshots, not mutable pipeline files.
        (tmp_path / "sample" / "clips" / "01-funny.mp4").write_bytes(b"replaced")
        assert client.get(asset_url).content == b"video-data"
        assert client.get("/api/v1/assets/../../.env").status_code == 404

        with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
            event_ids = [
                int(line.removeprefix("id: "))
                for line in stream.iter_lines()
                if line.startswith("id: ")
            ]
        with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
            data_lines = [
                line.removeprefix("data: ")
                for line in stream.iter_lines()
                if line.startswith("data: ")
            ]
        events = [json.loads(line) for line in data_lines]
        assert event_ids == sorted(event_ids)
        assert event_ids[-1] == events[-1]["sequence"]
        assert events[-1]["type"] == "completed"
        assert events[-1]["job"]["status"] == "completed"

        with client.stream(
            "GET",
            f"/api/v1/jobs/{job_id}/events",
            headers={"Last-Event-ID": str(event_ids[-2])},
        ) as resumed:
            resumed_data = [
                json.loads(line.removeprefix("data: "))
                for line in resumed.iter_lines()
                if line.startswith("data: ")
            ]
        assert [item["sequence"] for item in resumed_data] == [event_ids[-1]]


def test_same_source_jobs_snapshot_assets_before_next_run_mutates_workspace(
    tmp_path: Path,
) -> None:
    first_entered = Event()
    release_first = Event()
    calls_lock = Lock()
    calls = 0

    class SharedWorkspacePipeline:
        def run(self, url: str) -> PipelineResult:
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            payload = b"first-version" if call_number == 1 else b"second-version"
            result = _sample_pipeline_result(tmp_path, url, payload)
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(2)
            return result

    def factory(
        config: PipelineConfig,
        callback: object,
    ) -> SharedWorkspacePipeline:
        del config, callback
        return SharedWorkspacePipeline()

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=factory,
        max_workers=2,
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter", api_key="test-openrouter-key")
        source = {
            "source": {
                "kind": "youtube",
                "url": "https://youtu.be/abcdefghijk",
            }
        }
        project_one = client.post("/api/v1/projects", json=source).json()["id"]
        project_two = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                }
            },
        ).json()["id"]
        first_job = client.post(
            f"/api/v1/projects/{project_one}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        assert first_entered.wait(1)
        second_job = client.post(
            f"/api/v1/projects/{project_two}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        time.sleep(0.05)
        with calls_lock:
            assert calls == 1

        release_first.set()
        first = _wait_for_terminal(client, first_job)
        second = _wait_for_terminal(client, second_job)
        assert first["status"] == second["status"] == "completed"
        first_url = first["result"]["clips"][0]["assets"]["videoUrl"]
        second_url = second["result"]["clips"][0]["assets"]["videoUrl"]
        assert client.get(first_url).content == b"first-version"
        assert client.get(second_url).content == b"second-version"


def test_close_marks_queued_jobs_terminal(tmp_path: Path) -> None:
    running = Event()
    release = Event()
    base_factory = _successful_pipeline_factory({}, tmp_path)

    def factory(config: PipelineConfig, callback: object):
        pipeline = base_factory(config, callback)

        class BlockingPipeline:
            def run(self, url: str) -> PipelineResult:
                running.set()
                assert release.wait(2)
                return pipeline.run(url)

        return BlockingPipeline()

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=factory,
        max_workers=1,
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter", api_key="test-openrouter-key")
        source = {
            "source": {
                "kind": "youtube",
                "url": "https://youtu.be/abcdefghijk",
            }
        }
        project_one = client.post("/api/v1/projects", json=source).json()["id"]
        project_two = client.post("/api/v1/projects", json=source).json()["id"]
        first_job = client.post(
            f"/api/v1/projects/{project_one}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        assert running.wait(1)
        queued_job = client.post(
            f"/api/v1/projects/{project_two}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]

        app.state.jobs.close()
        queued = client.get(f"/api/v1/jobs/{queued_job}").json()
        assert queued["status"] == "failed"
        assert queued["error"]["code"] == "service_shutdown"
        assert app.state.jobs._jobs[queued_job].config is None

        release.set()
        assert _wait_for_terminal(client, first_job)["status"] == "completed"


def test_rejects_local_sources_unknown_ids_and_invalid_duration(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        assert client.post(
            "/api/v1/projects",
            json={"source": {"kind": "youtube", "url": "http://127.0.0.1/a"}},
        ).status_code == 422
        assert client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://user:password@youtube.com/watch?v=test",
                }
            },
        ).status_code == 422
        assert client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://example.com/video",
                }
            },
        ).status_code == 422
        assert client.get("/api/v1/jobs/missing").status_code == 404
        assert client.patch(
            "/api/v1/provider-profiles/missing",
            json={"model": "anything"},
        ).status_code == 404

        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        response = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={
                "providerProfileId": "nvidia-default",
                "options": {"minClipDuration": 50, "maxClipDuration": 20},
            },
        )
        assert response.status_code == 422


def test_job_errors_do_not_expose_upstream_secrets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingPipeline:
        def run(self, url: str) -> PipelineResult:
            raise AnalysisError(
                "provider rejected api_key=private-value at C:\\private\\config.json"
            )

    def factory(config: PipelineConfig, callback: object) -> FailingPipeline:
        return FailingPipeline()

    caplog.set_level("WARNING", logger="yt_clipper.webapi.jobs")
    app = create_app(output_root=tmp_path, pipeline_factory=factory)
    with TestClient(app) as client:
        _configure_and_activate(client, "nvidia", api_key="test-nvidia-key")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "nvidia-default", "options": {}},
        ).json()["id"]
        job = _wait_for_terminal(client, job_id)
        assert app.state.jobs._jobs[job_id].config is None
        serialized = json.dumps(job)
        assert job["error"] == {
            "code": "analysis_error",
            "message": "The selected AI provider could not analyze this video.",
        }
        assert "private-value" not in serialized
        assert "config.json" not in serialized
        assert "analysis_error" in caplog.text
        assert "AnalysisError" in caplog.text
        assert "private-value" not in caplog.text
        assert "config.json" not in caplog.text


def test_source_transfer_error_is_actionable_and_sanitized() -> None:
    sentinel = "https://signed.googlevideo.example/private-token"

    error = public_job_error(SourceTransferError(f"HTTP 403 for {sentinel}"))

    assert error.code == "source_transfer_error"
    assert error.message == (
        "YouTube temporarily rejected the media transfer. "
        "Please try this generation again."
    )
    assert sentinel not in error.model_dump_json()


def test_permanent_download_error_remains_generic_and_sanitized() -> None:
    sentinel = "https://signed.googlevideo.example/private-token"

    error = public_job_error(
        DownloadError(f"Requested format is unavailable at {sentinel}")
    )

    assert error.code == "download_error"
    assert error.message == "The source video could not be inspected or downloaded."
    assert sentinel not in error.model_dump_json()


def test_serves_built_spa_with_history_fallback(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    web_dist = tmp_path / "dist"
    (web_dist / "assets").mkdir(parents=True)
    (web_dist / "index.html").write_text("<main>ClipForge</main>", encoding="utf-8")
    (web_dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    app = create_app(output_root=output_root, web_dist=web_dist)
    with TestClient(app) as client:
        assert client.get("/").text == "<main>ClipForge</main>"
        assert client.get("/projects/example/clips/one/edit").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/v1/unknown-route").status_code == 404
        assert client.get("/api/v1/bootstrap").headers["content-type"].startswith(
            "application/json"
        )


def test_local_api_rejects_untrusted_hosts(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app, base_url="http://attacker.example") as client:
        response = client.get("/api/v1/bootstrap")
    assert response.status_code == 400


def test_completed_project_job_and_assets_survive_service_restart(
    tmp_path: Path,
) -> None:
    secret = "runtime-key-must-not-be-persisted"
    first_app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(first_app) as client:
        _configure_and_activate(client, "openrouter", api_key=secret)
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        completed = _wait_for_terminal(client, job_id)
        asset_url = completed["result"]["clips"][0]["assets"]["videoUrl"]

    database_bytes = (tmp_path / "_web_state" / "clipforge.sqlite3").read_bytes()
    assert secret.encode() not in database_bytes

    restored_app = create_app(output_root=tmp_path)
    with TestClient(restored_app) as client:
        project = client.get(f"/api/v1/projects/{project_id}")
        assert project.status_code == 200
        assert project.json()["source"] == {
            "kind": "youtube",
            "url": "https://youtu.be/abcdefghijk",
        }
        assert project.json()["jobs"][0]["id"] == job_id
        assert project.json()["jobs"][0]["result"]["clips"][0]["title"]
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
        assert client.get(asset_url).content == b"video-data"

        deleted = client.delete(f"/api/v1/jobs/{job_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404
        assert client.get(asset_url).status_code == 404
        # Projects are lifecycle containers for jobs; deleting the final job
        # prunes the now-empty project rather than leaving an orphaned draft.
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 404


def test_retention_keeps_newest_completed_web_job(tmp_path: Path) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        jobs: list[str] = []
        for suffix in ("abcdefghijk", "lmnopqrstuv"):
            project_id = client.post(
                "/api/v1/projects",
                json={
                    "source": {
                        "kind": "youtube",
                        "url": f"https://youtu.be/{suffix}",
                    }
                },
            ).json()["id"]
            job_id = client.post(
                f"/api/v1/projects/{project_id}/generations",
                json={"providerProfileId": "openrouter-default", "options": {}},
            ).json()["id"]
            assert _wait_for_terminal(client, job_id)["status"] == "completed"
            jobs.append(job_id)

        retained = client.post(
            "/api/v1/maintenance/retention",
            json={"maxJobs": 1},
        )
        assert retained.status_code == 200
        assert retained.json()["deletedJobIds"] == [jobs[0]]
        assert retained.json()["reclaimedBytes"] > 0
        assert client.get(f"/api/v1/jobs/{jobs[0]}").status_code == 404
        assert client.get(f"/api/v1/jobs/{jobs[1]}").status_code == 200


def test_delete_rejects_running_job(tmp_path: Path) -> None:
    entered = Event()
    release = Event()

    class BlockingPipeline:
        def run(self, url: str) -> PipelineResult:
            entered.set()
            assert release.wait(2)
            return _sample_pipeline_result(tmp_path, url, b"video-data")

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=lambda config, callback: BlockingPipeline(),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        assert entered.wait(1)
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 409
        release.set()
        assert _wait_for_terminal(client, job_id)["status"] == "completed"


def test_system_diagnostics_are_display_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    import yt_clipper.webapi.app as app_module
    from yt_clipper.infrastructure.media_tools import MediaTools

    monkeypatch.setattr(
        app_module,
        "resolve_media_tools",
        lambda: MediaTools(ffmpeg=Path("private/ffmpeg"), ffprobe=Path("private/ffprobe")),
    )
    monkeypatch.setattr(
        app_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args=args, returncode=0),
    )
    app = create_app(
        output_root=tmp_path,
        provider_tester=lambda config: (True, "ready"),
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/system/diagnostics")
    assert response.status_code == 200
    diagnostics = response.json()
    assert diagnostics["status"] == "healthy"
    assert diagnostics["mediaTools"]["ffmpeg"]["status"] == "healthy"
    assert diagnostics["mediaTools"]["ffprobe"]["status"] == "healthy"
    assert diagnostics["output"]["writable"] is True
    assert diagnostics["output"]["freeBytes"] > 0
    assert "private" not in response.text


def test_close_flushes_and_closes_local_state_database(tmp_path: Path) -> None:
    app = create_app(output_root=tmp_path)
    with TestClient(app) as client:
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
    assert app.state.jobs._database_closed is True

    restored = create_app(output_root=tmp_path)
    with TestClient(restored) as client:
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 200


def test_executor_submission_failure_rolls_back_durable_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        output_root=tmp_path,
        provider_tester=lambda config: (True, "ready"),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]

        def reject_submission(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("executor unavailable")

        service = app.state.jobs
        monkeypatch.setattr(service._executor, "submit", reject_submission)
        response = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        )

        assert response.status_code == 422
        with service._lock:
            persisted_jobs = service._database.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]
        assert persisted_jobs == 0
        assert service._jobs == {}

    restored = create_app(output_root=tmp_path)
    with TestClient(restored) as client:
        project = client.get(f"/api/v1/projects/{project_id}")
        assert project.status_code == 200
        assert project.json()["jobs"] == []


def test_startup_removes_crash_staging_and_stale_asset_rows(tmp_path: Path) -> None:
    import sqlite3

    initial = create_app(output_root=tmp_path)
    with TestClient(initial):
        pass

    staging = tmp_path / "_web_jobs" / ".job_interrupted"
    staging.mkdir(parents=True)
    (staging / "partial.mp4").write_bytes(b"partial")
    database_path = tmp_path / "_web_state" / "clipforge.sqlite3"
    with sqlite3.connect(database_path) as database:
        database.execute(
            "PRAGMA foreign_keys=ON"
        )
        database.execute(
            "INSERT INTO projects(id, source_url, created_at) VALUES (?, ?, ?)",
            (
                "prj_stale",
                "https://youtu.be/abcdefghijk",
                "2026-08-12T00:00:00+00:00",
            ),
        )
        response = {
            "id": "job_stale",
            "projectId": "prj_stale",
            "status": "completed",
            "createdAt": "2026-08-12T00:00:00Z",
            "result": {"clips": []},
        }
        database.execute(
            "INSERT INTO jobs(id, project_id, source_url, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "job_stale",
                "prj_stale",
                "https://youtu.be/abcdefghijk",
                json.dumps(response),
                "2026-08-12T00:00:00+00:00",
            ),
        )
        database.execute(
            "INSERT INTO assets(id, job_id, relative_path, media_type) "
            "VALUES (?, ?, ?, ?)",
            ("asset_missing", "job_stale", "_web_jobs/job_stale/missing.mp4", "video/mp4"),
        )

    restored = create_app(output_root=tmp_path)
    with TestClient(restored) as client:
        assert client.get("/api/v1/assets/asset_missing").status_code == 404
    assert not staging.exists()
    with sqlite3.connect(database_path) as database:
        assert database.execute(
            "SELECT COUNT(*) FROM assets WHERE id = 'asset_missing'"
        ).fetchone()[0] == 0


def test_interrupted_job_is_durably_restored_as_failed(tmp_path: Path) -> None:
    import sqlite3

    initial = create_app(output_root=tmp_path)
    with TestClient(initial):
        pass
    database_path = tmp_path / "_web_state" / "clipforge.sqlite3"
    snapshot = tmp_path / "_web_jobs" / "job_interrupted"
    snapshot.mkdir(parents=True)
    partial = snapshot / "partial.mp4"
    partial.write_bytes(b"partial")
    response = {
        "id": "job_interrupted",
        "projectId": "prj_interrupted",
        "status": "running",
        "stage": "download",
        "createdAt": "2026-08-12T00:00:00Z",
        "startedAt": "2026-08-12T00:00:01Z",
    }
    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys=ON")
        database.execute(
            "INSERT INTO projects(id, source_url, created_at) VALUES (?, ?, ?)",
            (
                "prj_interrupted",
                "https://youtu.be/abcdefghijk",
                "2026-08-12T00:00:00+00:00",
            ),
        )
        database.execute(
            "INSERT INTO jobs(id, project_id, source_url, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "job_interrupted",
                "prj_interrupted",
                "https://youtu.be/abcdefghijk",
                json.dumps(response),
                "2026-08-12T00:00:00+00:00",
            ),
        )
        database.execute(
            "INSERT INTO assets(id, job_id, relative_path, media_type) "
            "VALUES (?, ?, ?, ?)",
            (
                "asset_partial",
                "job_interrupted",
                "_web_jobs/job_interrupted/partial.mp4",
                "video/mp4",
            ),
        )

    for _ in range(2):
        restored = create_app(output_root=tmp_path)
        with TestClient(restored) as client:
            job = client.get("/api/v1/jobs/job_interrupted").json()
            assert job["status"] == "failed"
            assert job["error"]["code"] == "service_restart"
            assert client.get("/api/v1/assets/asset_partial").status_code == 404
    assert not snapshot.exists()


def test_active_provider_edit_requires_check_and_explicit_reactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")
    registry = ProviderRegistry(lambda config: (True, "ready"))

    assert registry.profile("openrouter-default").generation_ready is True
    registry.update(
        "openrouter-default",
        ProviderPatch(model="openrouter/new-model"),
    )
    assert registry.profile("openrouter-default").generation_ready is False
    with pytest.raises(ValueError, match="re-apply"):
        registry.generation_config("openrouter-default")

    assert registry.test("openrouter-default").status == "healthy"
    assert registry.profile("openrouter-default").generation_ready is False
    registry.activate("openrouter-default")
    assert registry.profile("openrouter-default").generation_ready is True
    assert registry.generation_config("openrouter-default").model == "openrouter/new-model"


def test_inactive_provider_cannot_be_used_by_direct_generation_api(
    tmp_path: Path,
) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
        provider_tester=lambda config: (True, "ready"),
    )
    with TestClient(app) as client:
        assert client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"apiKey": "configured-key"},
        ).status_code == 200
        assert client.post(
            "/api/v1/provider-profiles/openrouter-default/test"
        ).json()["status"] == "healthy"
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]

        rejected = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        )
        assert rejected.status_code == 422
        assert "active" in rejected.json()["detail"]


def test_failed_snapshot_transaction_removes_files_and_asset_rows(
    tmp_path: Path,
) -> None:
    class MissingSecondAssetPipeline:
        def run(self, url: str) -> PipelineResult:
            result = _sample_pipeline_result(tmp_path, url, b"first-video")
            second = result.candidates[0].model_copy(
                update={"title": "Missing render", "start": 50, "end": 80}
            )
            return result.model_copy(
                update={
                    "candidates": [result.candidates[0], second],
                    "clip_paths": [
                        result.clip_paths[0],
                        tmp_path / "sample" / "clips" / "02-missing.mp4",
                    ],
                }
            )

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=lambda config, callback: MissingSecondAssetPipeline(),
        provider_tester=lambda config: (True, "ready"),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        assert _wait_for_terminal(client, job_id)["status"] == "failed"
        assert app.state.jobs._jobs[job_id].config is None
        assert not (tmp_path / "_web_jobs" / job_id).exists()
        assert not list((tmp_path / "_web_jobs").glob(f".{job_id}-*"))
        assert app.state.jobs._database.execute(
            "SELECT COUNT(*) FROM assets WHERE job_id = ?", (job_id,)
        ).fetchone()[0] == 0


def test_codex_setup_check_does_not_claim_login_or_model_access() -> None:
    registry = ProviderRegistry(lambda config: (True, "available"))

    result = registry.test("codex-default")

    assert result.status == "healthy"
    assert result.message is not None
    assert "executable" in result.message
    assert "will be verified when analysis starts" in result.message


def test_public_results_include_versioned_non_destructive_edit_decisions(
    tmp_path: Path,
) -> None:
    class MontagePipeline:
        def run(self, url: str) -> PipelineResult:
            result = _sample_pipeline_result(tmp_path, url, b"continuous")
            montage_dir = tmp_path / "sample" / "montage"
            montage_dir.mkdir(parents=True, exist_ok=True)
            montage_video = montage_dir / "best-moments.mp4"
            montage_thumbnail = montage_dir / "best-moments.thumbnail.jpg"
            montage_poster = montage_dir / "best-moments.poster.jpg"
            montage_video.write_bytes(b"montage")
            montage_thumbnail.write_bytes(b"thumbnail")
            montage_poster.write_bytes(b"poster")
            montage = HighlightMontage(
                title="Best moments",
                summary="Two moments from across the source",
                moments=[
                    HighlightMoment(
                        start=5,
                        end=9,
                        score=0.98,
                        hook="First hook",
                        reason="First reason",
                    ),
                    HighlightMoment(
                        start=70,
                        end=75,
                        score=0.96,
                        hook="Second hook",
                        reason="Second reason",
                    ),
                ],
            )
            return result.model_copy(
                update={
                    "montage": montage,
                    "montage_video_path": montage_video,
                    "montage_thumbnail_path": montage_thumbnail,
                    "montage_poster_path": montage_poster,
                }
            )

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=lambda config, callback: MontagePipeline(),
        provider_tester=lambda config: (True, "ready"),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={
                "providerProfileId": "openrouter-default",
                "options": {"videoLayout": "fit-blur"},
            },
        ).json()["id"]
        result = _wait_for_terminal(client, job_id)["result"]

    continuous_edl = result["clips"][0]["editDecisionList"]
    assert continuous_edl == {
        "version": 1,
        "kind": "continuous",
        "sourceVideoId": "sample",
        "sourceRanges": [{"start": 10.0, "end": 40.0}],
        "videoLayout": "fit-blur",
        "captionPreset": "bold-yellow-charcoal-outline-v2",
    }
    montage_edl = result["montage"]["editDecisionList"]
    assert montage_edl["kind"] == "montage"
    assert montage_edl["sourceRanges"] == [
        {"start": 5.0, "end": 9.0},
        {"start": 70.0, "end": 75.0},
    ]


def test_live_provider_probe_is_explicit_sanitized_and_non_mutating(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup_calls: list[PipelineConfig] = []
    live_calls: list[PipelineConfig] = []

    def setup(config: PipelineConfig) -> tuple[bool, str]:
        setup_calls.append(config)
        return True, "configured"

    def live(config: PipelineConfig) -> tuple[bool, str]:
        live_calls.append(config)
        return True, "connected"

    app = create_app(
        output_root=tmp_path,
        provider_tester=setup,
        provider_live_probe=live,
    )
    with TestClient(app) as client:
        assert client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"apiKey": "private-live-key"},
        ).status_code == 200
        configured = client.post(
            "/api/v1/provider-profiles/openrouter-default/test"
        )
        assert configured.json()["status"] == "healthy"
        assert len(setup_calls) == 1
        assert live_calls == []

        probed = client.post(
            "/api/v1/provider-profiles/openrouter-default/test?live=true"
        )
        assert probed.json()["status"] == "healthy"
        assert "live request" in probed.json()["message"]
        assert len(live_calls) == 1
        assert live_calls[0].get_llm_api_key() == "private-live-key"
        profile = next(
            item
            for item in client.get("/api/v1/provider-profiles").json()
            if item["id"] == "openrouter-default"
        )
        assert profile["lastTest"] == configured.json()

    sentinel = "DO_NOT_LOG_LIVE_PROBE_SECRET"

    def failing_live(config: PipelineConfig) -> tuple[bool, str]:
        raise RuntimeError(f"{sentinel}: C:\\private\\provider.json")

    caplog.set_level("WARNING", logger="yt_clipper.webapi.providers")
    failing_app = create_app(
        output_root=tmp_path / "failing",
        provider_live_probe=failing_live,
    )
    with TestClient(failing_app) as client:
        assert client.patch(
            "/api/v1/provider-profiles/openrouter-default",
            json={"apiKey": "private-live-key"},
        ).status_code == 200
        failed = client.post(
            "/api/v1/provider-profiles/openrouter-default/test?live=true"
        )
    assert failed.json()["status"] == "unhealthy"
    assert sentinel not in failed.text
    assert sentinel not in caplog.text
    assert "provider.json" not in caplog.text


def test_delete_restores_snapshot_when_database_transaction_fails(
    tmp_path: Path,
) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        completed = _wait_for_terminal(client, job_id)
        asset_url = completed["result"]["clips"][0]["assets"]["videoUrl"]
        snapshot = tmp_path / "_web_jobs" / job_id
        tombstone = tmp_path / "_web_jobs" / f".tombstone-{job_id}"

        service = app.state.jobs
        with service._lock, service._database:
            service._database.execute(
                "CREATE TRIGGER reject_web_job_delete BEFORE DELETE ON jobs "
                "BEGIN SELECT RAISE(ABORT, 'test rollback'); END"
            )

        failed = client.delete(f"/api/v1/jobs/{job_id}")
        assert failed.status_code == 409
        assert snapshot.is_dir()
        assert not tombstone.exists()
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 200
        assert client.get(asset_url).content == b"video-data"

        with service._lock, service._database:
            service._database.execute("DROP TRIGGER reject_web_job_delete")
        assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 404


def test_delete_leaves_state_intact_when_snapshot_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yt_clipper.webapi.jobs as jobs_module

    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        completed = _wait_for_terminal(client, job_id)
        asset_url = completed["result"]["clips"][0]["assets"]["videoUrl"]
        snapshot = (tmp_path / "_web_jobs" / job_id).resolve()
        tombstone = (tmp_path / "_web_jobs" / f".tombstone-{job_id}").resolve()
        original_replace = jobs_module.os.replace

        def reject_snapshot_rename(source: Any, destination: Any) -> None:
            if Path(source).resolve() == snapshot and Path(destination).resolve() == tombstone:
                raise PermissionError("held by test")
            original_replace(source, destination)

        monkeypatch.setattr(jobs_module.os, "replace", reject_snapshot_rename)
        failed = client.delete(f"/api/v1/jobs/{job_id}")
        assert failed.status_code == 409
        assert snapshot.is_dir()
        assert not tombstone.exists()
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        assert client.get(asset_url).content == b"video-data"


def test_startup_recovers_or_discards_delete_tombstones(tmp_path: Path) -> None:
    initial = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(initial) as client:
        _configure_and_activate(client, "openrouter")
        project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        completed = _wait_for_terminal(client, job_id)
        asset_url = completed["result"]["clips"][0]["assets"]["videoUrl"]

    snapshot = tmp_path / "_web_jobs" / job_id
    tombstone = tmp_path / "_web_jobs" / f".tombstone-{job_id}"
    snapshot.rename(tombstone)
    orphan_tombstone = tmp_path / "_web_jobs" / ".tombstone-job_orphan"
    orphan_tombstone.mkdir()
    (orphan_tombstone / "orphan.mp4").write_bytes(b"orphan")

    restored = create_app(output_root=tmp_path)
    with TestClient(restored) as client:
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        assert client.get(asset_url).content == b"video-data"
    assert snapshot.is_dir()
    assert not tombstone.exists()
    assert not orphan_tombstone.exists()


def test_retention_is_serialized_and_idempotent(tmp_path: Path) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        job_ids: list[str] = []
        project_ids: list[str] = []
        for video_id in ("abcdefghijk", "lmnopqrstuv", "wxyzABCDEFG"):
            project_id = client.post(
                "/api/v1/projects",
                json={
                    "source": {
                        "kind": "youtube",
                        "url": f"https://youtu.be/{video_id}",
                    }
                },
            ).json()["id"]
            job_id = client.post(
                f"/api/v1/projects/{project_id}/generations",
                json={"providerProfileId": "openrouter-default", "options": {}},
            ).json()["id"]
            assert _wait_for_terminal(client, job_id)["status"] == "completed"
            project_ids.append(project_id)
            job_ids.append(job_id)

        service = app.state.jobs
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    service.apply_retention,
                    max_age_days=None,
                    max_jobs=1,
                )
                for _ in range(2)
            ]
            results = [future.result(timeout=3) for future in futures]

        deleted = [
            job_id
            for result in results
            for job_id in result.deleted_job_ids
        ]
        assert sorted(deleted) == sorted(job_ids[:2])
        assert len(deleted) == len(set(deleted))
        assert service.apply_retention(
            max_age_days=None,
            max_jobs=1,
        ).deleted_job_ids == []
        assert client.get(f"/api/v1/projects/{project_ids[0]}").status_code == 404
        assert client.get(f"/api/v1/projects/{project_ids[1]}").status_code == 404
        assert client.get(f"/api/v1/projects/{project_ids[2]}").status_code == 200


def test_api_deletes_only_empty_project_drafts(tmp_path: Path) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_successful_pipeline_factory({}, tmp_path),
    )
    with TestClient(app) as client:
        _configure_and_activate(client, "openrouter")
        empty_project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/abcdefghijk",
                }
            },
        ).json()["id"]
        occupied_project_id = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://youtu.be/lmnopqrstuv",
                }
            },
        ).json()["id"]
        job_id = client.post(
            f"/api/v1/projects/{occupied_project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        ).json()["id"]
        assert _wait_for_terminal(client, job_id)["status"] == "completed"

        assert client.delete(f"/api/v1/projects/{empty_project_id}").status_code == 204
        assert client.get(f"/api/v1/projects/{empty_project_id}").status_code == 404
        assert client.delete(f"/api/v1/projects/{occupied_project_id}").status_code == 409
        assert client.get(f"/api/v1/projects/{occupied_project_id}").status_code == 200
