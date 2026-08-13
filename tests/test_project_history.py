from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from yt_clipper.domain.errors import AnalysisError
from yt_clipper.domain.models import PipelineConfig, PipelineResult
from yt_clipper.webapi.app import create_app


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLIPPER_LLM_PROVIDER",
        "CLIPPER_LLM_MODEL",
        "CLIPPER_LLM_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class _FailingPipeline:
    def __init__(self, config: PipelineConfig, callback: object) -> None:
        del config, callback

    def run(self, url: str) -> PipelineResult:
        del url
        raise AnalysisError("expected test failure")


def _activate_openrouter(client: TestClient) -> None:
    assert client.patch(
        "/api/v1/provider-profiles/openrouter-default",
        json={"apiKey": "test-key"},
    ).status_code == 200
    assert client.post(
        "/api/v1/provider-profiles/openrouter-default/test"
    ).status_code == 200
    assert client.patch(
        "/api/v1/settings/default-provider",
        json={"providerProfileId": "openrouter-default"},
    ).status_code == 200


def _create_generation(client: TestClient, suffix: str) -> tuple[str, str]:
    project = client.post(
        "/api/v1/projects",
        json={
            "source": {
                "kind": "youtube",
                "url": f"https://www.youtube.com/watch?v={suffix:0<11}",
            }
        },
    )
    assert project.status_code == 201
    project_id = project.json()["id"]
    generation = client.post(
        f"/api/v1/projects/{project_id}/generations",
        json={"providerProfileId": "openrouter-default", "options": {}},
    )
    assert generation.status_code == 202
    return project_id, generation.json()["id"]


def _wait_for_terminal(client: TestClient, job_id: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "failed":
            return
        time.sleep(0.01)
    raise AssertionError("generation did not finish")


def test_recent_projects_excludes_empty_drafts_and_paginates_stably(
    tmp_path: Path,
) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_FailingPipeline,
        provider_tester=lambda _: (True, "Configuration checked"),
    )
    with TestClient(app) as client:
        _activate_openrouter(client)

        empty = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://www.youtube.com/watch?v=emptydraft1",
                }
            },
        )
        assert empty.status_code == 201

        discarded = client.delete(f"/api/v1/projects/{empty.json()['id']}")
        assert discarded.status_code == 204
        assert client.get(f"/api/v1/projects/{empty.json()['id']}").status_code == 404

        empty = client.post(
            "/api/v1/projects",
            json={
                "source": {
                    "kind": "youtube",
                    "url": "https://www.youtube.com/watch?v=emptydraft2",
                }
            },
        )
        assert empty.status_code == 201

        created: list[tuple[str, str]] = []
        for suffix in ("history001", "history002", "history003"):
            project_id, job_id = _create_generation(client, suffix)
            _wait_for_terminal(client, job_id)
            created.append((project_id, job_id))

        first = client.get("/api/v1/projects", params={"limit": 2})
        assert first.status_code == 200
        first_page = first.json()
        assert len(first_page["items"]) == 2
        assert first_page["nextCursor"]
        assert empty.json()["id"] not in {item["id"] for item in first_page["items"]}
        assert first_page["items"][0]["id"] == created[-1][0]
        assert "jobs" not in first_page["items"][0]
        assert first_page["items"][0]["generationCount"] == 1
        assert first_page["items"][0]["latestGeneration"] == {
            "id": created[-1][1],
            "status": "failed",
            "createdAt": first_page["items"][0]["latestGeneration"]["createdAt"],
            "startedAt": first_page["items"][0]["latestGeneration"]["startedAt"],
            "finishedAt": first_page["items"][0]["latestGeneration"]["finishedAt"],
            "title": None,
            "thumbnailUrl": None,
            "exportCount": 0,
        }

        second = client.get(
            "/api/v1/projects",
            params={"limit": 2, "cursor": first_page["nextCursor"]},
        )
        assert second.status_code == 200
        second_page = second.json()
        assert len(second_page["items"]) == 1
        assert second_page["nextCursor"] is None
        assert {
            item["id"] for item in first_page["items"]
        }.isdisjoint(item["id"] for item in second_page["items"])

        invalid = client.get("/api/v1/projects", params={"cursor": "not-a-cursor"})
        assert invalid.status_code == 422
        assert "cursor" in invalid.json()["detail"].lower()


def test_recent_project_counts_generations_and_uses_latest_activity(
    tmp_path: Path,
) -> None:
    app = create_app(
        output_root=tmp_path,
        pipeline_factory=_FailingPipeline,
        provider_tester=lambda _: (True, "Configuration checked"),
    )
    with TestClient(app) as client:
        _activate_openrouter(client)
        project_id, first_job_id = _create_generation(client, "versions01")
        _wait_for_terminal(client, first_job_id)
        second = client.post(
            f"/api/v1/projects/{project_id}/generations",
            json={"providerProfileId": "openrouter-default", "options": {}},
        )
        assert second.status_code == 202
        second_job_id = second.json()["id"]
        _wait_for_terminal(client, second_job_id)

        item = client.get("/api/v1/projects").json()["items"][0]
        assert item["id"] == project_id
        assert item["generationCount"] == 2
        assert item["latestGeneration"]["id"] == second_job_id
