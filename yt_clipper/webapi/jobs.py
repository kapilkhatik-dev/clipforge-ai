"""Durable local project/job service around the synchronous pipeline."""

from __future__ import annotations

import json
import base64
import binascii
import logging
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

from ..application.pipeline import ClipPipeline
from ..domain.models import PipelineConfig, PipelineEvent, PipelineResult, VideoLayout
from .models import (
    AssetLinks,
    ClipAsset,
    ClipEditDecisionList,
    EditorLink,
    GenerationOptions,
    JobError,
    JobEventEnvelope,
    JobResponse,
    JobResult,
    JobStatus,
    ProjectResponse,
    ProjectListResponse,
    ProjectSummary,
    GenerationSummary,
    ProjectSource,
    PublicPipelineEvent,
    RetentionResult,
    SourceSummary,
    SourceRange,
    public_job_error,
)
from .providers import ProviderRegistry


_ALLOWED_ASSET_SUFFIXES = {".mp4", ".webm", ".jpg", ".jpeg", ".png", ".webp"}
_STATE_SCHEMA_VERSION = 1
_MAX_JOB_EVENTS = 512
_PROGRESS_PERCENT_BUCKETS = 100
_YOUTUBE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CAPTION_PRESET = "bold-yellow-charcoal-outline-v2"
_DELETION_TOMBSTONE_PREFIX = ".tombstone-"

LOGGER = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _source_lock_key(source_url: str) -> str:
    """Return one local scheduling key for common URLs of the same YouTube video."""
    try:
        parsed = urlsplit(source_url)
        host = (parsed.hostname or "").casefold()
        candidate: str | None = None
        if host in {"youtu.be", "www.youtu.be"}:
            candidate = parsed.path.strip("/").split("/", 1)[0]
        elif host in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }:
            path_parts = [part for part in parsed.path.split("/") if part]
            if parsed.path.rstrip("/") == "/watch":
                candidate = (parse_qs(parsed.query).get("v") or [None])[0]
            elif len(path_parts) >= 2 and path_parts[0] in {
                "embed",
                "live",
                "shorts",
            }:
                candidate = path_parts[1]
        if candidate and _YOUTUBE_VIDEO_ID.fullmatch(candidate):
            return f"youtube:{candidate}"
    except (TypeError, ValueError):
        pass
    return f"source:{source_url.strip()}"


def _encode_project_cursor(created_at: datetime, project_id: str) -> str:
    payload = json.dumps(
        [created_at.isoformat(), project_id], separators=(",", ":")
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_project_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        created_at_value, project_id = json.loads(raw.decode("utf-8"))
        created_at = datetime.fromisoformat(created_at_value)
        if created_at.tzinfo is None or not isinstance(project_id, str) or not project_id:
            raise ValueError
        return created_at, project_id
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError, binascii.Error) as exc:
        raise ValueError("Invalid recent-projects cursor") from exc


class PipelineLike(Protocol):
    def run(self, url: str) -> PipelineResult: ...


class PipelineFactory(Protocol):
    def __call__(
        self, config: PipelineConfig, callback: object
    ) -> PipelineLike: ...


def _pipeline_factory(config: PipelineConfig, callback: object) -> ClipPipeline:
    return ClipPipeline(config, progress_callback=callback)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    source_url: str
    created_at: datetime


@dataclass
class JobRecord:
    response: JobResponse
    source_url: str
    config: PipelineConfig | None
    events: list[JobEventEnvelope] = field(default_factory=list)


@dataclass(frozen=True)
class AssetRecord:
    path: Path
    media_type: str


class AssetRegistry:
    """Maps random public IDs to generated files within one configured root."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._lock = RLock()
        self._assets: dict[str, AssetRecord] = {}

    def register(self, path: Path | None, *, asset_id: str | None = None) -> str | None:
        if path is None:
            return None
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("Generated asset lies outside the output directory") from exc
        if resolved.suffix.casefold() not in _ALLOWED_ASSET_SUFFIXES:
            raise ValueError("Generated asset type is not allowed")
        if not resolved.is_file():
            return None
        public_id = asset_id or secrets.token_urlsafe(24)
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        with self._lock:
            self._assets[public_id] = AssetRecord(resolved, media_type)
        return public_id

    def resolve(self, asset_id: str) -> AssetRecord:
        with self._lock:
            try:
                record = self._assets[asset_id]
            except KeyError as exc:
                raise KeyError(asset_id) from exc
        if not record.path.is_file():
            raise KeyError(asset_id)
        return record

    def remove(self, asset_id: str) -> None:
        with self._lock:
            self._assets.pop(asset_id, None)

    def remove_under(self, directory: Path) -> None:
        """Forget unpublished assets created during a failed snapshot transaction."""
        resolved_directory = directory.expanduser().resolve()
        with self._lock:
            stale = []
            for asset_id, record in self._assets.items():
                try:
                    record.path.relative_to(resolved_directory)
                except ValueError:
                    continue
                stale.append(asset_id)
            for asset_id in stale:
                self._assets.pop(asset_id, None)

    def rebind_under(self, source: Path, destination: Path) -> None:
        """Update registered paths after an atomic directory publication."""
        resolved_source = source.expanduser().resolve()
        resolved_destination = destination.expanduser().resolve()
        with self._lock:
            for asset_id, record in list(self._assets.items()):
                try:
                    relative = record.path.relative_to(resolved_source)
                except ValueError:
                    continue
                self._assets[asset_id] = AssetRecord(
                    path=resolved_destination / relative,
                    media_type=record.media_type,
                )


class JobService:
    """Bounded local executor; storage and execution can be replaced independently."""

    def __init__(
        self,
        providers: ProviderRegistry,
        output_root: Path,
        *,
        pipeline_factory: PipelineFactory = _pipeline_factory,
        max_workers: int | None = None,
    ) -> None:
        configured_workers = max_workers or int(os.getenv("CLIPPER_WEB_MAX_WORKERS", "1"))
        self._providers = providers
        self._output_root = output_root.expanduser().resolve()
        self._assets = AssetRegistry(self._output_root)
        self._state_dir = self._output_root / "_web_state"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._database_path = self._state_dir / "clipforge.sqlite3"
        self._database = sqlite3.connect(
            self._database_path,
            check_same_thread=False,
        )
        self._database.row_factory = sqlite3.Row
        self._pipeline_factory = pipeline_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(configured_workers, 4)),
            thread_name_prefix="clipforge-job",
        )
        self._lock = RLock()
        # Serializes manual deletion and retention selection/deletion.  Pipeline
        # workers still use ``_lock`` and are never held behind filesystem cleanup.
        self._maintenance_lock = RLock()
        self._projects: dict[str, ProjectRecord] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._source_locks: dict[str, RLock] = {}
        self._closed = False
        self._database_closed = False
        self._initialize_database()
        self._restore_state()

    @property
    def assets(self) -> AssetRegistry:
        return self._assets

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for job_id, record in self._jobs.items():
                if record.response.status != JobStatus.QUEUED:
                    continue
                record.response.status = JobStatus.FAILED
                record.response.finished_at = _now()
                record.response.message = "Generation canceled during service shutdown"
                record.response.error = JobError(
                    code="service_shutdown",
                    message="Generation was canceled because the local service stopped.",
                )
                record.config = None
                self._append_event(job_id, "failed", include_job=True)
                self._persist_job(job_id)
        self._close_database_if_idle()

    def _close_database_if_idle(self) -> None:
        with self._lock:
            if self._database_closed or not self._closed:
                return
            if any(
                record.response.status == JobStatus.RUNNING
                for record in self._jobs.values()
            ):
                return
            self._database.close()
            self._database_closed = True

    def _initialize_database(self) -> None:
        with self._database:
            self._database.execute("PRAGMA journal_mode=WAL")
            self._database.execute("PRAGMA foreign_keys=ON")
            self._database.execute(
                "CREATE TABLE IF NOT EXISTS schema_info "
                "(version INTEGER NOT NULL)"
            )
            row = self._database.execute(
                "SELECT version FROM schema_info LIMIT 1"
            ).fetchone()
            if row is None:
                self._database.execute(
                    "INSERT INTO schema_info(version) VALUES (?)",
                    (_STATE_SCHEMA_VERSION,),
                )
            elif int(row["version"]) != _STATE_SCHEMA_VERSION:
                raise RuntimeError("Unsupported ClipForge local state schema")
            self._database.execute(
                "CREATE TABLE IF NOT EXISTS projects ("
                "id TEXT PRIMARY KEY, source_url TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._database.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "id TEXT PRIMARY KEY, project_id TEXT NOT NULL, source_url TEXT NOT NULL, "
                "response_json TEXT NOT NULL, created_at TEXT NOT NULL, "
                "FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE)"
            )
            self._database.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "job_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_json TEXT NOT NULL, "
                "PRIMARY KEY(job_id, sequence), "
                "FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE)"
            )
            self._database.execute(
                "CREATE TABLE IF NOT EXISTS assets ("
                "id TEXT PRIMARY KEY, job_id TEXT NOT NULL, relative_path TEXT NOT NULL, "
                "media_type TEXT NOT NULL, "
                "FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE)"
            )

    @staticmethod
    def _remove_directory_best_effort(directory: Path) -> None:
        """Remove one internal snapshot directory without following symlinks."""
        try:
            if directory.is_symlink():
                directory.unlink(missing_ok=True)
            elif directory.is_dir():
                shutil.rmtree(directory)
        except OSError:
            # A committed deletion has already made the directory unreachable.
            # Startup retries cleanup, so an antivirus/file-handle race is harmless.
            pass

    def _reconcile_deletion_tombstones(self) -> None:
        """Finish or roll back snapshot deletion interrupted by process shutdown.

        A tombstone whose job row still exists represents a deletion that did not
        commit, so its final snapshot is restored before assets are registered.  A
        tombstone without a job row is unreachable garbage from a committed delete.
        """
        snapshots_root = self._output_root / "_web_jobs"
        if not snapshots_root.is_dir():
            return
        for tombstone in snapshots_root.iterdir():
            if not tombstone.name.startswith(_DELETION_TOMBSTONE_PREFIX):
                continue
            job_id = tombstone.name.removeprefix(_DELETION_TOMBSTONE_PREFIX)
            snapshot_dir = snapshots_root / job_id
            if job_id not in self._jobs or snapshot_dir.exists():
                self._remove_directory_best_effort(tombstone)
                continue
            try:
                os.replace(tombstone, snapshot_dir)
            except OSError as exc:
                # Refuse to serve state whose persisted result points at files that
                # cannot be recovered. The tombstone remains available for retry.
                raise RuntimeError(
                    "Could not recover an interrupted generated-file deletion"
                ) from exc

    def _restore_state(self) -> None:
        with self._lock:
            interrupted_job_ids: set[str] = set()
            for row in self._database.execute(
                "SELECT id, source_url, created_at FROM projects"
            ):
                project = ProjectRecord(
                    id=row["id"],
                    source_url=row["source_url"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                self._projects[project.id] = project
                self._source_locks.setdefault(_source_lock_key(project.source_url), RLock())

            for row in self._database.execute(
                "SELECT id, source_url, response_json FROM jobs ORDER BY created_at"
            ):
                response = JobResponse.model_validate_json(row["response_json"])
                if response.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    response.status = JobStatus.FAILED
                    response.finished_at = _now()
                    response.message = "Generation was interrupted by a service restart"
                    response.error = JobError(
                        code="service_restart",
                        message="Generation stopped because the local service restarted.",
                    )
                    interrupted_job_ids.add(row["id"])
                self._jobs[row["id"]] = JobRecord(
                    response=response,
                    source_url=row["source_url"],
                    config=None,
                )

            # Resolve interrupted delete transactions before persisted asset paths
            # are validated and registered.
            self._reconcile_deletion_tombstones()

            for row in self._database.execute(
                "SELECT job_id, event_json FROM events ORDER BY job_id, sequence"
            ):
                record = self._jobs.get(row["job_id"])
                if record is not None:
                    record.events.append(
                        JobEventEnvelope.model_validate_json(row["event_json"])
                    )

            stale_asset_ids: list[str] = []
            for row in self._database.execute(
                "SELECT id, relative_path FROM assets"
            ):
                candidate = (self._output_root / row["relative_path"]).resolve()
                try:
                    candidate.relative_to(self._output_root)
                except ValueError:
                    stale_asset_ids.append(row["id"])
                    continue
                if self._assets.register(candidate, asset_id=row["id"]) is None:
                    stale_asset_ids.append(row["id"])
            if stale_asset_ids:
                with self._database:
                    self._database.executemany(
                        "DELETE FROM assets WHERE id = ?",
                        ((asset_id,) for asset_id in stale_asset_ids),
                    )

            for job_id in interrupted_job_ids:
                self._discard_snapshot(
                    job_id,
                    self._output_root / "_web_jobs" / job_id,
                )

            for job_id, record in self._jobs.items():
                if record.response.error and record.response.error.code == "service_restart":
                    self._append_event(job_id, "failed", include_job=True)
                    self._persist_job(job_id)

            referenced_job_ids = set(self._jobs)
            snapshots_root = self._output_root / "_web_jobs"
            if snapshots_root.is_dir():
                for candidate in snapshots_root.iterdir():
                    if (
                        candidate.is_dir()
                        and (
                            candidate.name.startswith(".job_")
                            or (
                                candidate.name.startswith("job_")
                                and candidate.name not in referenced_job_ids
                            )
                        )
                    ):
                        shutil.rmtree(candidate, ignore_errors=True)

    def _persist_project(self, project: ProjectRecord) -> None:
        with self._lock:
            with self._database:
                self._database.execute(
                    "INSERT INTO projects(id, source_url, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "source_url=excluded.source_url, created_at=excluded.created_at",
                    (project.id, project.source_url, project.created_at.isoformat()),
                )

    def _persist_job(self, job_id: str) -> None:
        record = self._jobs[job_id]
        serialized = record.response.model_dump_json(by_alias=True)
        with self._lock:
            with self._database:
                self._database.execute(
                    "INSERT INTO jobs"
                    "(id, project_id, source_url, response_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "project_id=excluded.project_id, source_url=excluded.source_url, "
                    "response_json=excluded.response_json, created_at=excluded.created_at",
                    (
                        job_id,
                        record.response.project_id,
                        record.source_url,
                        serialized,
                        record.response.created_at.isoformat(),
                    ),
                )

    def _persist_event(self, event: JobEventEnvelope) -> None:
        with self._lock:
            with self._database:
                self._database.execute(
                    "INSERT OR REPLACE INTO events(job_id, sequence, event_json) VALUES (?, ?, ?)",
                    (
                        event.job_id,
                        event.sequence,
                        event.model_dump_json(by_alias=True),
                    ),
                )

    def create_project(self, source_url: str) -> ProjectRecord:
        project = ProjectRecord(
            id=f"prj_{secrets.token_urlsafe(12)}",
            source_url=source_url,
            created_at=_now(),
        )
        with self._lock:
            if self._closed:
                raise ValueError("The local service is shutting down")
            self._projects[project.id] = project
            self._persist_project(project)
        return project

    def project(self, project_id: str) -> ProjectResponse:
        with self._lock:
            try:
                project = self._projects[project_id]
            except KeyError as exc:
                raise KeyError(project_id) from exc
            jobs = sorted(
                (
                    record.response.model_copy(deep=True)
                    for record in self._jobs.values()
                    if record.response.project_id == project_id
                ),
                key=lambda item: item.created_at,
                reverse=True,
            )
            return ProjectResponse(
                id=project.id,
                created_at=project.created_at,
                source=ProjectSource(url=project.source_url),
                jobs=jobs,
            )

    @staticmethod
    def _generation_summary(response: JobResponse) -> GenerationSummary:
        result = response.result
        title = None
        thumbnail_url = None
        export_count = 0
        if result is not None:
            export_count = len(result.clips) + (1 if result.montage is not None else 0)
            title = result.source.title if result.source is not None else None
            thumbnail_url = (
                result.source.thumbnail_url if result.source is not None else None
            )
            preview = result.montage or (result.clips[0] if result.clips else None)
            if preview is not None:
                title = title or preview.title
                thumbnail_url = (
                    thumbnail_url
                    or preview.assets.poster_url
                    or preview.assets.thumbnail_url
                )
        return GenerationSummary(
            id=response.id,
            status=response.status,
            created_at=response.created_at,
            started_at=response.started_at,
            finished_at=response.finished_at,
            title=title,
            thumbnail_url=thumbnail_url,
            export_count=export_count,
        )

    def projects(
        self,
        *,
        limit: int = 12,
        cursor: str | None = None,
    ) -> ProjectListResponse:
        """Return non-empty projects ordered by their latest generation activity."""
        if not 1 <= limit <= 50:
            raise ValueError("Project page limit must be between 1 and 50")
        cursor_key = _decode_project_cursor(cursor) if cursor else None
        with self._lock:
            jobs_by_project: dict[str, list[JobResponse]] = {}
            for record in self._jobs.values():
                jobs_by_project.setdefault(record.response.project_id, []).append(
                    record.response
                )

            summaries: list[tuple[datetime, str, ProjectSummary]] = []
            for project_id, project_jobs in jobs_by_project.items():
                project = self._projects.get(project_id)
                if project is None or not project_jobs:
                    continue
                latest = max(project_jobs, key=lambda item: (item.created_at, item.id))
                key = (latest.created_at, project.id)
                if cursor_key is not None and key >= cursor_key:
                    continue
                summaries.append(
                    (
                        *key,
                        ProjectSummary(
                            id=project.id,
                            created_at=project.created_at,
                            source=ProjectSource(url=project.source_url),
                            generation_count=len(project_jobs),
                            latest_generation=self._generation_summary(latest),
                        ),
                    )
                )

            summaries.sort(key=lambda item: (item[0], item[1]), reverse=True)
            page = summaries[: limit + 1]
            has_more = len(page) > limit
            page = page[:limit]
            next_cursor = (
                _encode_project_cursor(page[-1][0], page[-1][1])
                if has_more and page
                else None
            )
            return ProjectListResponse(
                items=[item[2].model_copy(deep=True) for item in page],
                next_cursor=next_cursor,
            )

    def create_generation(
        self,
        project_id: str,
        profile_id: str,
        options: GenerationOptions,
    ) -> JobResponse:
        with self._lock:
            try:
                project = self._projects[project_id]
            except KeyError as exc:
                raise KeyError(project_id) from exc

        values = options.model_dump()
        if values["min_clip_duration"] > values["max_clip_duration"]:
            raise ValueError("minClipDuration cannot exceed maxClipDuration")
        # Each job owns an immutable provider/model snapshot and output root.
        config = self._providers.generation_config(
            profile_id,
            output_dir=self._output_root,
            **values,
        )
        job_id = f"job_{secrets.token_urlsafe(12)}"
        response = JobResponse(
            id=job_id,
            project_id=project_id,
            status=JobStatus.QUEUED,
            stage="setup",
            stage_progress=0,
            message="Queued for local processing",
            created_at=_now(),
        )
        with self._lock:
            if self._closed:
                raise ValueError("The local service is shutting down")
            # A terminal job deletion may prune its now-empty project while the
            # provider snapshot above is being prepared.
            if project_id not in self._projects:
                raise KeyError(project_id)
            self._jobs[job_id] = JobRecord(response, project.source_url, config)
            self._source_locks.setdefault(_source_lock_key(project.source_url), RLock())
            self._persist_job(job_id)
            try:
                self._executor.submit(self._run, job_id)
            except RuntimeError as exc:
                self._jobs.pop(job_id, None)
                # Submission can fail after the durable queued row is written
                # (for example, if the executor is shutting down). Roll it back
                # so a later service restart cannot resurrect a job that never ran.
                with self._database:
                    self._database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                raise ValueError("The local service is shutting down") from exc
            return response.model_copy(deep=True)

    def get(self, job_id: str) -> JobResponse:
        with self._lock:
            try:
                return self._jobs[job_id].response.model_copy(deep=True)
            except KeyError as exc:
                raise KeyError(job_id) from exc

    def events_after(self, job_id: str, sequence: int) -> list[JobEventEnvelope]:
        with self._lock:
            try:
                record = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc
            return [
                item.model_copy(deep=True)
                for item in record.events
                if item.sequence > sequence
            ]

    def _append_event(
        self,
        job_id: str,
        event_type: str,
        event: PublicPipelineEvent | None = None,
        *,
        include_job: bool = False,
    ) -> None:
        record = self._jobs[job_id]
        previous = record.events[-1] if record.events else None
        progress_bucket = (
            round((event.progress or 0) * _PROGRESS_PERCENT_BUCKETS)
            if event is not None and event.progress is not None
            else None
        )
        previous_bucket = (
            round((previous.event.progress or 0) * _PROGRESS_PERCENT_BUCKETS)
            if previous is not None
            and previous.type == "progress"
            and previous.event is not None
            and previous.event.progress is not None
            else None
        )
        if (
            event_type == "progress"
            and previous is not None
            and previous.type == "progress"
            and previous.event is not None
            and event is not None
            and previous.event.stage == event.stage
            and previous.event.message == event.message
            and previous_bucket == progress_bucket
        ):
            return
        envelope = JobEventEnvelope(
                job_id=job_id,
                sequence=(record.events[-1].sequence + 1) if record.events else 1,
                timestamp=_now(),
                type=event_type,  # type: ignore[arg-type]
                event=event,
                job=record.response.model_copy(deep=True) if include_job else None,
            )
        record.events.append(envelope)
        if len(record.events) > _MAX_JOB_EVENTS:
            record.events = record.events[-_MAX_JOB_EVENTS:]
        self._persist_event(envelope)
        if record.events:
            oldest_sequence = record.events[0].sequence
            with self._database:
                self._database.execute(
                    "DELETE FROM events WHERE job_id = ? AND sequence < ?",
                    (job_id, oldest_sequence),
                )

    def _progress(self, job_id: str, pipeline_event: PipelineEvent) -> None:
        public = PublicPipelineEvent(
            stage=pipeline_event.stage.value,
            message=pipeline_event.message,
            progress=pipeline_event.progress,
            current=pipeline_event.current,
            total=pipeline_event.total,
        )
        with self._lock:
            record = self._jobs[job_id]
            record.response.stage = pipeline_event.stage.value
            record.response.stage_progress = pipeline_event.progress
            record.response.message = pipeline_event.message
            self._append_event(job_id, "progress", public)
            self._persist_job(job_id)

    @staticmethod
    def _asset_url(asset_id: str | None, *, download: bool = False) -> str | None:
        if asset_id is None:
            return None
        suffix = "?download=1" if download else ""
        return f"/api/v1/assets/{asset_id}{suffix}"

    def _links(
        self,
        job_id: str,
        snapshot_dir: Path,
        stem: str,
        video: Path,
        thumbnail: Path | None,
        poster: Path | None,
    ) -> AssetLinks:
        video_snapshot = self._snapshot_asset(
            video,
            snapshot_dir,
            f"{stem}-video",
            required=True,
        )
        thumbnail_snapshot = self._snapshot_asset(
            thumbnail,
            snapshot_dir,
            f"{stem}-thumbnail",
        )
        poster_snapshot = self._snapshot_asset(
            poster,
            snapshot_dir,
            f"{stem}-poster",
        )
        video_id = self._assets.register(video_snapshot)
        if video_id is None:
            raise ValueError("The rendered video asset is missing")
        thumbnail_id = self._assets.register(thumbnail_snapshot)
        poster_id = self._assets.register(poster_snapshot)
        for asset_id, snapshot in (
            (video_id, video_snapshot),
            (thumbnail_id, thumbnail_snapshot),
            (poster_id, poster_snapshot),
        ):
            if asset_id is not None and snapshot is not None:
                self._persist_asset(asset_id, job_id, snapshot)
        return AssetLinks(
            video_url=self._asset_url(video_id) or "",
            download_url=self._asset_url(video_id, download=True),
            thumbnail_url=self._asset_url(thumbnail_id),
            poster_url=self._asset_url(poster_id),
        )

    def _persist_asset(self, asset_id: str, job_id: str, path: Path) -> None:
        # Snapshot builds use a hidden staging directory. Persist the path at its
        # final job namespace so restart recovery never depends on the staging name.
        persisted_path = path
        if path.parent.name.startswith(f".{job_id}-") and path.parent.parent.name == "_web_jobs":
            persisted_path = path.parent.parent / job_id / path.name
        relative = persisted_path.resolve().relative_to(self._output_root).as_posix()
        media_type = mimetypes.guess_type(persisted_path.name)[0] or "application/octet-stream"
        with self._lock:
            with self._database:
                self._database.execute(
                    "INSERT OR REPLACE INTO assets(id, job_id, relative_path, media_type) "
                    "VALUES (?, ?, ?, ?)",
                    (asset_id, job_id, relative, media_type),
                )

    def _snapshot_asset(
        self,
        source: Path | None,
        snapshot_dir: Path,
        stem: str,
        *,
        required: bool = False,
    ) -> Path | None:
        """Atomically copy one approved artifact into an immutable job namespace."""
        if source is None:
            if required:
                raise ValueError("A required rendered asset is missing")
            return None
        resolved = source.expanduser().resolve()
        try:
            resolved.relative_to(self._output_root)
        except ValueError as exc:
            raise ValueError("Generated asset lies outside the output directory") from exc
        suffix = resolved.suffix.casefold()
        if suffix not in _ALLOWED_ASSET_SUFFIXES:
            raise ValueError("Generated asset type is not allowed")
        if not resolved.is_file():
            if required:
                raise ValueError("A required rendered asset is missing")
            return None

        destination = snapshot_dir / f"{stem}{suffix}"
        temporary = snapshot_dir / f".{stem}-{secrets.token_urlsafe(8)}.tmp"
        try:
            shutil.copyfile(resolved, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _discard_snapshot(self, job_id: str, snapshot_dir: Path) -> None:
        """Remove every unpublished file and opaque mapping from a failed result."""
        self._assets.remove_under(snapshot_dir)
        with self._lock:
            asset_ids = [
                row["id"]
                for row in self._database.execute(
                    "SELECT id FROM assets WHERE job_id = ?", (job_id,)
                )
            ]
            with self._database:
                self._database.execute("DELETE FROM assets WHERE job_id = ?", (job_id,))
        for asset_id in asset_ids:
            self._assets.remove(asset_id)
        try:
            if snapshot_dir.is_dir():
                shutil.rmtree(snapshot_dir)
        except OSError:
            # The job is still failed and none of these files has a public mapping.
            pass

    def _result(
        self,
        job_id: str,
        project_id: str,
        result: PipelineResult,
        video_layout: VideoLayout,
    ) -> JobResult:
        web_jobs_dir = self._output_root / "_web_jobs"
        snapshot_dir = web_jobs_dir / job_id
        staging_dir = web_jobs_dir / f".{job_id}-{secrets.token_urlsafe(8)}"
        try:
            staging_dir.mkdir(parents=True, exist_ok=False)
            public_result = self._build_result(
                job_id,
                project_id,
                result,
                staging_dir,
                video_layout,
            )
            os.replace(staging_dir, snapshot_dir)
            self._assets.rebind_under(staging_dir, snapshot_dir)
            return public_result
        except Exception:
            self._discard_snapshot(job_id, staging_dir)
            self._discard_snapshot(job_id, snapshot_dir)
            raise

    def _build_result(
        self,
        job_id: str,
        project_id: str,
        result: PipelineResult,
        snapshot_dir: Path,
        video_layout: VideoLayout,
    ) -> JobResult:
        clips: list[ClipAsset] = []
        for index, (candidate, video_path) in enumerate(
            zip(result.candidates, result.clip_paths, strict=False), start=1
        ):
            thumbnail = (
                result.thumbnail_paths[index - 1]
                if index <= len(result.thumbnail_paths)
                else None
            )
            poster = (
                result.poster_paths[index - 1]
                if index <= len(result.poster_paths)
                else None
            )
            clip_id = f"{job_id}-clip-{index}"
            clips.append(
                ClipAsset(
                    id=clip_id,
                    title=candidate.title,
                    start=candidate.start,
                    end=candidate.end,
                    duration=candidate.duration,
                    score=candidate.score,
                    hook=candidate.hook,
                    reason=candidate.reason,
                    assets=self._links(
                        job_id,
                        snapshot_dir,
                        f"clip-{index:03d}",
                        video_path,
                        thumbnail,
                        poster,
                    ),
                    editor=EditorLink(
                        route=f"/projects/{project_id}/clips/{clip_id}/edit",
                        available=False,
                    ),
                    edit_decision_list=ClipEditDecisionList(
                        kind="continuous",
                        source_video_id=result.video.metadata.video_id,
                        source_ranges=[
                            SourceRange(start=candidate.start, end=candidate.end)
                        ],
                        video_layout=video_layout,
                        caption_preset=_CAPTION_PRESET,
                    ),
                )
            )

        montage_asset = None
        if result.montage is not None and result.montage_video_path is not None:
            montage_id = f"{job_id}-montage"
            montage_asset = ClipAsset(
                id=montage_id,
                title=result.montage.title,
                start=0,
                end=result.montage.duration,
                duration=result.montage.duration,
                assets=self._links(
                    job_id,
                    snapshot_dir,
                    "montage",
                    result.montage_video_path,
                    result.montage_thumbnail_path,
                    result.montage_poster_path,
                ),
                editor=EditorLink(
                    route=(
                        f"/projects/{project_id}/clips/"
                        f"{montage_id}/edit"
                    ),
                    available=False,
                ),
                edit_decision_list=ClipEditDecisionList(
                    kind="montage",
                    source_video_id=result.video.metadata.video_id,
                    source_ranges=[
                        SourceRange(start=moment.start, end=moment.end)
                        for moment in result.montage.moments
                    ],
                    video_layout=video_layout,
                    caption_preset=_CAPTION_PRESET,
                ),
            )

        source_thumbnail_path = self._snapshot_asset(
            result.video.thumbnail_path,
            snapshot_dir,
            "source-thumbnail",
        )
        source_thumbnail = self._assets.register(source_thumbnail_path)
        if source_thumbnail is not None and source_thumbnail_path is not None:
            self._persist_asset(source_thumbnail, job_id, source_thumbnail_path)
        return JobResult(
            source=SourceSummary(
                title=result.video.metadata.title,
                thumbnail_url=self._asset_url(source_thumbnail),
                duration=result.video.metadata.duration_seconds,
            ),
            montage=montage_asset,
            clips=clips,
        )

    def _run(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record.response.status != JobStatus.QUEUED:
                return
            record.response.status = JobStatus.RUNNING
            record.response.started_at = _now()
            record.response.message = "Starting local processing"
            source_lock = self._source_locks[_source_lock_key(record.source_url)]
            config = record.config
            # The worker owns this immutable secret-bearing snapshot from here on;
            # completed/failed JobRecords never retain provider credentials.
            record.config = None
        try:
            if config is None:
                raise RuntimeError("Generation configuration is unavailable")
            # Keep the pipeline's deterministic source workspace stable until every
            # public asset has been copied into this job's immutable namespace.
            with source_lock:
                pipeline = self._pipeline_factory(
                    config,
                    lambda event: self._progress(job_id, event),
                )
                pipeline_result = pipeline.run(record.source_url)
                public_result = self._result(
                    job_id,
                    record.response.project_id,
                    pipeline_result,
                    config.video_layout,
                )
        except Exception as exc:
            public_error = public_job_error(exc)
            with self._lock:
                record.response.status = JobStatus.FAILED
                record.response.finished_at = _now()
                record.response.error = public_error
                record.response.message = "Generation failed"
                self._append_event(job_id, "failed", include_job=True)
                self._persist_job(job_id)
            # Exception messages can contain signed media URLs, API credentials,
            # or local paths. Log only stable, display-safe diagnostics.
            LOGGER.warning(
                "Generation %s failed at stage %s (%s; %s)",
                job_id,
                record.response.stage,
                public_error.code,
                type(exc).__name__,
            )
            self._close_database_if_idle()
            return

        with self._lock:
            record.response.status = JobStatus.COMPLETED
            record.response.stage = "complete"
            record.response.stage_progress = 1
            record.response.message = "Your clips are ready"
            record.response.finished_at = _now()
            record.response.result = public_result
            self._append_event(job_id, "completed", include_job=True)
            self._persist_job(job_id)
        self._close_database_if_idle()

    @staticmethod
    def _snapshot_size(directory: Path) -> int:
        total = 0
        try:
            if directory.is_file():
                return directory.stat().st_size
            for path in directory.rglob("*"):
                try:
                    if path.is_file():
                        total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            pass
        return total

    def _delete_job_locked(
        self,
        job_id: str,
        *,
        missing_ok: bool,
    ) -> tuple[bool, int]:
        """Delete one terminal job while ``_maintenance_lock`` is held.

        The final snapshot is first renamed to a hidden tombstone. The database
        transaction then removes the job and (when empty) its project. This order
        lets a failed transaction restore the immutable files without ever
        publishing a persisted result whose assets were permanently removed.
        """
        tombstone: Path | None = None
        reclaimed_bytes = 0
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                if missing_ok:
                    return False, 0
                raise KeyError(job_id)
            if record.response.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                raise ValueError("A running generation cannot be deleted")

            snapshots_root = self._output_root / "_web_jobs"
            snapshot_dir = snapshots_root / job_id
            candidate_tombstone = snapshots_root / (
                f"{_DELETION_TOMBSTONE_PREFIX}{job_id}"
            )
            if candidate_tombstone.exists() and snapshot_dir.exists():
                self._remove_directory_best_effort(candidate_tombstone)
                if candidate_tombstone.exists():
                    raise ValueError("Generated files are already being cleaned up")

            if snapshot_dir.exists():
                snapshots_root.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(snapshot_dir, candidate_tombstone)
                except OSError as exc:
                    raise ValueError("Generated files could not be prepared for removal") from exc
                self._assets.rebind_under(snapshot_dir, candidate_tombstone)
                tombstone = candidate_tombstone
            elif candidate_tombstone.exists():
                # A prior rollback may have been unable to rename the directory
                # back while an external process held it. Its opaque links remain
                # valid through the registry and startup will recover it.
                self._assets.rebind_under(snapshot_dir, candidate_tombstone)
                tombstone = candidate_tombstone

            if tombstone is not None:
                reclaimed_bytes = self._snapshot_size(tombstone)

            asset_ids = [
                row["id"]
                for row in self._database.execute(
                    "SELECT id FROM assets WHERE job_id = ?", (job_id,)
                )
            ]
            project_id = record.response.project_id
            project_pruned = False
            try:
                with self._database:
                    self._database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                    cursor = self._database.execute(
                        "DELETE FROM projects WHERE id = ? "
                        "AND NOT EXISTS (SELECT 1 FROM jobs WHERE project_id = ?)",
                        (project_id, project_id),
                    )
                    project_pruned = cursor.rowcount > 0
            except Exception as exc:
                if tombstone is not None and tombstone.exists():
                    try:
                        os.replace(tombstone, snapshot_dir)
                    except OSError:
                        # Keep serving the opaque IDs from the tombstone. On the
                        # next startup it is restored before asset registration.
                        pass
                    else:
                        self._assets.rebind_under(tombstone, snapshot_dir)
                raise ValueError("Job state could not be removed") from exc

            for asset_id in asset_ids:
                self._assets.remove(asset_id)
            if tombstone is not None:
                self._assets.remove_under(tombstone)
            self._jobs.pop(job_id, None)
            if project_pruned:
                self._projects.pop(project_id, None)
                source_key = _source_lock_key(record.source_url)
                if not any(
                    _source_lock_key(project.source_url) == source_key
                    for project in self._projects.values()
                ):
                    self._source_locks.pop(source_key, None)

        if tombstone is not None:
            self._remove_directory_best_effort(tombstone)
        return True, reclaimed_bytes

    def delete_job(self, job_id: str) -> None:
        with self._maintenance_lock:
            self._delete_job_locked(job_id, missing_ok=False)

    def delete_empty_project(self, project_id: str) -> None:
        """Delete an abandoned project draft, but never cascade real jobs."""
        with self._maintenance_lock:
            with self._lock:
                try:
                    project = self._projects[project_id]
                except KeyError as exc:
                    raise KeyError(project_id) from exc
                if any(
                    record.response.project_id == project_id
                    for record in self._jobs.values()
                ):
                    raise ValueError("A project with generations cannot be deleted")
                try:
                    with self._database:
                        cursor = self._database.execute(
                            "DELETE FROM projects WHERE id = ? "
                            "AND NOT EXISTS "
                            "(SELECT 1 FROM jobs WHERE project_id = ?)",
                            (project_id, project_id),
                        )
                        if cursor.rowcount != 1:
                            raise ValueError(
                                "A project with generations cannot be deleted"
                            )
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError("Project state could not be removed") from exc
                self._projects.pop(project_id, None)
                source_key = _source_lock_key(project.source_url)
                if not any(
                    _source_lock_key(item.source_url) == source_key
                    for item in self._projects.values()
                ):
                    self._source_locks.pop(source_key, None)

    def apply_retention(
        self,
        *,
        max_age_days: int | None,
        max_jobs: int | None,
    ) -> RetentionResult:
        with self._maintenance_lock:
            with self._lock:
                terminal = sorted(
                    (
                        record.response.model_copy(deep=True)
                        for record in self._jobs.values()
                        if record.response.status in {
                            JobStatus.COMPLETED,
                            JobStatus.FAILED,
                        }
                    ),
                    key=lambda item: item.finished_at or item.created_at,
                    reverse=True,
                )
            now = _now()
            delete_ids: set[str] = set()
            if max_jobs is not None:
                delete_ids.update(item.id for item in terminal[max_jobs:])
            if max_age_days is not None:
                cutoff_seconds = max_age_days * 86_400
                delete_ids.update(
                    item.id
                    for item in terminal
                    if (now - (item.finished_at or item.created_at)).total_seconds()
                    > cutoff_seconds
                )
            deleted_ids: list[str] = []
            reclaimed = 0
            for job_id in sorted(delete_ids):
                deleted, job_bytes = self._delete_job_locked(
                    job_id,
                    missing_ok=True,
                )
                if deleted:
                    deleted_ids.append(job_id)
                    reclaimed += job_bytes
        return RetentionResult(
            deleted_job_ids=deleted_ids,
            reclaimed_bytes=reclaimed,
        )

    @staticmethod
    def encode_sse(event: JobEventEnvelope) -> bytes:
        payload = event.model_dump(mode="json", by_alias=True, exclude_none=True)
        return (
            f"id: {event.sequence}\n"
            f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        ).encode()
