"""FastAPI application factory for the local ClipForge UI."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .jobs import JobService, PipelineFactory
from .models import (
    BootstrapResponse,
    DefaultProviderPatch,
    DefaultProviderResponse,
    GenerationCreate,
    JobResponse,
    MediaToolDiagnostics,
    OutputDiagnostics,
    ProjectCreate,
    ProjectCreated,
    ProjectListResponse,
    ProjectResponse,
    ProviderDescriptor,
    ProviderPatch,
    ProviderProfile,
    ProviderTestResult,
    ProviderDiagnostic,
    RetentionRequest,
    RetentionResult,
    DiagnosticCheck,
    SystemDiagnostics,
)
from .providers import ProviderLiveProbe, ProviderRegistry, ProviderTester
from ..infrastructure.media_tools import resolve_media_tools


_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_WEB_CLIENT_HEADER = "X-ClipForge-Client"


def _origin_identity(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme, parsed.hostname.casefold(), port


def create_app(
    *,
    output_root: Path | str = "output",
    pipeline_factory: PipelineFactory | None = None,
    provider_tester: ProviderTester | None = None,
    provider_live_probe: ProviderLiveProbe | None = None,
    max_workers: int | None = None,
    web_dist: Path | str | None = None,
) -> FastAPI:
    providers = ProviderRegistry(provider_tester, provider_live_probe)
    jobs = JobService(
        providers,
        Path(output_root),
        **({"pipeline_factory": pipeline_factory} if pipeline_factory else {}),
        max_workers=max_workers,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.close()

    app = FastAPI(
        title="ClipForge AI local API",
        version="0.1.0",
        lifespan=lifespan,
    )
    configured_hosts = [
        item.strip()
        for item in os.getenv("CLIPPER_WEB_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    ]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver", *configured_hosts],
    )
    app.state.providers = providers
    app.state.jobs = jobs

    @app.middleware("http")
    async def protect_local_mutations(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Reject browser cross-site writes while preserving native local clients."""
        if request.method in _UNSAFE_METHODS:
            trusted_client = request.headers.get(_WEB_CLIENT_HEADER) == "web"
            origin = request.headers.get("origin")
            fetch_site = request.headers.get("sec-fetch-site", "").casefold()
            same_origin = bool(
                origin
                and _origin_identity(origin)
                == _origin_identity(str(request.base_url).rstrip("/"))
            )
            if not trusted_client and (
                (origin is not None and not same_origin) or fetch_site == "cross-site"
            ):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": {
                            "code": "cross_site_request_blocked",
                            "message": "Cross-site changes are not allowed by the local service.",
                        }
                    },
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        safe_errors = [
            {
                "loc": list(item.get("loc", ())),
                "type": item.get("type", "validation_error"),
                "msg": item.get("msg", "Invalid value"),
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "validation_error",
                    "message": "One or more submitted values are invalid.",
                    "errors": safe_errors,
                }
            },
        )

    @app.get("/api/v1/bootstrap", response_model=BootstrapResponse)
    def bootstrap() -> BootstrapResponse:
        return BootstrapResponse(
            default_provider_profile_id=providers.active_profile_id,
        )

    @app.get("/api/v1/providers", response_model=list[ProviderDescriptor])
    def get_providers() -> list[ProviderDescriptor]:
        return providers.descriptors()

    @app.get("/api/v1/provider-profiles", response_model=list[ProviderProfile])
    def get_provider_profiles() -> list[ProviderProfile]:
        return providers.profiles()

    @app.patch(
        "/api/v1/provider-profiles/{profile_id}",
        response_model=ProviderProfile,
    )
    def update_provider(profile_id: str, patch: ProviderPatch) -> ProviderProfile:
        try:
            return providers.update(profile_id, patch)
        except KeyError as exc:
            raise HTTPException(404, "Provider profile not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post(
        "/api/v1/provider-profiles/{profile_id}/test",
        response_model=ProviderTestResult,
    )
    def test_provider(
        profile_id: str,
        live: bool = Query(default=False),
    ) -> ProviderTestResult:
        try:
            return providers.probe(profile_id) if live else providers.test(profile_id)
        except KeyError as exc:
            raise HTTPException(404, "Provider profile not found") from exc

    @app.patch(
        "/api/v1/settings/default-provider",
        response_model=DefaultProviderResponse,
    )
    def set_default_provider(
        patch: DefaultProviderPatch,
    ) -> DefaultProviderResponse:
        try:
            profile_id = providers.activate(patch.provider_profile_id)
        except KeyError as exc:
            raise HTTPException(404, "Provider profile not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return DefaultProviderResponse(default_provider_profile_id=profile_id)

    @app.post("/api/v1/projects", response_model=ProjectCreated, status_code=201)
    def create_project(body: ProjectCreate) -> ProjectCreated:
        project = jobs.create_project(str(body.source.url))
        return ProjectCreated(id=project.id)

    @app.get("/api/v1/projects", response_model=ProjectListResponse)
    def get_projects(
        limit: int = Query(default=12, ge=1, le=50),
        cursor: str | None = Query(default=None, min_length=1, max_length=512),
    ) -> ProjectListResponse:
        try:
            return jobs.projects(limit=limit, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/projects/{project_id}", response_model=ProjectResponse)
    def get_project(project_id: str) -> ProjectResponse:
        try:
            return jobs.project(project_id)
        except KeyError as exc:
            raise HTTPException(404, "Project not found") from exc

    @app.delete("/api/v1/projects/{project_id}", status_code=204)
    def delete_empty_project(project_id: str) -> Response:
        try:
            jobs.delete_empty_project(project_id)
        except KeyError as exc:
            raise HTTPException(404, "Project not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @app.post(
        "/api/v1/projects/{project_id}/generations",
        response_model=JobResponse,
        status_code=202,
    )
    def create_generation(project_id: str, body: GenerationCreate) -> JobResponse:
        try:
            return jobs.create_generation(
                project_id,
                body.provider_profile_id,
                body.options,
            )
        except KeyError as exc:
            raise HTTPException(404, "Project or provider profile not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        try:
            jobs.delete_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @app.post("/api/v1/maintenance/retention", response_model=RetentionResult)
    def apply_retention(body: RetentionRequest) -> RetentionResult:
        if body.max_age_days is None and body.max_jobs is None:
            raise HTTPException(422, "Provide maxAgeDays or maxJobs")
        return jobs.apply_retention(
            max_age_days=body.max_age_days,
            max_jobs=body.max_jobs,
        )

    @app.get("/api/v1/system/diagnostics", response_model=SystemDiagnostics)
    def system_diagnostics() -> SystemDiagnostics:
        def executable_check(executable: Path, label: str) -> DiagnosticCheck:
            try:
                completed = subprocess.run(
                    [str(executable), "-version"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                healthy = completed.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                healthy = False
            return DiagnosticCheck(
                status="healthy" if healthy else "unhealthy",
                message=(
                    f"{label} is available."
                    if healthy
                    else f"{label} could not be started."
                ),
            )

        try:
            media_tools = resolve_media_tools()
            ffmpeg = executable_check(media_tools.ffmpeg, "FFmpeg")
            ffprobe = executable_check(media_tools.ffprobe, "FFprobe")
        except Exception:
            ffmpeg = DiagnosticCheck(
                status="unhealthy", message="FFmpeg is not configured."
            )
            ffprobe = DiagnosticCheck(
                status="unhealthy", message="FFprobe is not configured."
            )

        output_path = Path(output_root).expanduser().resolve()
        writable = False
        output_message = "The output directory is not writable."
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output_path, delete=True):
                writable = True
                output_message = "The output directory is writable."
        except OSError:
            pass
        try:
            free_bytes = shutil.disk_usage(output_path).free
        except OSError:
            free_bytes = None

        provider_checks: list[ProviderDiagnostic] = []
        for profile_id, state, message in providers.diagnostics():
            provider_checks.append(
                ProviderDiagnostic(
                    profile_id=profile_id,
                    status=state,  # type: ignore[arg-type]
                    message=message,
                )
            )

        healthy = (
            ffmpeg.status == "healthy"
            and ffprobe.status == "healthy"
            and writable
        )
        return SystemDiagnostics(
            status="healthy" if healthy else "degraded",
            media_tools=MediaToolDiagnostics(ffmpeg=ffmpeg, ffprobe=ffprobe),
            providers=provider_checks,
            output=OutputDiagnostics(
                writable=writable,
                free_bytes=free_bytes,
                message=output_message,
            ),
            timestamp=datetime.now(timezone.utc),
        )

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str) -> StreamingResponse:
        try:
            jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

        raw_last_event_id = request.headers.get("last-event-id", "").strip()
        try:
            initial_sequence = max(0, int(raw_last_event_id))
        except ValueError:
            initial_sequence = 0

        async def stream() -> AsyncIterator[bytes]:
            sequence = initial_sequence
            loop = asyncio.get_running_loop()
            last_heartbeat = loop.time()
            yield b"retry: 2500\n\n"
            while True:
                if await request.is_disconnected():
                    return
                events = jobs.events_after(job_id, sequence)
                for event in events:
                    sequence = event.sequence
                    yield jobs.encode_sse(event)
                    if event.type in {"completed", "failed"}:
                        return
                if not events and jobs.get(job_id).status.value in {"completed", "failed"}:
                    return
                if loop.time() - last_heartbeat >= 15:
                    yield b": keep-alive\n\n"
                    last_heartbeat = loop.time()
                await asyncio.sleep(0.15)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/assets/{asset_id}", response_class=FileResponse)
    def get_asset(
        asset_id: str,
        download: bool = Query(default=False),
    ) -> FileResponse:
        try:
            asset = jobs.assets.resolve(asset_id)
        except KeyError as exc:
            raise HTTPException(404, "Asset not found") from exc
        return FileResponse(
            asset.path,
            media_type=asset.media_type,
            filename=asset.path.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    # Mount the production React bundle last so API routes always take precedence.
    resolved_web_dist = (
        Path(web_dist).expanduser().resolve()
        if web_dist is not None
        else Path(__file__).resolve().parents[2] / "web" / "dist"
    )
    index_path = resolved_web_dist / "index.html"
    assets_path = resolved_web_dist / "assets"
    if index_path.is_file():
        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="web-assets")

        @app.get("/api/{unmatched_api_path:path}", include_in_schema=False)
        def unmatched_api(unmatched_api_path: str) -> None:
            del unmatched_api_path
            raise HTTPException(404, "API route not found")

        @app.get("/{route:path}", include_in_schema=False)
        def spa_fallback(route: str) -> FileResponse:
            requested = (resolved_web_dist / route).resolve()
            try:
                requested.relative_to(resolved_web_dist)
            except ValueError:
                requested = index_path
            if route and requested.is_file():
                return FileResponse(requested)
            return FileResponse(index_path)

    return app


app = create_app()
