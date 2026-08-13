import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import {
  ArrowRight,
  ChevronDown,
  Clapperboard,
  Clock3,
  Download,
  Film,
  FolderOpen,
  Gauge,
  Layers3,
  Link2,
  Maximize2,
  Play,
  RefreshCw,
  Scissors,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import { api, ApiError } from "../lib/api";
import { formatDuration, titleCase } from "../lib/format";
import { CONTENT_TYPES, loadProcessingPreferences } from "../lib/preferences";
import type {
  GenerationOptions,
  Job,
  PipelineStage,
  Project,
  ProjectSummary,
  ProviderProfile,
  ClipAsset,
} from "../lib/types";
import { Button, InlineAlert, Segmented, SelectMenu, Switch } from "../components/Ui";

const PIPELINE_STAGES: Array<{
  id: PipelineStage;
  label: string;
  description: string;
}> = [
  { id: "setup", label: "Setup", description: "Checking local tools" },
  { id: "inspect", label: "Inspect", description: "Reading source metadata" },
  { id: "download", label: "Download", description: "Preparing source media" },
  { id: "transcribe", label: "Transcript", description: "Building word timing" },
  { id: "analyze", label: "Analyze", description: "Finding standout moments" },
  { id: "render", label: "Render", description: "Styling and exporting" },
  { id: "complete", label: "Complete", description: "Your clips are ready" },
];

const DEFAULT_OPTIONS: GenerationOptions = {
  clipCount: null,
  contentType: "auto",
  videoLayout: "fill-crop",
  minClipDuration: 20,
  maxClipDuration: 60,
  highlightMontage: true,
  highlightWindowSeconds: 4,
  highlightMontageMaxDuration: 60,
  highlightMontageMaxMoments: 12,
  transcriptMode: "auto",
  analysisMaxConcurrency: 2,
  analysisRequestMaxAttempts: 3,
  force: false,
};

function isSupportedUrl(value: string) {
  try {
    const url = new URL(value);
    const hostname = url.hostname.toLowerCase();
    const youtubeHosts = new Set([
      "youtube.com",
      "www.youtube.com",
      "m.youtube.com",
      "music.youtube.com",
      "youtu.be",
      "www.youtu.be",
      "youtube-nocookie.com",
      "www.youtube-nocookie.com",
    ]);
    return (
      (url.protocol === "https:" || url.protocol === "http:") &&
      youtubeHosts.has(hostname)
    );
  } catch {
    return false;
  }
}

function SourcePreview({ url }: { url: string }) {
  const ready = isSupportedUrl(url);
  return (
    <div className={`source-preview ${ready ? "has-source" : ""}`}>
      <div className="source-preview-grid" aria-hidden="true" />
      <div className="preview-orb preview-orb-one" aria-hidden="true" />
      <div className="preview-orb preview-orb-two" aria-hidden="true" />
      <div className="source-preview-content">
        <span className="preview-icon">
          {ready ? <Play size={24} fill="currentColor" /> : <Film size={28} />}
        </span>
        <div>
          <strong>{ready ? "Source link ready" : "Your source preview"}</strong>
          <p>
            {ready
              ? "We’ll inspect the video after you start generation."
              : "Paste a public video link to begin your clip session."}
          </p>
        </div>
      </div>
      <div className="preview-ratio">16:9 source</div>
    </div>
  );
}

function ProgressPanel({ job }: { job: Job }) {
  const activeIndex = Math.max(
    0,
    PIPELINE_STAGES.findIndex((stage) => stage.id === job.stage),
  );
  return (
    <section className="panel progress-panel" aria-live="polite">
      <div className="section-heading">
        <div>
          <span className="section-kicker">Generation run</span>
          <h2>{job.status === "queued" ? "Waiting to begin" : "Crafting your clips"}</h2>
        </div>
        <span className={`status-badge status-${job.status}`}>{titleCase(job.status)}</span>
      </div>
      <div className="progress-summary">
        <span className="pulse-mark"><Sparkles size={18} /></span>
        <div>
          <strong>{job.message || "The pipeline is starting…"}</strong>
          <span>
            Stage progress
            {job.stageProgress != null ? ` · ${Math.round(job.stageProgress * 100)}%` : ""}
          </span>
        </div>
        {job.stageProgress != null && (
          <div className="progress-value">{Math.round(job.stageProgress * 100)}%</div>
        )}
      </div>
      <div className="progress-track" aria-hidden="true">
        <span style={{ width: `${(activeIndex / (PIPELINE_STAGES.length - 1)) * 100}%` }} />
      </div>
      <ol className="stage-list">
        {PIPELINE_STAGES.map((stage, index) => {
          const complete = job.status === "completed" || index < activeIndex;
          const active = index === activeIndex && job.status !== "completed";
          return (
            <li key={stage.id} className={`${complete ? "is-complete" : ""} ${active ? "is-active" : ""}`}>
              <span className="stage-index">{complete ? "✓" : index + 1}</span>
              <div>
                <strong>{stage.label}</strong>
                <small>{stage.description}</small>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="process-note">
        You can leave this page open while ClipForge works. Long sources may take several minutes.
      </p>
    </section>
  );
}

function ClipCard({ clip, featured, onPreview, returnTo }: { clip: ClipAsset; featured?: boolean; onPreview: () => void; returnTo: string }) {
  const image = clip.assets.posterUrl || clip.assets.thumbnailUrl;
  const montageMomentCount = clip.editDecisionList?.kind === "montage"
    ? clip.editDecisionList.sourceRanges.length
    : null;
  const isMontage = Boolean(featured) || montageMomentCount != null;
  return (
    <article className={`clip-card ${featured ? "clip-featured" : ""}`}>
      <button className="clip-poster" type="button" onClick={onPreview} aria-label={`Preview ${clip.title}`}>
        {image ? <img src={image} alt="" /> : <div className="poster-fallback"><Clapperboard size={30} /></div>}
        <span className="poster-play"><Play size={17} fill="currentColor" /></span>
        <span className="poster-duration">{formatDuration(clip.duration)}</span>
        {featured && <span className="featured-label"><Sparkles size={13} /> Highlight montage</span>}
      </button>
      <div className="clip-card-body">
        <div className="clip-card-title">
          <h3>{clip.title}</h3>
          {clip.score != null && <span className="score"><Gauge size={14} /> {Math.round(clip.score * 100)}</span>}
        </div>
        {(clip.hook || clip.reason) && <p>{clip.hook || clip.reason}</p>}
        <div className="clip-meta">
          {isMontage ? (
            <>
              <span><Clock3 size={14} /> {formatDuration(clip.duration)} total</span>
              <span><Layers3 size={14} /> {montageMomentCount == null ? "Multiple moments" : `${montageMomentCount} ${montageMomentCount === 1 ? "moment" : "moments"}`}</span>
            </>
          ) : (
            <span><Clock3 size={14} /> {formatDuration(clip.start)}–{formatDuration(clip.end)}</span>
          )}
          <span><Maximize2 size={14} /> 9:16</span>
        </div>
        <div className="clip-actions">
          <Button variant="secondary" onClick={onPreview}><Play size={15} /> Preview</Button>
          {clip.assets.downloadUrl && (
            <a className="button button-secondary" href={clip.assets.downloadUrl} download>
              <Download size={15} /> Export
            </a>
          )}
          {clip.editor?.available ? (
            <Link className="button button-ghost" to={`${clip.editor.route}?returnTo=${encodeURIComponent(returnTo)}`}><Scissors size={15} /> Edit</Link>
          ) : (
            <span className="coming-soon" title="Non-destructive clip editing is planned">
              <Scissors size={14} /> Editor soon
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

function ResultsPanel({
  job,
  onPreview,
  returnTo,
  onDelete,
  deleting,
}: {
  job: Job;
  onPreview: (clip: ClipAsset) => void;
  returnTo: string;
  onDelete: () => void;
  deleting: boolean;
}) {
  const result = job.result;
  const clips = result?.clips ?? [];
  if (!result || (!result.montage && clips.length === 0)) {
    return (
      <section className="panel result-empty result-empty-with-action">
        <div className="result-empty-copy"><span><Clapperboard size={25} /></span><div><h2>Generation completed without exported clips</h2><p>The AI did not find moments that met this run's quality and duration requirements.</p></div></div>
        <Button type="button" variant="danger" busy={deleting} onClick={onDelete}><Trash2 size={15} /> Delete generation</Button>
      </section>
    );
  }
  return (
    <section className="results-section">
      <div className="results-heading">
        <div>
          <span className="section-kicker">Ready to publish</span>
          <h2>{clips.length + (result.montage ? 1 : 0)} exports created</h2>
        </div>
        <p>Review each cut, then download the versions you want to share.</p>
      </div>
      {result.montage && (
        <section className="montage-spotlight" aria-labelledby={`montage-heading-${result.montage.id}`}>
          <header className="montage-spotlight-header">
            <span className="montage-spotlight-icon" aria-hidden="true"><Sparkles size={20} /></span>
            <div>
              <span className="section-kicker">Featured export</span>
              <h3 id={`montage-heading-${result.montage.id}`}>Best moments montage</h3>
              <p>The strongest moments, combined into one fast-paced vertical video.</p>
            </div>
            <div className="montage-output-specs" aria-label="Montage export limits">
              <span><Maximize2 size={14} /> 1080 × 1920 · 9:16 MP4</span>
              <span><Clock3 size={14} /> Up to 60 seconds · 20 moments</span>
            </div>
          </header>
          <ClipCard clip={result.montage} featured onPreview={() => onPreview(result.montage!)} returnTo={returnTo} />
        </section>
      )}
      {clips.length > 0 && (
        <div className="clip-grid">
          {clips.map((clip) => <ClipCard key={clip.id} clip={clip} onPreview={() => onPreview(clip)} returnTo={returnTo} />)}
        </div>
      )}
      <div className="results-danger-zone">
        <div><strong>Delete this generation</strong><p>Remove its exported clips, thumbnails, and local job record. This cannot be undone.</p></div>
        <Button type="button" variant="danger" busy={deleting} onClick={onDelete}><Trash2 size={15} /> Delete generation</Button>
      </div>
    </section>
  );
}

function PreviewDrawer({ clip, montage, onClose }: { clip: ClipAsset; montage: boolean; onClose: () => void }) {
  const drawerRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const montageMomentCount = clip.editDecisionList?.kind === "montage"
    ? clip.editDecisionList.sourceRanges.length
    : null;

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !drawerRef.current) return;
      const focusable = Array.from(
        drawerRef.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), video[controls], [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (focusable.length === 0) {
        event.preventDefault();
        drawerRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!drawerRef.current.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [onClose]);

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <aside
        ref={drawerRef}
        className="preview-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="preview-title"
        aria-describedby={clip.reason ? "preview-reason" : undefined}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-header">
          <div><span className="section-kicker">Clip preview</span><h2 id="preview-title">{clip.title}</h2></div>
          <button ref={closeButtonRef} className="icon-button" type="button" onClick={onClose} aria-label="Close preview"><X size={20} /></button>
        </div>
        <div className="phone-preview">
          <video controls playsInline poster={clip.assets.posterUrl || clip.assets.thumbnailUrl} src={clip.assets.videoUrl}>
            Your browser does not support video preview.
          </video>
        </div>
        <div className="drawer-details">
          <div><span>Duration</span><strong>{formatDuration(clip.duration)}</strong></div>
          <div>
            <span>{montage ? "Moments" : "Source range"}</span>
            <strong>{montage
              ? montageMomentCount == null
                ? "Multiple moments"
                : `${montageMomentCount} ${montageMomentCount === 1 ? "moment" : "moments"}`
              : `${formatDuration(clip.start)}–${formatDuration(clip.end)}`}</strong>
          </div>
          <div><span>AI score</span><strong>{clip.score != null ? `${Math.round(clip.score * 100)}/100` : "—"}</strong></div>
        </div>
        {clip.reason && <p className="drawer-reason" id="preview-reason">{clip.reason}</p>}
        {clip.assets.downloadUrl && <a className="button button-primary drawer-download" href={clip.assets.downloadUrl} download><Download size={16} /> Download MP4</a>}
      </aside>
    </div>
  );
}

function generationExportCount(job: Job) {
  return (job.result?.clips.length ?? 0) + (job.result?.montage ? 1 : 0);
}

function shortDate(value: string | undefined) {
  if (!value) return "Just now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Recently";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  }).format(date);
}

function RecentProjectsSection({
  projects,
  loading,
  loadingMore,
  error,
  hasMore,
  onRetry,
  onLoadMore,
}: {
  projects: ProjectSummary[];
  loading: boolean;
  loadingMore: boolean;
  error: string;
  hasMore: boolean;
  onRetry: () => void;
  onLoadMore: () => void;
}) {
  return (
    <section className="recent-projects" aria-labelledby="recent-projects-title">
      <div className="recent-projects-heading">
        <div>
          <span className="section-kicker"><FolderOpen size={14} /> Local workspace</span>
          <h2 id="recent-projects-title">Recent projects</h2>
        </div>
        <p>Reopen a previous source and switch between all of its generations.</p>
      </div>

      {loading && projects.length === 0 && (
        <div className="panel recent-projects-state" role="status">
          <span className="recent-state-icon"><FolderOpen size={22} /></span>
          <div><strong>Loading recent projects</strong><p>Reading your local ClipForge workspace…</p></div>
        </div>
      )}
      {!loading && error && projects.length === 0 && (
        <div className="panel recent-projects-state" role="alert">
          <span className="recent-state-icon"><FolderOpen size={22} /></span>
          <div><strong>Recent projects are unavailable</strong><p>{error}</p></div>
          <Button type="button" variant="secondary" onClick={onRetry}>Try again</Button>
        </div>
      )}
      {!loading && !error && projects.length === 0 && (
        <div className="panel recent-projects-state recent-projects-empty">
          <span className="recent-state-icon"><FolderOpen size={22} /></span>
          <div><strong>No saved generations yet</strong><p>Your first generated project will appear here automatically.</p></div>
        </div>
      )}
      {projects.length > 0 && (
        <>
          {error && <InlineAlert title="Could not load more projects" action={onRetry}>{error}</InlineAlert>}
          <div className="recent-project-grid">
            {projects.map((project) => {
              const latest = project.latestGeneration;
              return (
                <Link
                  key={project.id}
                  className="recent-project-card panel"
                  to={`/projects/${encodeURIComponent(project.id)}?job=${encodeURIComponent(latest.id)}`}
                >
                  <div className="recent-project-thumbnail">
                    {latest.thumbnailUrl ? <img src={latest.thumbnailUrl} alt="" /> : <Clapperboard size={25} aria-hidden="true" />}
                    <span className={`status-badge status-${latest.status}`}>{titleCase(latest.status)}</span>
                  </div>
                  <div className="recent-project-copy">
                    <h3>{latest.title || "Untitled video project"}</h3>
                    <p>{shortDate(latest.finishedAt || latest.startedAt || latest.createdAt)} · {latest.exportCount} {latest.exportCount === 1 ? "export" : "exports"}</p>
                    <span>{project.generationCount} {project.generationCount === 1 ? "generation" : "generations"}<ArrowRight size={14} /></span>
                  </div>
                </Link>
              );
            })}
          </div>
          {hasMore && (
            <div className="recent-load-more">
              <Button type="button" variant="secondary" busy={loadingMore} onClick={onLoadMore}>Load more projects</Button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export function CreatePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { projectId: routeProjectId } = useParams();
  const routeState = location.state as { job?: Job } | null;
  const routeStateJob = routeState?.job;
  const routeStateMismatch = Boolean(
    routeStateJob && routeProjectId && routeStateJob.projectId !== routeProjectId,
  );
  const initialPreferences = useMemo(() => loadProcessingPreferences(), []);
  const [url, setUrl] = useState("");
  const [countMode, setCountMode] = useState<"all" | "limit">(initialPreferences.clipCountMode);
  const [options, setOptions] = useState<GenerationOptions>({
    ...DEFAULT_OPTIONS,
    clipCount: initialPreferences.clipCountMode === "limit" ? initialPreferences.clipCount : null,
    contentType: initialPreferences.contentType,
    videoLayout: initialPreferences.layout,
    minClipDuration: initialPreferences.minDuration,
    maxClipDuration: initialPreferences.maxDuration,
    highlightMontage: initialPreferences.montage,
  });
  const [profiles, setProfiles] = useState<ProviderProfile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [projectDetails, setProjectDetails] = useState<Project | null>(null);
  const [recentProjects, setRecentProjects] = useState<ProjectSummary[]>([]);
  const [recentCursor, setRecentCursor] = useState<string | null>(null);
  const [recentLoading, setRecentLoading] = useState(false);
  const [recentLoadingMore, setRecentLoadingMore] = useState(false);
  const [recentError, setRecentError] = useState("");
  const [job, setJob] = useState<Job | null>(() => routeStateMismatch ? null : routeStateJob ?? null);
  const currentJobRef = useRef(job);
  const routeJobId = useMemo(
    () => new URLSearchParams(location.search).get("job"),
    [location.search],
  );
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [formError, setFormError] = useState(
    routeStateMismatch ? "This generation does not belong to the requested project." : "",
  );
  const [streamError, setStreamError] = useState("");
  const [preview, setPreview] = useState<{ clip: ClipAsset; montage: boolean } | null>(null);
  const restoredProjectRouteRef = useRef("");
  const restoredJobRouteRef = useRef("");
  const previousRouteProjectIdRef = useRef(routeProjectId);
  const jobId = job?.id;

  useEffect(() => {
    currentJobRef.current = job;
  }, [job]);

  useEffect(() => {
    const previousRouteProjectId = previousRouteProjectIdRef.current;
    previousRouteProjectIdRef.current = routeProjectId;
    if (!previousRouteProjectId || routeProjectId) return;

    setUrl("");
    setProjectDetails(null);
    setJob(null);
    setPreview(null);
    setFormError("");
    setStreamError("");
    restoredProjectRouteRef.current = "";
    restoredJobRouteRef.current = "";
  }, [routeProjectId]);

  const loadProviders = useCallback(async () => {
    setConnectionError("");
    try {
      const [bootstrap, nextProfiles] = await Promise.all([
        api.bootstrap(),
        api.providerProfiles(),
      ]);
      setProfiles(nextProfiles);
      const selected =
        nextProfiles.find((profile) => profile.active)?.id ??
        bootstrap.defaultProviderProfileId ??
        nextProfiles[0]?.id ??
        "";
      setSelectedProfileId(selected);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : "Could not load provider settings.");
    }
  }, []);

  useEffect(() => {
    void loadProviders();
  }, [loadProviders]);

  const loadRecentProjects = useCallback(async (cursor?: string, append = false) => {
    if (append) setRecentLoadingMore(true);
    else setRecentLoading(true);
    setRecentError("");
    try {
      const page = await api.projects(cursor);
      if (!Array.isArray(page.items)) {
        throw new Error("The local service returned an invalid recent-projects response.");
      }
      setRecentProjects((current) => append
        ? [...current, ...page.items.filter((item) => !current.some((existing) => existing.id === item.id))]
        : page.items);
      setRecentCursor(page.nextCursor ?? null);
    } catch (error) {
      setRecentError(error instanceof Error ? error.message : "Could not load recent projects.");
    } finally {
      setRecentLoading(false);
      setRecentLoadingMore(false);
    }
  }, []);

  useEffect(() => {
    if (!routeProjectId) void loadRecentProjects();
  }, [loadRecentProjects, routeProjectId]);

  useEffect(() => {
    if (!routeProjectId || restoredProjectRouteRef.current === routeProjectId) return;
    let active = true;
    restoredProjectRouteRef.current = routeProjectId;
    setProjectDetails(null);
    if (currentJobRef.current?.projectId !== routeProjectId) setJob(null);
    setPreview(null);
    setStreamError("");
    api.project(routeProjectId)
      .then((project) => {
        if (!active) return;
        setProjectDetails(project);
        setUrl(project.source.url);
        if (!routeJobId) setFormError("");
        setStreamError("");
        if (!routeJobId) setJob(project.jobs[0] ?? null);
      })
      .catch((projectError) => {
        if (!active) return;
        setProjectDetails(null);
        if (!routeJobId) setJob(null);
        if (!routeJobId) {
          setFormError(
            projectError instanceof ApiError && projectError.status === 404
              ? "This clip project is no longer available in the local service."
              : projectError instanceof Error
                ? projectError.message
                : "This clip project could not be restored.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [routeJobId, routeProjectId]);

  useEffect(() => {
    if (!job) return;
    setProjectDetails((current) => {
      if (!current || current.id !== job.projectId) return current;
      const jobs = current.jobs.some((item) => item.id === job.id)
        ? current.jobs.map((item) => item.id === job.id ? job : item)
        : [job, ...current.jobs];
      jobs.sort((left, right) => (right.createdAt ?? "").localeCompare(left.createdAt ?? ""));
      return { ...current, jobs };
    });
  }, [job]);

  useEffect(() => {
    if (!routeJobId || currentJobRef.current?.id === routeJobId) return;
    const jobRouteKey = `${routeProjectId ?? ""}:${routeJobId}`;
    if (restoredJobRouteRef.current === jobRouteKey) return;
    let active = true;
    restoredJobRouteRef.current = jobRouteKey;
    setJob(null);
    setPreview(null);
    setStreamError("");
    api.job(routeJobId)
      .then((restored) => {
        if (!active) return;
        if (routeProjectId && restored.projectId !== routeProjectId) {
          setJob(null);
          setFormError("This generation does not belong to the requested project.");
          return;
        }
        setFormError("");
        setStreamError("");
        setJob(restored);
      })
      .catch((restoreError) => {
        if (active) {
          setJob(null);
          setPreview(null);
          setFormError(
            restoreError instanceof Error
              ? restoreError.message
              : "This generation could not be restored.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [routeJobId, routeProjectId]);

  useEffect(() => {
    if (!jobId || !["queued", "running"].includes(currentJobRef.current?.status ?? "")) return;
    let disposed = false;
    let unsubscribe = () => {};
    let refreshTimer: number | undefined;
    let refreshInFlight = false;
    let consecutiveRefreshFailures = 0;
    let latestSequence = -1;
    const stopFallbackPolling = () => {
      if (refreshTimer !== undefined) {
        window.clearInterval(refreshTimer);
        refreshTimer = undefined;
      }
    };
    const refresh = async () => {
      if (refreshInFlight || disposed) return;
      refreshInFlight = true;
      try {
        const next = await api.job(jobId);
        consecutiveRefreshFailures = 0;
        if (!disposed) {
          setStreamError("");
          setJob((current) => {
            if (!current || current.id !== next.id) return next;
            if (["completed", "failed"].includes(current.status) &&
                !["completed", "failed"].includes(next.status)) return current;
            return next;
          });
          if (["completed", "failed"].includes(next.status)) {
            stopFallbackPolling();
            unsubscribe();
          }
        }
      } catch (refreshError) {
        consecutiveRefreshFailures += 1;
        if (!disposed && consecutiveRefreshFailures >= 3) {
          setStreamError(
            refreshError instanceof ApiError && refreshError.status === 404
              ? "This generation is no longer available in the local service."
              : "ClipForge lost contact with this generation. Check the local service, then refresh the page.",
          );
        }
      } finally {
        refreshInFlight = false;
      }
    };
    unsubscribe = api.subscribeToJob(
      jobId,
      (event) => {
        if (disposed) return;
        if (event.sequence != null && event.sequence <= latestSequence) return;
        if (event.sequence != null) latestSequence = event.sequence;
        consecutiveRefreshFailures = 0;
        setStreamError("");
        if (event.job) {
          setJob(event.job);
          if (["completed", "failed"].includes(event.job.status)) {
            stopFallbackPolling();
            unsubscribe();
          }
        } else if (event.event) {
          setJob((current) => {
            if (!current || current.id !== jobId) return current;
            return {
              ...current,
              stage: event.event?.stage ?? current.stage,
              stageProgress: event.event?.progress ?? current.stageProgress,
              message: event.event?.message ?? current.message,
            };
          });
        } else {
          void refresh();
        }
      },
      () => {
        if (disposed) return;
        if (refreshTimer === undefined) {
          void refresh();
          refreshTimer = window.setInterval(() => void refresh(), 2500);
        }
      },
      () => {
        if (disposed) return;
        stopFallbackPolling();
        consecutiveRefreshFailures = 0;
        setStreamError("");
      },
    );
    return () => {
      disposed = true;
      unsubscribe();
      stopFallbackPolling();
    };
  }, [jobId]);

  const retryJobSync = useCallback(async () => {
    if (!jobId) return;
    setStreamError("");
    try {
      setJob(await api.job(jobId));
    } catch (error) {
      setStreamError(
        error instanceof ApiError && error.status === 404
          ? "This generation is no longer available in the local service."
          : "ClipForge still cannot reach this generation. Check the local service and try again.",
      );
    }
  }, [jobId]);

  const deleteGeneration = useCallback(async () => {
    if (!job || !["completed", "failed"].includes(job.status) || deleting) return;
    if (!window.confirm("Delete this generation and all of its exported clips and thumbnails? This cannot be undone.")) return;
    setDeleting(true);
    setFormError("");
    setStreamError("");
    try {
      let projectJobs = projectDetails?.jobs ?? [];
      if (routeProjectId) {
        try {
          const refreshedProject = await api.project(routeProjectId);
          if (Array.isArray(refreshedProject.jobs)) {
            projectJobs = refreshedProject.jobs;
            setProjectDetails(refreshedProject);
          }
        } catch {
          // The already-rendered project snapshot remains sufficient for local
          // navigation when a refresh races with service cleanup.
        }
      }
      const nextJob = projectJobs
        .filter((candidate) => candidate.id !== job.id)
        .sort((left, right) => (right.createdAt ?? "").localeCompare(left.createdAt ?? ""))[0];
      await api.deleteJob(job.id);
      setPreview(null);
      restoredJobRouteRef.current = "";
      if (nextJob && routeProjectId) {
        setProjectDetails((current) => current ? {
          ...current,
          jobs: current.jobs.filter((candidate) => candidate.id !== job.id),
        } : current);
        setJob(nextJob);
        navigate(`/projects/${encodeURIComponent(routeProjectId)}?job=${encodeURIComponent(nextJob.id)}`, {
          replace: true,
          state: { job: nextJob },
        });
      } else {
        setProjectDetails(null);
        setJob(null);
        restoredProjectRouteRef.current = "";
        navigate("/create", { replace: true });
      }
    } catch (error) {
      setFormError(error instanceof ApiError ? error.message : "This generation could not be deleted.");
    } finally {
      setDeleting(false);
    }
  }, [deleting, job, navigate, projectDetails, routeProjectId]);

  const activeProfile = useMemo(
    () => profiles.find((profile) => profile.id === selectedProfileId),
    [profiles, selectedProfileId],
  );
  const providerReady = Boolean(activeProfile?.generationReady);

  async function startGeneration(event: FormEvent) {
    event.preventDefault();
    setFormError("");
    if (!isSupportedUrl(url)) {
      setFormError("Enter a complete public YouTube video URL.");
      return;
    }
    if (!selectedProfileId) {
      setFormError("Configure and activate an AI provider before generating clips.");
      return;
    }
    if (!providerReady) {
      setFormError("Finish checking and activating the selected AI provider before generating clips.");
      return;
    }
    if (options.minClipDuration > options.maxClipDuration) {
      setFormError("Minimum clip length cannot exceed the maximum clip length.");
      return;
    }
    setSubmitting(true);
    let createdProjectId: string | undefined;
    try {
      const normalizedUrl = url.trim();
      const canReuseCurrentProject = Boolean(
        routeProjectId &&
        projectDetails?.id === routeProjectId &&
        projectDetails.source.url.trim() === normalizedUrl,
      );
      let targetProjectId = routeProjectId;
      if (!canReuseCurrentProject || !targetProjectId) {
        const project = await api.createProject(normalizedUrl);
        targetProjectId = project.id;
        createdProjectId = project.id;
      }
      const created = await api.createGeneration(targetProjectId, selectedProfileId, {
        ...options,
        clipCount: countMode === "all" ? null : options.clipCount ?? 5,
      });
      setJob(created);
      navigate(`/projects/${encodeURIComponent(targetProjectId)}?job=${encodeURIComponent(created.id)}`, {
        replace: true,
        state: { job: created },
      });
    } catch (error) {
      if (createdProjectId) {
        try {
          await api.deleteProject(createdProjectId);
        } catch {
          // Keep the generation error actionable; abandoned drafts are omitted
          // from recent projects and may be cleaned by local maintenance later.
        }
      }
      setFormError(
        error instanceof ApiError ? error.message : "Clip generation could not be started.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  const busy = submitting || job?.status === "queued" || job?.status === "running";
  const closePreview = useCallback(() => setPreview(null), []);
  const returnPath = `${location.pathname}${location.search}`;

  return (
    <div className="create-page">
      <div className="page-intro">
        <div>
          <span className="section-kicker"><WandSparkles size={14} /> AI-assisted editing</span>
          <h2>Turn one long video into<br /><em>every moment worth sharing.</em></h2>
        </div>
        <p>Set the creative brief once. ClipForge finds the strongest moments, adds polished captions, and prepares vertical exports.</p>
      </div>

      {projectDetails && projectDetails.jobs.length > 0 && (
        <section className="panel generation-switcher" aria-label="Project generation history">
          <div>
            <span className="section-kicker">Project history</span>
            <strong>{projectDetails.jobs.length} {projectDetails.jobs.length === 1 ? "generation" : "generations"} for this source</strong>
          </div>
          <div className="generation-field">
            <label htmlFor="generation-select">Viewing generation</label>
            <SelectMenu
              id="generation-select"
              ariaLabel="Viewing generation"
              value={job?.id ?? ""}
              onChange={(jobId) => navigate(`/projects/${encodeURIComponent(projectDetails.id)}?job=${encodeURIComponent(jobId)}`)}
              options={projectDetails.jobs.map((candidate) => ({
                value: candidate.id,
                label: `${shortDate(candidate.finishedAt || candidate.startedAt || candidate.createdAt)} · ${titleCase(candidate.status)} · ${generationExportCount(candidate)} exports`,
              }))}
            />
          </div>
        </section>
      )}

      <form className="creation-layout" onSubmit={startGeneration}>
        <div className="creation-canvas">
          <section className="panel source-panel">
            <div className="section-heading">
              <div><span className="step-label">01</span><h2>Add your source</h2></div>
              <span className="support-note">Public YouTube URLs</span>
            </div>
            <div className="source-input-wrap">
              <Link2 size={18} aria-hidden="true" />
              <label className="sr-only" htmlFor="source-url">Video URL</label>
              <input
                id="source-url"
                type="url"
                value={url}
                placeholder="Paste a public YouTube video URL"
                onChange={(event) => setUrl(event.target.value)}
                disabled={busy}
                autoComplete="url"
              />
              {url && <button type="button" onClick={() => setUrl("")} aria-label="Clear video URL"><X size={16} /></button>}
            </div>
            <SourcePreview url={url} />
          </section>

          {job && ["queued", "running"].includes(job.status) && <ProgressPanel job={job} />}
          {job?.status === "failed" && (
            <section className="failed-generation">
              <InlineAlert title="This generation stopped">
                {job.error?.message || "The pipeline could not complete this run. Your settings have been preserved."}
              </InlineAlert>
              <Button type="button" variant="danger" busy={deleting} onClick={() => void deleteGeneration()}><Trash2 size={15} /> Delete generation</Button>
            </section>
          )}
          {job?.status === "completed" && <ResultsPanel job={job} onPreview={(clip) => setPreview({ clip, montage: job.result?.montage?.id === clip.id })} returnTo={returnPath} onDelete={() => void deleteGeneration()} deleting={deleting} />}

          {!job && (
            <section className="panel empty-workspace">
              <div className="empty-visual" aria-hidden="true">
                <div className="mini-frame frame-a"><span>HOOK</span></div>
                <div className="mini-frame frame-b"><Scissors size={19} /></div>
                <div className="mini-frame frame-c"><span>9:16</span></div>
              </div>
              <div><h2>Your best moments will land here</h2><p>Generated clips appear in a review-ready grid, with the highlight montage shown first.</p></div>
            </section>
          )}
        </div>

        <aside className="recipe-panel panel">
          <div className="recipe-header">
            <div><span className="step-label">02</span><h2>Shape the edit</h2></div>
            <Sparkles size={18} />
          </div>

          <fieldset className="recipe-controls" disabled={busy}>
          <div className="control-group">
            <label htmlFor="content-type">Creative direction</label>
            <p>Guide the AI toward moments that fit your content.</p>
            <SelectMenu
              id="content-type"
              ariaLabel="Creative direction"
              value={options.contentType}
              onChange={(contentType) => setOptions({ ...options, contentType })}
              disabled={busy}
              options={CONTENT_TYPES.map((type) => ({
                value: type,
                label: type === "auto" ? "Auto-detect video type" : titleCase(type),
              }))}
            />
          </div>

          <div className="control-group">
            <label>Clip quantity</label>
            <p>Quality always wins. Choose every strong moment or set a cap.</p>
            <Segmented
              label="Clip quantity mode"
              value={countMode}
              onChange={(value) => {
                setCountMode(value);
                setOptions({ ...options, clipCount: value === "all" ? null : options.clipCount ?? 5 });
              }}
              options={[
                { value: "all", label: "All best clips", description: "AI decides" },
                { value: "limit", label: "Limit output", description: "Choose a count" },
              ]}
            />
            {countMode === "limit" && (
              <div className="range-line count-line">
                <input aria-label="Maximum number of clips" type="range" min="1" max="20" value={options.clipCount ?? 5} onChange={(e) => setOptions({ ...options, clipCount: Number(e.target.value) })} />
                <output>{options.clipCount ?? 5} clips</output>
              </div>
            )}
          </div>

          <div className="control-group">
            <label>Clip duration</label>
            <p>Set the acceptable range for each continuous clip.</p>
            <div className="dual-fields">
              <label><span>Minimum</span><span className="number-input"><input aria-label="Minimum clip duration" type="number" min="5" max="60" value={options.minClipDuration} onChange={(e) => setOptions({ ...options, minClipDuration: Number(e.target.value) })} /> sec</span></label>
              <label><span>Maximum</span><span className="number-input"><input aria-label="Maximum clip duration" type="number" min="5" max="60" value={options.maxClipDuration} onChange={(e) => setOptions({ ...options, maxClipDuration: Number(e.target.value) })} /> sec</span></label>
            </div>
          </div>

          <div className="control-group">
            <label>Vertical framing</label>
            <Segmented
              label="Vertical video layout"
              value={options.videoLayout}
              onChange={(videoLayout) => setOptions({ ...options, videoLayout })}
              options={[
                { value: "fill-crop", label: "Fill & crop", description: "Bold, edge-to-edge" },
                { value: "fit-blur", label: "Fit & blur", description: "Keep full frame" },
              ]}
            />
          </div>

          <div className="control-group montage-control">
            <div className="montage-control-heading">
              <span className="montage-control-icon" aria-hidden="true"><Sparkles size={18} /></span>
              <div><strong>Best moments montage</strong><small>A featured compilation alongside your individual clips.</small></div>
              <span className="montage-featured-badge">Featured</span>
            </div>
            <Switch checked={options.highlightMontage} onChange={(highlightMontage) => setOptions({ ...options, highlightMontage })} label="Include montage export" description="Combine the strongest short moments into one polished vertical reel." />
            <div className="montage-support" aria-label="Supported montage format and limits">
              <span><strong>1080 × 1920</strong>9:16 MP4</span>
              <span><strong>3–6 sec</strong>per moment</span>
              <span><strong>2–20</strong>moments</span>
              <span><strong>60 sec</strong>maximum</span>
            </div>
            {options.highlightMontage && (
              <div className="montage-settings">
                <label><span>Moment length</span><span className="number-input"><input aria-label="Highlight moment size" type="number" min="3" max="6" step="0.5" value={options.highlightWindowSeconds} onChange={(e) => setOptions({ ...options, highlightWindowSeconds: Number(e.target.value) })} /> sec</span></label>
                <label><span>Total duration cap</span><span className="number-input"><input aria-label="Maximum montage duration" type="number" min="12" max="60" value={options.highlightMontageMaxDuration} onChange={(e) => setOptions({ ...options, highlightMontageMaxDuration: Number(e.target.value) })} /> sec</span></label>
                <label className="montage-moment-limit"><span>Maximum moments</span><span className="number-input"><input aria-label="Maximum montage moments" type="number" min="2" max="20" value={options.highlightMontageMaxMoments} onChange={(e) => setOptions({ ...options, highlightMontageMaxMoments: Number(e.target.value) })} /> moments</span></label>
              </div>
            )}
          </div>

          <details className="advanced-settings">
            <summary><span><Layers3 size={16} /> Advanced processing</span><ChevronDown size={16} /></summary>
            <div className="advanced-body">
              <div className="advanced-select-field">
                <label htmlFor="transcript-source">Transcript source</label>
                <SelectMenu
                  id="transcript-source"
                  ariaLabel="Transcript source"
                  value={options.transcriptMode}
                  onChange={(transcriptMode) => setOptions({ ...options, transcriptMode })}
                  options={[{ value: "auto", label: "Automatic" }, { value: "captions", label: "Captions only" }, { value: "whisper", label: "Local Whisper" }]}
                />
              </div>
              <div className="dual-fields">
                <label><span>AI concurrency</span><input type="number" min="1" max="4" value={options.analysisMaxConcurrency} onChange={(e) => setOptions({ ...options, analysisMaxConcurrency: Number(e.target.value) })} /></label>
                <label><span>Retry attempts</span><input type="number" min="1" max="6" value={options.analysisRequestMaxAttempts} onChange={(e) => setOptions({ ...options, analysisRequestMaxAttempts: Number(e.target.value) })} /></label>
              </div>
              <Switch checked={options.force} onChange={(force) => setOptions({ ...options, force })} label="Rebuild cached work" description="Re-run analysis and rendering for this source." />
            </div>
          </details>
          </fieldset>

          {connectionError && <InlineAlert title="Local service unavailable" action={() => void loadProviders()}>{connectionError}</InlineAlert>}
          {streamError && <InlineAlert title="Generation connection interrupted" action={() => void retryJobSync()}>{streamError}</InlineAlert>}
          {formError && <InlineAlert title="Check your setup">{formError}</InlineAlert>}

          <div className="generate-area">
            <div className="selected-provider">
              <span>Using</span>
              <strong>{activeProfile?.name ?? "No active provider"}</strong>
              {activeProfile && <small>{activeProfile.model}</small>}
            </div>
            <Button type="submit" busy={submitting} disabled={busy || Boolean(connectionError) || !selectedProfileId || !providerReady}>
              {busy ? "Generation in progress" : "Generate clips"}<ArrowRight size={17} />
            </Button>
            {(!selectedProfileId || !providerReady) && !connectionError && (
              <div className="provider-setup-note" role="note">
                <p>Configure your AI provider in <code>.env</code>, restart ClipForge, then refresh this status.</p>
                <Button type="button" variant="ghost" onClick={() => void loadProviders()}><RefreshCw size={14} /> Refresh provider</Button>
              </div>
            )}
          </div>
        </aside>
      </form>

      {!routeProjectId && (
        <RecentProjectsSection
          projects={recentProjects}
          loading={recentLoading}
          loadingMore={recentLoadingMore}
          error={recentError}
          hasMore={Boolean(recentCursor)}
          onRetry={() => void loadRecentProjects()}
          onLoadMore={() => recentCursor && void loadRecentProjects(recentCursor, true)}
        />
      )}

      {preview && <PreviewDrawer clip={preview.clip} montage={preview.montage} onClose={closePreview} />}
    </div>
  );
}
