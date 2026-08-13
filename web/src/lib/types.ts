export type ProviderHealth = "healthy" | "unhealthy" | "untested" | "testing";
export type JobStatus = "queued" | "running" | "completed" | "failed";
export type PipelineStage =
  | "setup"
  | "inspect"
  | "download"
  | "transcribe"
  | "analyze"
  | "render"
  | "complete";

export interface ProviderModel {
  id: string;
  label: string;
}

export interface ProviderConfigurationField {
  key: string;
  label: string;
  inputType: "text" | "secret" | "number";
  section: string;
  sectionDescription?: string | null;
  helpText?: string | null;
  placeholder?: string | null;
  required: boolean;
  writeOnly: boolean;
  clearable: boolean;
  minLength?: number | null;
  maxLength?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  step?: number | null;
  suffix?: string | null;
}

export interface ProviderDescriptor {
  id: string;
  displayName: string;
  description?: string;
  transport: "local-cli" | "hosted-api" | string;
  requiresCredential: boolean;
  defaultModel: string;
  models: ProviderModel[];
  allowCustomModel: boolean;
  capabilities: string[];
  configurationFields?: ProviderConfigurationField[];
}

export interface ProviderProfile {
  id: string;
  providerId: string;
  name: string;
  model: string;
  active: boolean;
  generationReady: boolean;
  credential: {
    configured: boolean;
    source?: "environment" | "keyring" | string;
  };
  config?: {
    codexBinary?: string;
    codexTimeoutSeconds?: number;
  };
  configuration?: Record<string, {
    value?: string | number | boolean | null;
    configured: boolean;
    source?: "environment" | "runtime" | "default" | string | null;
  }>;
  lastTest?: {
    status: Exclude<ProviderHealth, "testing">;
    testedAt?: string;
    latencyMs?: number;
    message?: string;
  };
}

export interface BootstrapResponse {
  appName?: string;
  version?: string;
  defaultProviderProfileId?: string | null;
  capabilities?: {
    localUpload?: boolean;
    clipEditor?: boolean;
  };
}

export interface ClipAsset {
  id: string;
  title: string;
  start: number;
  end: number;
  duration: number;
  score?: number;
  hook?: string;
  reason?: string;
  assets: {
    videoUrl: string;
    downloadUrl?: string;
    thumbnailUrl?: string;
    posterUrl?: string;
  };
  editor?: {
    route: string;
    available: boolean;
  };
  editDecisionList?: {
    version: 1;
    kind: "continuous" | "montage";
    sourceVideoId: string;
    sourceRanges: Array<{ start: number; end: number }>;
    videoLayout: "fill-crop" | "fit-blur";
    captionPreset: string;
  };
}

export interface JobResult {
  source?: {
    title?: string;
    thumbnailUrl?: string;
    duration?: number;
  };
  montage?: ClipAsset | null;
  clips: ClipAsset[];
}

export interface Job {
  id: string;
  projectId: string;
  status: JobStatus;
  stage?: PipelineStage;
  stageProgress?: number | null;
  message?: string;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  result?: JobResult | null;
  error?: { code?: string; message: string } | null;
}

export interface Project {
  id: string;
  createdAt: string;
  source: {
    kind: "youtube";
    url: string;
  };
  jobs: Job[];
}

export interface GenerationSummary {
  id: string;
  status: JobStatus;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  title?: string | null;
  thumbnailUrl?: string | null;
  exportCount: number;
}

export interface ProjectSummary {
  id: string;
  createdAt: string;
  source: {
    kind: "youtube";
    url: string;
  };
  generationCount: number;
  latestGeneration: GenerationSummary;
}

export interface ProjectListResponse {
  items: ProjectSummary[];
  nextCursor?: string | null;
}

export interface SystemDiagnostics {
  status: "healthy" | "degraded";
  version: string;
  mediaTools: {
    ffmpeg: { status: "healthy" | "unhealthy"; message: string };
    ffprobe: { status: "healthy" | "unhealthy"; message: string };
  };
  providers: Array<{
    profileId: string;
    status: "healthy" | "unhealthy" | "unconfigured";
    message: string;
  }>;
  output: {
    writable: boolean;
    freeBytes?: number | null;
    message: string;
  };
  timestamp: string;
}

export interface GenerationOptions {
  clipCount: number | null;
  contentType: string;
  videoLayout: "fill-crop" | "fit-blur";
  minClipDuration: number;
  maxClipDuration: number;
  highlightMontage: boolean;
  highlightWindowSeconds: number;
  highlightMontageMaxDuration: number;
  highlightMontageMaxMoments: number;
  transcriptMode: "auto" | "captions" | "whisper";
  analysisMaxConcurrency: number;
  analysisRequestMaxAttempts: number;
  force: boolean;
}

export interface ProviderPatch {
  model?: string;
  configuration?: Record<string, string | number | boolean | null>;
  apiKey?: string;
  clearApiKey?: boolean;
  codexBinary?: string;
  codexTimeoutSeconds?: number;
}

export interface JobEvent {
  jobId?: string;
  sequence?: number;
  timestamp?: string;
  type?: string;
  event?: {
    stage?: PipelineStage;
    message?: string;
    progress?: number | null;
    current?: number | null;
    total?: number | null;
  };
  job?: Job;
}
