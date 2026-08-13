import type {
  BootstrapResponse,
  GenerationOptions,
  Job,
  JobEvent,
  Project,
  ProjectListResponse,
  ProviderDescriptor,
  ProviderPatch,
  ProviderProfile,
  SystemDiagnostics,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status = 0, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const method = (init?.method ?? "GET").toUpperCase();
  const isUnsafeMethod = !["GET", "HEAD", "OPTIONS"].includes(method);
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...(isUnsafeMethod ? { "X-ClipForge-Client": "web" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(
      "ClipForge could not reach the local service. Start the API and try again.",
    );
  }

  if (!response.ok) {
    let message = `The service returned ${response.status}.`;
    let code: string | undefined;
    try {
      const problem = (await response.json()) as {
        detail?: string | { message?: string; code?: string };
        message?: string;
        code?: string;
      };
      if (typeof problem.detail === "string") message = problem.detail;
      if (typeof problem.detail === "object" && problem.detail?.message) {
        message = problem.detail.message;
        code = problem.detail.code;
      }
      message = problem.message ?? message;
      code = problem.code ?? code;
    } catch {
      // The sanitized status message above is safer than leaking an upstream body.
    }
    throw new ApiError(message, response.status, code);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function unwrapList<T>(value: T[] | Record<string, T[]>, key: string): T[] {
  if (Array.isArray(value)) return value;
  return value[key] ?? [];
}

export const api = {
  bootstrap: () => request<BootstrapResponse>("/bootstrap"),
  diagnostics: () => request<SystemDiagnostics>("/system/diagnostics"),

  async providers() {
    const response = await request<
      ProviderDescriptor[] | { providers: ProviderDescriptor[] }
    >("/providers");
    return unwrapList(response, "providers");
  },

  async providerProfiles() {
    const response = await request<
      ProviderProfile[] | { profiles: ProviderProfile[] }
    >("/provider-profiles");
    return unwrapList(response, "profiles");
  },

  updateProvider: (id: string, patch: ProviderPatch) =>
    request<ProviderProfile>(`/provider-profiles/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  testProvider: (id: string, live = false) =>
    request<ProviderProfile["lastTest"]>(
      `/provider-profiles/${encodeURIComponent(id)}/test${live ? "?live=true" : ""}`,
      { method: "POST" },
    ),

  activateProvider: (providerProfileId: string) =>
    request<{ defaultProviderProfileId: string }>("/settings/default-provider", {
      method: "PATCH",
      body: JSON.stringify({ providerProfileId }),
    }),

  createProject: (url: string) =>
    request<{ id: string }>("/projects", {
      method: "POST",
      body: JSON.stringify({ source: { kind: "youtube", url } }),
    }),

  deleteProject: (id: string) => request<void>(`/projects/${encodeURIComponent(id)}`, {
    method: "DELETE",
  }),

  project: (id: string) => request<Project>(`/projects/${encodeURIComponent(id)}`),

  projects: (cursor?: string, limit = 12) => {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor) query.set("cursor", cursor);
    return request<ProjectListResponse>(`/projects?${query.toString()}`);
  },

  createGeneration: (
    projectId: string,
    providerProfileId: string,
    options: GenerationOptions,
  ) =>
    request<Job>(`/projects/${encodeURIComponent(projectId)}/generations`, {
      method: "POST",
      body: JSON.stringify({ providerProfileId, options }),
    }),

  job: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),

  deleteJob: (id: string) => request<void>(`/jobs/${encodeURIComponent(id)}`, {
    method: "DELETE",
  }),

  subscribeToJob(
    id: string,
    onEvent: (event: JobEvent) => void,
    onError: () => void,
    onOpen?: () => void,
  ) {
    const source = new EventSource(`${API_BASE}/jobs/${encodeURIComponent(id)}/events`);
    source.onopen = () => onOpen?.();
    source.onmessage = (message) => {
      try {
        onEvent(JSON.parse(message.data) as JobEvent);
      } catch {
        onError();
      }
    };
    source.onerror = onError;
    return () => source.close();
  },
};
