export interface ProcessingPreferences {
  contentType: string;
  clipCountMode: "all" | "limit";
  clipCount: number;
  minDuration: number;
  maxDuration: number;
  layout: "fill-crop" | "fit-blur";
  montage: boolean;
}

export const PROCESSING_DEFAULTS: ProcessingPreferences = {
  contentType: "auto",
  clipCountMode: "all",
  clipCount: 5,
  minDuration: 20,
  maxDuration: 60,
  layout: "fill-crop",
  montage: true,
};

const STORAGE_KEY = "clipforge.processingDefaults";
export const CONTENT_TYPES = [
  "auto", "general", "comedy", "interview", "podcast", "education",
  "storytelling", "news", "commentary", "gaming", "sports", "business",
] as const;

const CONTENT_TYPE_SET = new Set<string>(CONTENT_TYPES);

function boundedNumber(value: unknown, fallback: number, min: number, max: number) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, value))
    : fallback;
}

export function loadProcessingPreferences(): ProcessingPreferences {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") as Record<string, unknown>;
    const minDuration = boundedNumber(parsed.minDuration, PROCESSING_DEFAULTS.minDuration, 5, 60);
    const maxDuration = boundedNumber(parsed.maxDuration, PROCESSING_DEFAULTS.maxDuration, 5, 60);
    return {
      contentType: typeof parsed.contentType === "string" && CONTENT_TYPE_SET.has(parsed.contentType)
        ? parsed.contentType
        : PROCESSING_DEFAULTS.contentType,
      clipCountMode: parsed.clipCountMode === "limit" ? "limit" : "all",
      clipCount: boundedNumber(parsed.clipCount, PROCESSING_DEFAULTS.clipCount, 1, 20),
      minDuration: Math.min(minDuration, maxDuration),
      maxDuration: Math.max(minDuration, maxDuration),
      layout: parsed.layout === "fit-blur" ? "fit-blur" : "fill-crop",
      montage: typeof parsed.montage === "boolean" ? parsed.montage : PROCESSING_DEFAULTS.montage,
    };
  } catch {
    return PROCESSING_DEFAULTS;
  }
}

export function saveProcessingPreferences(preferences: ProcessingPreferences) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
}
