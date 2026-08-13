import { Clapperboard } from "lucide-react";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand" aria-label="ClipForge AI">
      <span className="brand-mark" aria-hidden="true">
        <Clapperboard size={18} strokeWidth={2.4} />
      </span>
      {!compact && (
        <span className="brand-name">
          ClipForge <em>AI</em>
        </span>
      )}
    </div>
  );
}
