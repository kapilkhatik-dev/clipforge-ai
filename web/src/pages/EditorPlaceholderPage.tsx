import { ArrowLeft, Captions, Layers3, Scissors, SlidersHorizontal } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";

export function EditorPlaceholderPage() {
  const { projectId, clipId } = useParams();
  const location = useLocation();
  const requestedReturnPath = new URLSearchParams(location.search).get("returnTo");
  const returnPath = requestedReturnPath?.startsWith("/") && !requestedReturnPath.startsWith("//")
    ? requestedReturnPath
    : projectId ? `/projects/${encodeURIComponent(projectId)}` : "/create";
  return (
    <div className="editor-placeholder">
      <div className="editor-toolbar panel">
        <Link className="button button-ghost" to={returnPath}><ArrowLeft size={16} /> Back to results</Link>
        <div><span>Project {projectId}</span><strong>Clip {clipId}</strong></div>
        <span className="future-badge">Future workspace</span>
      </div>
      <div className="editor-stage panel">
        <div className="editor-stage-copy"><span className="section-kicker"><Scissors size={14} /> Non-destructive editing</span><h2>The editor has a home.</h2><p>This reserved route will support source-range trimming, caption styling, layout adjustments, and versioned exports without changing the project navigation later.</p></div>
        <div className="editor-wireframe" aria-hidden="true">
          <div className="wire-assets"><span /><span /><span /></div>
          <div className="wire-canvas"><div><Captions size={18} /><strong>YOUR CAPTION</strong></div></div>
          <div className="wire-inspector"><SlidersHorizontal size={17} /><span /><span /><span /></div>
          <div className="wire-timeline"><Layers3 size={16} /><span /><span /><span /></div>
        </div>
      </div>
    </div>
  );
}
