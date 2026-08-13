import { ArrowLeft, Film } from "lucide-react";
import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="not-found panel">
      <span><Film size={26} /></span>
      <h2>That frame isn’t in this cut.</h2>
      <p>The page may have moved or the link may be incomplete.</p>
      <Link className="button button-primary" to="/create"><ArrowLeft size={16} /> Return to creation</Link>
    </div>
  );
}
