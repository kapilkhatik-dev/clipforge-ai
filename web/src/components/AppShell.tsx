import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { CircleHelp, Plus, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import { Brand } from "./Brand";
import { StatusDot } from "./Ui";

function pageDetails(pathname: string) {
  if (pathname.includes("/edit")) {
    return { eyebrow: "Clip workspace", title: "Editor" };
  }
  return { eyebrow: "Creation studio", title: "Create clips" };
}

export function AppShell() {
  const location = useLocation();
  const [service, setService] = useState<"online" | "offline" | "checking">("checking");
  const [providerName, setProviderName] = useState("Checking provider");
  const details = pageDetails(location.pathname);
  const createRouteActive = location.pathname === "/create" || location.pathname.startsWith("/projects/");

  useEffect(() => {
    let active = true;
    const refreshProvider = () => {
      setService("checking");
      Promise.all([api.bootstrap(), api.providerProfiles()])
        .then(([bootstrap, profiles]) => {
          if (!active) return;
          const selected =
            profiles.find((profile) => profile.active) ??
            profiles.find((profile) => profile.id === bootstrap.defaultProviderProfileId);
          setProviderName(selected?.name ?? "Provider setup needed");
          setService("online");
        })
        .catch(() => {
          if (!active) return;
          setProviderName("Service offline");
          setService("offline");
        });
    };
    const handleProviderChanged = () => refreshProvider();
    refreshProvider();
    window.addEventListener("clipforge:provider-changed", handleProviderChanged);
    return () => {
      active = false;
      window.removeEventListener("clipforge:provider-changed", handleProviderChanged);
    };
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <Brand />
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <Link to="/create" className={createRouteActive ? "active" : ""} aria-current={createRouteActive ? "page" : undefined}>
            <Plus size={18} />
            <span>Create</span>
          </Link>
        </nav>
        <div className="sidebar-spacer" />
        <div className="studio-note">
          <Sparkles size={17} />
          <strong>Editing is next</strong>
          <p>Your clip workspace is ready for a future timeline editor.</p>
        </div>
        <nav className="secondary-nav" aria-label="Support navigation">
          <a href="https://github.com/kapilkhatik-dev/clipforge-ai" target="_blank" rel="noreferrer">
            <CircleHelp size={17} /> Documentation
          </a>
        </nav>
      </aside>

      <div className="app-content">
        <header className="topbar">
          <div>
            <span className="page-eyebrow">{details.eyebrow}</span>
            <h1>{details.title}</h1>
          </div>
          <div className="topbar-actions">
            <div
              className={`provider-chip provider-${service}`}
              role="status"
              aria-live="polite"
              aria-label={`Local service ${service}. Active provider: ${providerName}`}
              title="Local API connection status"
            >
              <StatusDot status={service} />
              <span>{providerName}</span>
            </div>
            <span className="avatar" aria-label="Local workspace">
              CF
            </span>
          </div>
        </header>
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
