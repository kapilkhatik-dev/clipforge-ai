import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { CreatePage } from "./pages/CreatePage";
import { EditorPlaceholderPage } from "./pages/EditorPlaceholderPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/create" replace />} />
        <Route path="create" element={<CreatePage />} />
        <Route path="projects/:projectId" element={<CreatePage />} />
        <Route path="projects/:projectId/clips/:clipId/edit" element={<EditorPlaceholderPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
