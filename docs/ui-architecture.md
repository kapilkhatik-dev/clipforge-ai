# ClipForge UI architecture

## Product structure

ClipForge uses a persistent workbench instead of a multi-step wizard. The primary
navigation intentionally contains one destination:

- `/create` - source selection, clip recipe, processing progress, and results.

Provider and model configuration is handled through `.env`. The unused browser
Settings surface and its routes were removed to keep the product focused.

Generated work is addressed by stable project and clip identifiers. Project, job,
result, event, and opaque-asset records are persisted in a versioned local SQLite
store so completed work can be reopened after restarting the service:

- `/projects/:projectId` reopens a generation workspace.
- `/projects/:projectId/clips/:clipId/edit` is reserved for the future
  non-destructive editor.

The Create page also renders a cursor-paginated Recent Projects library. Its cards
use bounded summary records rather than embedding every clip result, and a project
workspace provides a generation selector for revisiting earlier runs of the same
source. Empty drafts never appear in the library and are removed after a failed
submission or after their final generation is deleted.

The editor route is part of the routing and data model now, but it is not shown as
a dead primary-navigation item. A clip result may expose an unavailable Edit
affordance until editing is implemented.

## Google Stitch direction

The UI direction was designed and reviewed in a private Google Stitch workspace.
Private workspace, screen, and asset identifiers are intentionally omitted from
this public repository. The reviewed concept was a media-first Clip Creation
Workbench that keeps source setup, processing, and results in one workspace.

The visual system uses:

- graphite canvas `#0B0D12` and raised panels near `#121722`;
- warm yellow `#F5C84C` for primary actions and caption-related cues;
- indigo `#7C83FF` only for AI and provider information;
- Geist-style headings, Inter-style body copy, an 8 px spacing grid, and
  restrained tonal elevation;
- 16:9 source and thumbnail surfaces, with 9:16 output previews;
- visible focus states, status text in addition to color, and 44 px minimum
  interactive targets.

## Creation workbench

Desktop uses an eight-column media/results canvas and a sticky four-column Clip
Recipe panel. Smaller layouts stack the recipe below the source.

The main workflow is:

1. Enter and inspect a supported video URL.
2. Choose an editorial goal and either **All best clips** (`clipCount: null`) or
   an explicit output limit.
3. Configure length, output layout, and optional best-moments montage.
4. Follow the real pipeline stages: setup, inspect, download, transcribe,
   analyze, render, and complete.
5. Review the montage first, followed by individual clips and their approved
   video, thumbnail, and poster assets.

Stage progress is local to the current operation and must not be represented as a
monotonic global percentage. The UI pairs a stage stepper with the latest human
readable pipeline event.

## Provider configuration boundary

The Create workbench reads the normalized active-provider profile and readiness
state returned by Python. It does not receive provider secrets or branch on Codex
versus a hosted API. Provider/model/key selection is made through `.env`, followed
by a local-service restart.

Provider-specific metadata, model ownership, fields, environment resolution, and
translation into `PipelineConfig` are centralized in Python's provider-definition
registry. Adding a provider still requires registering its core analysis adapter
and provider enum entry, but a new configuration shape no longer requires a
page-specific React form. Write-only field values are omitted from profiles and
generic patch payloads are validated against the selected provider's manifest.

Provider state and active-provider state remain separate in the API. The API
exposes `generationReady` separately from `active`, and a caller cannot bypass this
state by submitting an inactive profile ID directly to the generation endpoint.
The environment-selected startup profile remains usable without an interactive
check for CLI compatibility.
Credentials are write-only. API responses expose only whether a credential is
configured and whether its source is the environment or a process-memory override.
Runtime overrides last for the current local server session; durable secrets should
remain in environment variables until an OS keychain adapter is introduced.

Provider diagnostics distinguish local readiness from hosted configuration
presence. **Check setup** and system diagnostics intentionally avoid a potentially
billable inference call. The separate opt-in
`POST /api/v1/provider-profiles/:id/test?live=true` path makes one timeout-bounded
model request and may incur provider charges. Its sanitized result is informational:
it does not mutate the saved setup result or activation state.

Environment-controlled values are never rewritten by the browser.

## Future editor boundary

A project route identifies a generation workspace. Completed web-job assets are
snapshotted as generation-owned exports, while the core pipeline retains reusable
source/transcript caches. Projects, jobs, results, events, and opaque asset mappings
are persisted in the versioned local SQLite store. Every new continuous clip and
montage includes a versioned `editDecisionList` with:

- `sourceVideoId` and ordered `sourceRanges` (`start`/`end` seconds);
- `kind` (`continuous` or `montage`) and `videoLayout`;
- the caption preset used by the render.

A continuous clip has one source range. A montage retains all selected highlight
moments in playback order instead of flattening away their source timing. The future
editor should update this non-destructive decision list and render a new version
instead of changing an existing MP4.

The planned editor can therefore reuse the current shell and project context:

- asset/clip bin on the left;
- 9:16 player and canvas in the center;
- multitrack timeline at the bottom;
- transform, captions, title/thumbnail, and export inspectors on the right.

## Security boundary

The browser never receives arbitrary local paths, raw provider secrets, or the
contents of the output directory. The API maps approved artifacts to opaque asset
identifiers, binds locally by default, validates source URLs before download, and
runs the synchronous pipeline through a bounded worker rather than an async
request handler.

Public generation assets are copied into immutable per-job directories. Terminal
jobs can be deleted explicitly, and retention can prune them by age or total count;
active jobs are never removed. These policies remove public snapshots and their
opaque IDs while leaving reusable source and transcript caches intact.

Deletion uses a recoverable same-directory tombstone before the SQLite transaction.
If database removal fails, the snapshot and asset mappings are restored; startup
repairs or discards tombstones left by an interrupted cleanup. A maintenance lock
serializes manual deletion and retention.
