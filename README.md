# YouTube Clipper

A framework-neutral Python pipeline that downloads one YouTube video, obtains a timestamped transcript, asks a configured AI backend for strong short-form moments, and renders vertical captioned clips with FFmpeg. Analysis can use the locally installed Codex CLI with saved ChatGPT authentication or a LiteLLM-supported API provider. By default, it exports every high-quality clip that survives final review (up to the 20-candidate safety ceiling); callers can instead provide `clip_count=N` to export at most N clips.

The current release processes complete source videos whose YouTube metadata duration is up to **1 hour (3,600 seconds)**. `PipelineConfig` can impose a lower limit, but the application-wide content ceiling cannot be raised above one hour. Duration is checked during metadata inspection, enforced again inside yt-dlp before transfer, and verified on cached or downloaded media with FFprobe. Up to five seconds of mux/container padding is tolerated around otherwise valid metadata; transcript and clip timestamps remain bounded to the metadata duration. Downloads remain limited to <=1080p MP4, and the final source plus aggregate reported stream transfer are limited to **4 GiB** by default.

## Project structure

```text
main.py                              Debug-friendly development runner
.zed/
└── debug.json                       Zed Debugpy launch configurations
yt_clipper/
├── __init__.py                      Stable public API
├── config.py                        Environment-aware application defaults
├── application/
│   └── pipeline.py                  Framework-neutral orchestration
├── domain/
│   ├── errors.py                    Expected application failures
│   └── models.py                    Pydantic contracts and progress events
├── services/
│   ├── downloader.py                yt-dlp metadata and <=1080p MP4 download
│   ├── transcript.py                Captions with faster-whisper fallback
│   ├── analyzer.py                  Provider-neutral selection and validation
│   ├── codex_cli.py                 Structured local Codex CLI integration
│   └── renderer.py                  ASS captions and FFmpeg clip rendering
└── infrastructure/
    ├── artifacts.py                 Atomic writes and fingerprints
    └── media_tools.py               FFmpeg discovery, probing, and checks
tests/                               Offline unit tests
requirements.txt                     Runtime dependencies
requirements-dev.txt                 Development and test dependencies
```

`main.py` is only a development adapter. The package does not print, parse command-line arguments, or depend on a frontend framework. A future Streamlit, NiceGUI, API, or desktop adapter can import the same application service without refactoring the pipeline.

## LLM provider configuration

The application supports five provider values. Changing `CLIPPER_LLM_PROVIDER`
selects one backend; inactive providers' model and credential settings are ignored.

```dotenv
# codex | nvidia | openrouter | openai | anthropic
CLIPPER_LLM_PROVIDER=codex
```

The supported providers and built-in defaults are:

| Provider | Provider-specific model variable | Default model | Credential |
| --- | --- | --- | --- |
| `codex` | `CLIPPER_CODEX_MODEL` | `codex/default` | Saved local Codex login |
| `nvidia` | `CLIPPER_NVIDIA_MODEL` | `nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b` | `NVIDIA_NIM_API_KEY` or `NVIDIA_API_KEY` |
| `openrouter` | `CLIPPER_OPENROUTER_MODEL` | `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | `OPENROUTER_API_KEY` |
| `openai` | `CLIPPER_OPENAI_MODEL` | `openai/gpt-4.1-mini` | `OPENAI_API_KEY` |
| `anthropic` | `CLIPPER_ANTHROPIC_MODEL` | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |

Model configuration is resolved in this order:

1. `CLIPPER_LLM_MODEL`, a common override for the active provider.
2. The active provider's variable from the table above.
3. `CLIPPER_MODEL`, the deprecated compatibility alias.
4. The built-in provider default.

For switching configurations, leave `CLIPPER_LLM_MODEL` and `CLIPPER_MODEL`
unset and store one model per provider. Then only `CLIPPER_LLM_PROVIDER` needs to
change. For API credentials, `CLIPPER_LLM_API_KEY` is the common override;
otherwise the active provider's credential variable is used. Codex always ignores
API-key settings.

`nvidia` is the default and NVIDIA Nemotron 3 Ultra 550B A55B remains the recommended free development model for this workload. NVIDIA positions it for frontier reasoning, long-context analysis, tool use, and complex structured or agentic workflows—all directly relevant to finding coherent clips in long transcripts and returning validated selections. It supports up to a 1-million-token context window, although this application still sends bounded, overlapping transcript chunks to control latency and retries. See the [NVIDIA model reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b).

NVIDIA describes its hosted NIM endpoints as free serverless APIs for development. Treat that access as a rate- and capacity-limited trial suitable for development and prototyping, not as guaranteed unlimited free production service; availability, quotas, and terms can change. OpenAI and Anthropic API usage is not free.

### Environment variable reference

These are all user-configurable environment variables read by the application
(`LOCALAPPDATA` is a Windows system variable used only for Winget discovery):

| Variable | Required | Purpose |
| --- | --- | --- |
| `CLIPPER_VIDEO_URL` | For `main.py` when no URL argument is supplied | Input YouTube URL |
| `CLIPPER_CONTENT_TYPE` | No; defaults to `auto` | Editorial genre used to rank clips; set `comedy` to prioritize complete jokes and punchlines |
| `CLIPPER_LLM_PROVIDER` | No; defaults to `nvidia` | Active AI backend |
| `CLIPPER_LLM_MODEL` | No | Common active-provider model override; highest environment precedence |
| `CLIPPER_CODEX_MODEL` | No | Codex model, such as `codex/default` or `codex/gpt-5.6-sol` |
| `CLIPPER_OPENROUTER_MODEL` | No | LiteLLM-qualified OpenRouter model beginning with `openrouter/` |
| `CLIPPER_NVIDIA_MODEL` | No | LiteLLM-qualified NVIDIA NIM model beginning with `nvidia_nim/` |
| `CLIPPER_OPENAI_MODEL` | No | LiteLLM-qualified OpenAI model beginning with `openai/` |
| `CLIPPER_ANTHROPIC_MODEL` | No | LiteLLM-qualified Anthropic model beginning with `anthropic/` |
| `CLIPPER_LLM_API_KEY` | API providers only, unless using a provider-specific key | Common credential override for the active API provider |
| `NVIDIA_NIM_API_KEY`, `NVIDIA_API_KEY` | NVIDIA only | NVIDIA credential fallbacks |
| `OPENROUTER_API_KEY` | OpenRouter only | OpenRouter credential fallback |
| `OPENAI_API_KEY` | OpenAI only | OpenAI credential fallback |
| `ANTHROPIC_API_KEY` | Anthropic only | Anthropic credential fallback |
| `CLIPPER_CODEX_BINARY` | No; defaults to `codex` | Codex executable command or absolute path |
| `CLIPPER_CODEX_TIMEOUT_SECONDS` | No; defaults to `300` | Per-request Codex timeout, from 30 to 1,800 seconds |
| `FFMPEG_HOME` | Only when FFmpeg is not discoverable | Directory containing both FFmpeg and FFprobe |
| `FFMPEG_BINARY`, `FFPROBE_BINARY` | No | Individual media-tool command or path overrides |
| `CLIPPER_MODEL` | No | Deprecated model override retained for compatibility |

Other processing controls such as `clip_count`, durations, Whisper settings, output
directory, and analysis concurrency are typed `PipelineConfig` fields rather than
environment variables. Pass them from application/frontend code; the full list is
in [One-hour and resource settings](#one-hour-and-resource-settings).

### Using the local Codex CLI

Codex mode avoids configuring a separate model API key in this application. It is
not offline inference: the local CLI sends transcript prompts using the ChatGPT or
API authentication saved by Codex. Sign in once and verify the login:

```powershell
codex login
codex login status
```

Use `codex/default` to omit the CLI `--model` flag and let Codex select its
recommended model. To pin a model, use `codex/<model-id>`; the application removes
the `codex/` routing prefix and passes the remainder to `codex exec --model`.
Currently documented Codex choices include `gpt-5.6-sol` for maximum quality,
`gpt-5.6-terra` for balanced everyday work, and `gpt-5.6-luna` for fast,
well-scoped work. Availability depends on the signed-in account. See OpenAI's
[Codex model selection](https://learn.chatgpt.com/docs/models) and
[`codex exec` reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli#cli-codex-exec).

Complete Codex `.env` example:

```dotenv
CLIPPER_VIDEO_URL=https://www.youtube.com/watch?v=...
CLIPPER_LLM_PROVIDER=codex
CLIPPER_CODEX_MODEL=codex/gpt-5.6-sol
CLIPPER_CODEX_BINARY=codex
CLIPPER_CODEX_TIMEOUT_SECONDS=300
```

`CLIPPER_CODEX_BINARY=codex` first checks `PATH`. On Windows, the application
also discovers the CLI bundled with the newest installed OpenAI extension under
VS Code or VS Code Insiders. Set an absolute path only for another installation.

Each model step runs as a separate ephemeral `codex exec` process in an empty temporary directory. The integration supplies a strict JSON output schema, uses a read-only sandbox, disables shell, multi-agent, and web-search tools, ignores project/user rules, strips API-key and token environment variables, and deletes the temporary directory afterward. Codex CLI version plus the selected model are included in analysis cache identity, so CLI upgrades do not silently reuse incompatible Codex results.

OpenRouter and the other API providers do not pass through this subprocess adapter. They continue to use the existing LiteLLM request options, credentials, structured tool calls, retries, and cache behavior.

### Using OpenRouter

OpenRouter model IDs use the `openrouter/<author>/<model>` LiteLLM form. The
project default is the free Nemotron endpoint below. OpenRouter's catalog changes
over time, so use its [models catalog](https://openrouter.ai/models) or
[Models API](https://openrouter.ai/docs/api/api-reference/models/get-models) to
find another current slug, then prepend `openrouter/`.

Complete OpenRouter `.env` example:

```dotenv
CLIPPER_VIDEO_URL=https://www.youtube.com/watch?v=...
CLIPPER_LLM_PROVIDER=openrouter
CLIPPER_OPENROUTER_MODEL=openrouter/nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER_API_KEY=replace-with-your-openrouter-key
```

The common variables remain supported if a deployment injects one active model and
key dynamically:

```dotenv
CLIPPER_LLM_PROVIDER=openrouter
CLIPPER_LLM_MODEL=openrouter/openai/gpt-oss-120b:free
CLIPPER_LLM_API_KEY=replace-with-your-openrouter-key
```

Application or frontend code can instead supply `PipelineConfig(model="...")`.
An explicit field value takes precedence over environment resolution.

### Switching between Codex and OpenRouter

Keep both inactive-provider settings in `.env` and change only the provider line:

```dotenv
# Change only this value: codex or openrouter
CLIPPER_LLM_PROVIDER=codex

CLIPPER_CODEX_MODEL=codex/default
CLIPPER_CODEX_BINARY=codex
CLIPPER_CODEX_TIMEOUT_SECONDS=300

CLIPPER_OPENROUTER_MODEL=openrouter/nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER_API_KEY=replace-with-your-openrouter-key
```

Do not set `CLIPPER_LLM_MODEL`, `CLIPPER_LLM_API_KEY`, or `CLIPPER_MODEL` in this
switching-ready layout because common overrides take precedence. When `codex` is
active, the OpenRouter model and key are ignored. When `openrouter` is active, no
Codex subprocess is started. The providers also have separate analysis-cache
identities, preventing results from one backend from being reused by the other.

### Other provider examples

NVIDIA NIM:

```dotenv
CLIPPER_LLM_PROVIDER=nvidia
CLIPPER_NVIDIA_MODEL=nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_NIM_API_KEY=replace-with-your-nvidia-key
```

OpenAI API:

```dotenv
CLIPPER_LLM_PROVIDER=openai
CLIPPER_OPENAI_MODEL=openai/gpt-4.1-mini
OPENAI_API_KEY=replace-with-your-openai-key
```

Anthropic API:

```dotenv
CLIPPER_LLM_PROVIDER=anthropic
CLIPPER_ANTHROPIC_MODEL=anthropic/claude-sonnet-4-6
ANTHROPIC_API_KEY=replace-with-your-anthropic-key
```

The provider-specific endpoint and request details remain encapsulated internally. For API providers, LiteLLM receives the shared key directly on each request; the application does not copy it into global environment variables, logs, caches, or serialized configuration. Existing `NVIDIA_NIM_API_KEY`, `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` variables are still accepted as compatibility fallbacks when `CLIPPER_LLM_API_KEY` is absent. Codex mode never passes this shared key to the CLI.

The supported API defaults use typed `submit_clip_candidates` tool calls. Direct NVIDIA requests use NVIDIA's thinking options, OpenRouter retains its bounded hidden-reasoning payload, and Anthropic receives its native forced-tool selection shape. Codex uses `--output-schema` for the equivalent structured result. Plain-JSON parsing, local Pydantic validation, truncation detection, and one corrective follow-up remain available where applicable.

## Transcript strategy

No YouTube Data API key or official transcript API is required.

1. Prefer creator-provided captions exposed by `yt-dlp`.
2. Fall back to YouTube automatic captions exposed by `yt-dlp`.
3. If captions are unavailable, transcribe the downloaded media locally with `faster-whisper`.

This caption-first policy is especially important for long videos: when usable captions exist, the application skips local audio extraction and Whisper inference entirely. Individual caption responses are limited to 32 MiB to prevent an unexpected endpoint response from consuming unbounded memory. `language="auto"` still searches deterministic manual-caption tracks first, then automatic tracks. `yt-dlp` necessarily communicates with YouTube's web endpoints and may need updating when YouTube changes its site.

### Bounded local Whisper fallback

When local transcription is required, the pipeline is designed not to decode an entire one-hour recording into a large in-memory float array:

- FFmpeg extracts at most the metadata duration once as temporary 16 kHz, mono, signed 16-bit PCM. A full hour occupies about 110 MiB of temporary disk space.
- NumPy memory-maps that disk-backed PCM file.
- Faster-whisper processes five-minute cores with five seconds of context on each side. Overlap segments are assigned by midpoint to prevent duplicate transcript entries.
- Only the current window is converted to float32, keeping the temporary audio array around 20 MiB with the defaults.
- The Whisper model is loaded once per video job and reused across all windows.
- `whisper_device="auto"` uses CUDA when available and otherwise selects CPU. CPU uses int8; CUDA prefers `int8_float16` when supported.
- A single `output/.whisper.lock` serializes both PCM extraction and local Whisper inference across video jobs so concurrent frontend requests cannot unexpectedly multiply extraction CPU, temporary disk, RAM, or VRAM use.

`whisper_batch_size=1` is intentional. Values up to 8 are supported and may improve GPU throughput, but they can significantly increase RAM/VRAM consumption. The transcription timeout is a soft job deadline checked between operations; a native CTranslate2 inference call cannot be interrupted in the middle of that call.

## Long-video analysis and performance

One-hour transcripts are analyzed as bounded units rather than one large request:

- Transcript chunks default to at most **45,000 formatted characters** with **60 seconds of timestamp overlap**. The overlap must be at least the maximum clip duration so a complete clip can cross a chunk boundary.
- A strict **64-chunk safety limit** bounds total first-pass model work. If a single transcript segment is too large, or the configured overlap leaves no room for new content, analysis fails with an actionable configuration error instead of creating oversized prompts or hundreds of requests.
- Up to **two chunk requests are submitted concurrently** by default. The configurable hard maximum is four; additional work is submitted only as a worker finishes, bounding API pressure, queued futures, and worker memory.
- A chunk may validly return no candidates. Other chunks continue, and the job fails only when no usable candidate exists anywhere in the transcript. In automatic-count mode, each chunk may propose candidates up to the safety ceiling so the final output is not anchored to an arbitrary default such as five clips.
- Each successful first-pass result, including an empty result, is atomically cached under `output/<video-id>/analysis_chunks/chunk-XXX.json`. If a later chunk or final review fails, a rerun reuses successful chunks instead of starting over.
- The final review pool is stratified across the complete chunk timeline, preventing only the beginning of a long video from reaching review.
- One or more final continuity-review batches verify standalone context and exact transcript boundaries. Each batch's candidate-context text is bounded by `analysis_chunk_max_characters`; a single oversized context is rejected with an actionable error rather than sent unbounded.

For an analysis without a valid final cache, normal model traffic is **one logical request per missing or stale transcript chunk, plus one logical request per final review batch**. A typical single-chunk transcript still needs two calls. The 64 first-pass chunks and 20-candidate review-pool limits bound a fully uncached analysis to at most 84 logical calls in the extreme case where every review candidate needs its own batch. Each completion may still use configured provider retries, and malformed output can trigger one corrective completion. A valid `candidates.json` avoids all model calls, while valid chunk caches reduce a partially repeated run to only the missing chunks plus bounded final review batches.

These controls bound peak resource use, but they do not make a one-hour job instantaneous. Wall-clock time still depends on network speed, caption availability, CPU/GPU performance, model-provider latency and rate limits, and how many clips FFmpeg must render.

## Standalone clip selection

Clip selection uses two editorial stages for every uncached analysis:

1. A first pass proposes engaging excerpts, with explicit instructions not to truncate a longer discussion merely to fit the duration limit.
2. A separate continuity-review stage receives each proposal plus 15 seconds of transcript context before and after it. Proposals are split into bounded review batches when needed. The reviewer can adjust boundaries or reject the proposal and must document the single topic, opening setup, and closing resolution.

Final candidates must:

- Last no more than **60 seconds**. `PipelineConfig` rejects larger limits.
- Prefer the longest naturally complete version closest to the configured maximum.
- Never add unrelated material or a second topic merely to approach 60 seconds.
- Begin and end on transcript segment boundaries; small model timestamp drift is snapped to the exact boundary.
- Contain one complete thought, explanation, story beat, or topic.
- Supply enough opening context for a new viewer.
- End after the answer, conclusion, takeaway, or punchline.
- Include `standalone`, `topic`, `opening_context`, and `closing_resolution` evidence in `candidates.json`.

The reviewer is allowed to return fewer clips than its candidate pool. This intentionally favors standalone quality over filling a quota. After review, approved clips are ranked first by distance from the configured maximum duration and then by editorial score. Automatic mode exports the complete ranked set; an explicit clip count truncates it to that upper bound. Model-call counts depend on the number of missing transcript chunks as described above, which matters for free-tier rate limits.

## Content-aware clip selection

Set `CLIPPER_CONTENT_TYPE=comedy` to explicitly tell both existing AI analysis
stages that the input is comedy. The editor then prioritizes complete joke arcs,
punchlines, timing, surprise, quotability, and tightly connected reactions. It
rejects isolated laughter, setup without payoff, and callbacks that need missing
context.

`auto` remains the default. In auto mode, the model infers the dominant genre
inside the existing candidate-selection and continuity-review requests. There is
no separate classification API call, so Codex and OpenRouter retain their current
request count and provider behavior.

Supported values are `auto`, `general`, `comedy`, `interview`, `podcast`,
`education`, `storytelling`, `news`, `commentary`, `gaming`, `sports`, and
`business`. The same setting is available to application code:

```python
from yt_clipper import ContentType, PipelineConfig

config = PipelineConfig(content_type=ContentType.COMEDY)
```

The development runner also accepts a direct override:

```python
result = run(video_url, content_type="comedy")
```

## Clip count and duration settings

Current defaults are:

```text
clip_count = None (automatic)
min_clip_duration = 20 seconds
max_clip_duration = 60 seconds
```

`max_clip_duration` is both the hard ceiling and the preferred target. It can be lowered, but cannot exceed 60 seconds. When `clip_count` is omitted or set to `None`, analysis returns every high-quality candidate approved by the final continuity review, up to the 20-candidate safety ceiling. When `clip_count=N` is supplied, N is an upper bound rather than a quota: the system exports fewer clips when the AI finds fewer genuinely strong, standalone moments. Explicit values accept integers from 1 through 20.

The direct development runner now uses automatic-count mode when `clip_count` is omitted. Pass a value to request up to a specific number of clips:

```python
result = run(
    video_url,
    clip_count=10,
    video_layout=VideoLayout.FILL_CROP,
)
```

For application or frontend code, configure both values through `PipelineConfig`:

```python
config = PipelineConfig(
    clip_count=10,
    max_clip_duration=60,
)
```

To explicitly use automatic selection in application code:

```python
config = PipelineConfig(
    clip_count=None,
    max_clip_duration=60,
)
```

## One-hour and resource settings

All long-video controls are typed `PipelineConfig` fields, so a future frontend can expose them without changing the services:

| Setting | Default | Supported range / behavior |
| --- | ---: | --- |
| `clip_count` | `None` | Automatic: export every approved candidate up to 20; otherwise an integer from 1–20 is the output ceiling |
| `content_type` | `ContentType.AUTO` | Genre-specific ranking; supports the values documented in [Content-aware clip selection](#content-aware-clip-selection) |
| `max_source_duration_seconds` | `3600` | 60–3,600; lowers but cannot exceed the one-hour ceiling |
| `max_source_download_bytes` | `4 * 1024**3` | 256 MiB–16 GiB; checked by yt-dlp, aggregate stream progress, and final source validation |
| `whisper_device` | `WhisperDevice.AUTO` | `auto`, `cpu`, or `cuda` |
| `whisper_cpu_threads` | `4` | 1–32 |
| `whisper_batch_size` | `1` | 1–8; larger values use more RAM/VRAM |
| `whisper_chunk_seconds` | `300` | 60–600 seconds |
| `whisper_chunk_overlap_seconds` | `5` | 0–30 seconds and less than half the chunk size |
| `whisper_timeout_seconds` | `3600` | 300–14,400 seconds; soft deadline |
| `analysis_chunk_max_characters` | `45_000` | 8,000–100,000 formatted characters per first-pass transcript chunk or final-review context batch |
| `analysis_chunk_overlap_seconds` | `60.0` | Must be at least `max_clip_duration`; maximum 300 seconds |
| `analysis_max_concurrency` | `2` | 1–4 concurrent model requests |
| `analysis_request_max_attempts` | `3` | 1–6 provider attempts per completion call |
| `codex_timeout_seconds` | `300` | 30–1,800 seconds per `codex exec` invocation; used only by the Codex provider |

The defaults are the recommended balanced settings for one-hour support:

```python
from yt_clipper import PipelineConfig, WhisperDevice

config = PipelineConfig(
    max_source_duration_seconds=3600,
    max_source_download_bytes=4 * 1024**3,
    whisper_device=WhisperDevice.AUTO,
    whisper_batch_size=1,
    whisper_chunk_seconds=300,
    whisper_chunk_overlap_seconds=5,
    analysis_chunk_max_characters=45_000,
    analysis_chunk_overlap_seconds=60,
    analysis_max_concurrency=2,
)
```

Use a smaller `max_source_duration_seconds` or `max_source_download_bytes` when a deployment has stricter time, bandwidth, disk, or upload limits. Separate video/audio files and the merged output can coexist during yt-dlp post-processing, so peak temporary disk use can be higher than the final-source limit even though aggregate reported transfer is cancelled at the limit. Increase Whisper batching or analysis concurrency only after measuring the target machine and provider limits.

## Prerequisites

- Python 3.11 or newer
- FFmpeg and FFprobe with `libx264`, AAC, blur, overlay, and `subtitles`/libass support
- Either a signed-in local Codex CLI (`codex login`) or an API key from [NVIDIA](https://build.nvidia.com/settings/api-keys), [OpenRouter](https://openrouter.ai/settings/keys), [OpenAI](https://platform.openai.com/api-keys), or [Anthropic](https://console.anthropic.com/settings/keys)

Install FFmpeg on Windows:

```powershell
winget install --id Gyan.FFmpeg --exact
```

Open a new terminal afterward. The media resolver also detects Winget's Gyan FFmpeg installation directly. For another non-`PATH` installation, set `FFMPEG_HOME` to the directory containing both executables.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and choose a supported provider. The example keeps
Codex and OpenRouter model settings side by side so switching requires changing
only `CLIPPER_LLM_PROVIDER`. For Codex, run `codex login`; for an API provider,
replace its provider-specific key placeholder. When `main.run()` is not passed a
URL directly, also set the development input:

```dotenv
CLIPPER_VIDEO_URL=https://www.youtube.com/watch?v=...
CLIPPER_CONTENT_TYPE=comedy

# codex | nvidia | openrouter | openai | anthropic
CLIPPER_LLM_PROVIDER=codex

CLIPPER_CODEX_MODEL=codex/default
CLIPPER_OPENROUTER_MODEL=openrouter/nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER_API_KEY=replace-with-your-openrouter-key
```

For the default NVIDIA route, LiteLLM uses `https://integrate.api.nvidia.com/v1`. Other API endpoints or the local Codex subprocess path are selected automatically from `CLIPPER_LLM_PROVIDER`.

`.env`, `.venv`, downloaded media, and generated outputs are ignored by Git. `.env.example` contains placeholders only and can remain tracked.

## Run and debug directly

There is intentionally no argparse or command-line interface. `main.run()` reads `CLIPPER_VIDEO_URL` from `.env` when its `video_url` argument is omitted. The development block near the bottom of `main.py` currently passes a URL explicitly, which takes precedence; change that value directly for debugging or remove the argument to use `.env`:

```powershell
.venv\Scripts\python.exe main.py
```

For direct debugging in Zed:

1. Set the URL in the development block in `main.py`, or omit its `video_url` argument and set `CLIPPER_VIDEO_URL` in `.env`.
2. Put breakpoints in `main.py` or any `yt_clipper/` module.
3. Press **F4** and select **Debug YouTube Clipper**.
4. Step through the pipeline normally; exceptions are intentionally not swallowed by the runner.

`.zed/debug.json` also includes **Debug Active Python File**. Zed's Debugpy adapter should discover the project-local `.venv`; if it does not, select `.venv/Scripts/python.exe` as the workspace Python interpreter.

### Local video files

`main.run()` and `ClipPipeline.run()` also accept existing `.mp4`, `.m4v`, `.mkv`,
`.mov`, and `.webm` files. Local MP4 files are hard-linked into the output workspace
when possible, avoiding a second large copy. Other supported containers are remuxed
to MP4 without re-encoding.

Place an optional caption file beside the video using the same base name, such as
`video.srt`, `video.vtt`, or `video.en-auto.srt`. Matching sidecar captions are used
before Whisper; when no sidecar exists, automatic transcript mode falls back to
local Whisper normally.

```python
result = run(
    r"D:\Videos\comedy-special.mp4",
    content_type="comedy",
    clip_count=None,
)
```

You can also call the runner directly from a debugger or Python code without environment-based URL input:

```python
from dotenv import load_dotenv

from main import run
from yt_clipper import VideoLayout

load_dotenv()
result = run(
    "https://www.youtube.com/watch?v=...",
    video_layout=VideoLayout.FILL_CROP,
)
```

## Video layouts

Rendering is controlled by the typed `VideoLayout` user setting:

- `VideoLayout.FILL_CROP` (`"fill-crop"`) is the default. It uses a sharp, center-cropped version of the source as the full 9:16 frame—the same framing previously used by the blurred background.
- `VideoLayout.FIT_BLUR` (`"fit-blur"`) preserves the original composition, with the complete source fitted over a dimmed blurred background.

```python
from yt_clipper import PipelineConfig, VideoLayout

mobile_fill = PipelineConfig(video_layout=VideoLayout.FILL_CROP)
original_blur = PipelineConfig(video_layout=VideoLayout.FIT_BLUR)
```

A future frontend can map a select input directly to `"fill-crop"` or `"fit-blur"`. The selected layout is included in each render fingerprint. Switching it reuses the source video, transcript, and AI candidates, but safely rerenders the clips in the same video folder.

### Caption rendering

Generated captions use bold 80 px warm-yellow text, a crisp 5 px charcoal outline,
and a controlled dark shadow. This high-contrast treatment stays legible over both
bright and dark footage without the haloing caused by a light outline. Captions are
centered in the lower safe area with wider side margins and enough bottom clearance
for common short-form playback controls. Text is limited to seven words per cue and
balanced across at most two lines. Clips are encoded at CRF 18 so caption edges and
the underlying video retain more detail after compression.

Before writing the ASS file, the renderer merges repeated overlapping cues and
resolves all remaining timing overlaps into a single-active-caption timeline. This
prevents libass from stacking duplicate caption events while retaining the existing
word-level timing when it is available. The caption style and timeline policy are
included in the render fingerprint, so older cached clips are rerendered with the
new behavior on the next pipeline run.

### Clip titles and thumbnails

Every rendered clip uses the title selected by the AI during transcript analysis.
The title is stored in the MP4 container metadata and is also rendered onto a
1280x720 JPEG derived from the original YouTube video thumbnail. The thumbnail
design preserves the source artwork while adding a dark lower readability panel,
a warm-yellow accent, and balanced high-contrast title typography.

Each JPEG is saved beside its clip as `<clip-name>.thumbnail.jpg` for direct use by
a frontend, upload form, or social-media workflow. The same JPEG is embedded in the
MP4 as an attached cover-art stream for players and file browsers that support MP4
cover images. Players that do not display embedded MP4 artwork can use the sibling
JPEG explicitly. The normalized original artwork is cached as
`<video-id>/source-thumbnail.jpg`. If YouTube does not provide a usable original
thumbnail, rendering falls back to a representative frame from the middle of the
clip so video creation can still complete.

The renderer also creates `<clip-name>.poster.jpg`, a 1080x1920 (9:16) vertical
version for Instagram Reels covers and Stories. It places the original thumbnail
artwork flush against the top of a darkened, blurred full-frame background, then
anchors the accent and balanced AI title together near the top of a generous lower
safe area. Poster caching is independent from video
rendering, so adding or refreshing the vertical artwork does not re-encode a valid
MP4 or change its embedded horizontal preview.

## Frontend integration

Use the package's public API rather than importing `main.py`:

```python
from dotenv import load_dotenv

from yt_clipper import (
    ClipPipeline,
    LLMProvider,
    PipelineConfig,
    PipelineEvent,
    VideoLayout,
)

load_dotenv()


def on_progress(event: PipelineEvent) -> None:
    state = event.model_dump(mode="json")
    # Send state to Streamlit session state, a queue, WebSocket, etc.


pipeline = ClipPipeline(
    PipelineConfig(
        llm_provider=LLMProvider.NVIDIA,
        video_layout=VideoLayout.FILL_CROP,
        max_clip_duration=60,
    ),
    progress_callback=on_progress,
)
result = pipeline.run("https://www.youtube.com/watch?v=...")
```

The callback and result are typed Pydantic contracts. A frontend can pass its provider selector and, for API providers, a secret-input value as `llm_provider` and `llm_api_key`; the key is stored as Pydantic `SecretStr` and excluded from representation and serialization. Codex mode needs no secret-input value. Per-video file locks protect artifacts when a future frontend starts concurrent jobs.

## Output

```text
output/
├── .whisper.lock
└── <video-id>/
    ├── metadata.json
    ├── source.mp4
    ├── transcript.json
    ├── candidates.json
    ├── analysis_chunks/
    │   ├── chunk-001.json
    │   └── chunk-002.json
    ├── clips/
    │   ├── 01-example-title.mp4
    │   ├── 01-example-title.render.json
    │   ├── 02-another-title.mp4
    │   └── 02-another-title.render.json
    ├── .pipeline.lock
    └── temp/
```

Each MP4 in `clips/` has matching `.thumbnail.jpg`, `.poster.jpg`, and `.render.json` files. `output/.whisper.lock` is created only when local Whisper is used, and the number of chunk-analysis files depends on transcript size, up to the 64-chunk safety limit. Source media, the original thumbnail, transcripts, analysis, rendered clips, and titled artwork are fingerprinted or probed before reuse. The inexpensive source fingerprint includes file size, modification time, and the first and last MiB; Whisper cache fingerprints also include its processing options, while final analysis cache validation includes the prompt and chunk configuration. Video layout and horizontal thumbnail design are part of the render fingerprint, while vertical posters have an independent fingerprint. JSON, JPEG, and final MP4 artifacts are written transactionally, stale managed clips and artwork are removed after a successful render, and `PipelineConfig(force=True)` recreates the complete pipeline.

## Validation

Offline validation does not download YouTube videos or call model APIs:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m compileall -q main.py yt_clipper tests
.venv\Scripts\python.exe -m pip check
```

## Privacy, cost, and permitted use

The selected hosted AI service receives transcript text. This remains true in Codex mode: the executable is local, but model inference is provided through the Codex service under its active login. Review the selected service's retention policy before processing sensitive content. Local Whisper processes media on the machine but downloads model weights the first time a model is used.

Download and republish only content you own or are authorized to use. You are responsible for complying with YouTube's terms, copyright law, privacy rights, and platform rules.
