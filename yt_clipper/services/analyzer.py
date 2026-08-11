"""Provider-neutral transcript analysis through Codex CLI or LiteLLM."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from litellm import completion
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import (
    DEFAULT_ANALYSIS_MODEL,
    LLMProvider,
    default_analysis_model,
    normalize_llm_provider,
)
from ..domain.errors import AnalysisError
from ..domain.models import (
    ANALYSIS_PROMPT_VERSION,
    MAX_CLIP_COUNT,
    MAX_CLIP_DURATION_SECONDS,
    ClipCandidate,
    TranscriptDocument,
    TranscriptSegment,
)
from ..infrastructure.artifacts import atomic_write_text, fingerprint_payload
from .codex_cli import CodexCLIClient, CodexCLIError

CompletionFunction = Callable[..., Any]
CandidateValidator = Callable[[Sequence[Any]], list[ClipCandidate]]
AnalysisChunkProgressCallback = Callable[[int, int], None]

LOGGER = logging.getLogger(__name__)
_CHUNK_ANALYSIS_SCHEMA_VERSION = 1
_CLIP_TOOL_NAME = "submit_clip_candidates"
_MAX_ANALYSIS_OUTPUT_TOKENS = 10_000
_NEMOTRON_REASONING_TOKENS = 2_000
_REVIEW_CONTEXT_PADDING_SECONDS = 15.0
_BOUNDARY_TOLERANCE_SECONDS = 0.35
_MAX_TRANSCRIPT_CHUNKS = 64
_NVIDIA_NIM_NEMOTRON_ULTRA_MODEL = DEFAULT_ANALYSIS_MODEL
_OPENROUTER_NEMOTRON_ULTRA_MODEL = default_analysis_model(LLMProvider.OPENROUTER)
_OPENAI_DEFAULT_MODEL = default_analysis_model(LLMProvider.OPENAI)
_ANTHROPIC_DEFAULT_MODEL = default_analysis_model(LLMProvider.ANTHROPIC)
_STRUCTURED_TOOL_MODELS = frozenset(
    {
        _NVIDIA_NIM_NEMOTRON_ULTRA_MODEL,
        _OPENROUTER_NEMOTRON_ULTRA_MODEL,
        _OPENAI_DEFAULT_MODEL,
        _ANTHROPIC_DEFAULT_MODEL,
    }
)


class ChunkAnalysisDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    fingerprint: str = Field(min_length=64, max_length=64)
    candidates: list[ClipCandidate]


class StandaloneClipReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=120)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    hook: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1000)
    standalone: bool
    topic: str = Field(min_length=1, max_length=240)
    opening_context: str = Field(min_length=1, max_length=500)
    closing_resolution: str = Field(min_length=1, max_length=500)


def _format_transcript_line(segment: TranscriptSegment) -> str:
    return f"[{segment.start:.3f} - {segment.end:.3f}] {segment.text}"


def chunk_transcript(
    segments: Sequence[TranscriptSegment],
    max_characters: int = 45_000,
    overlap_seconds: float = 30.0,
) -> list[list[TranscriptSegment]]:
    """Split a transcript into bounded chunks with timestamp overlap."""
    if not segments:
        return []
    if max_characters <= 0:
        raise AnalysisError("Transcript max_characters must be greater than zero.")
    if overlap_seconds < 0:
        raise AnalysisError("Transcript overlap_seconds cannot be negative.")

    line_sizes = [len(_format_transcript_line(segment)) for segment in segments]
    for index, (segment, line_size) in enumerate(zip(segments, line_sizes)):
        if line_size > max_characters:
            raise AnalysisError(
                f"A single transcript segment ({index + 1}, starting at "
                + f"{segment.start:.3f}s) formats to {line_size} characters, exceeding "
                + f"max_characters={max_characters}. Increase max_characters or split "
                + "the source transcript segment."
            )

    chunks: list[list[TranscriptSegment]] = []
    start = 0
    while start < len(segments):
        if len(chunks) >= _MAX_TRANSCRIPT_CHUNKS:
            raise AnalysisError(
                f"The transcript requires more than {_MAX_TRANSCRIPT_CHUNKS} chunks. "
                + "Increase max_characters or reduce overlap_seconds to keep model calls "
                + "within the safety limit."
            )

        end = start
        chunk_size = 0
        while end < len(segments):
            separator_size = 1 if end > start else 0
            proposed_size = chunk_size + separator_size + line_sizes[end]
            if proposed_size > max_characters:
                break
            chunk_size = proposed_size
            end += 1

        chunks.append(list(segments[start:end]))
        if end == len(segments):
            break

        if overlap_seconds == 0:
            next_start = end
        else:
            overlap_floor = start
            for gap_index in range(end, start, -1):
                gap_seconds = (
                    segments[gap_index].start - segments[gap_index - 1].end
                )
                if gap_seconds >= overlap_seconds:
                    overlap_floor = gap_index
                    break

            if overlap_floor == end:
                next_start = end
            else:
                overlap_boundary = segments[end - 1].end - overlap_seconds
                next_start = end - 1
                while (
                    next_start > overlap_floor
                    and segments[next_start].start > overlap_boundary
                ):
                    next_start -= 1

        overlap_with_new_size = (
            sum(line_sizes[next_start : end + 1]) + end - next_start
        )
        if next_start <= start or overlap_with_new_size > max_characters:
            next_segment = segments[end]
            raise AnalysisError(
                f"Cannot preserve {overlap_seconds:.1f}s of transcript overlap: "
                + f"it leaves no room for new content at {next_segment.start:.3f}s "
                + f"within max_characters={max_characters}. Increase max_characters "
                + "or reduce overlap_seconds."
            )

        start = next_start

    return chunks


def _chunk_fingerprint(
    chunk: Sequence[TranscriptSegment],
    model: str,
    requested_count: int,
    min_duration: float,
    max_duration: float,
    video_duration: float,
) -> str:
    return fingerprint_payload(
        {
            "schema_version": _CHUNK_ANALYSIS_SCHEMA_VERSION,
            "prompt_version": ANALYSIS_PROMPT_VERSION,
            "model": model,
            "requested_count": requested_count,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "video_duration": video_duration,
            "segments": [segment.model_dump(mode="json") for segment in chunk],
        }
    )


def select_diverse_review_pool(
    chunk_candidates: Sequence[Sequence[ClipCandidate]],
    limit: int,
    target_duration: float,
) -> list[ClipCandidate]:
    """Select duration-ranked candidates across the complete chunk timeline."""
    if limit <= 0 or not chunk_candidates:
        return []

    ranked_chunks = [
        deduplicate_candidates(candidates, target_duration=target_duration)
        for candidates in chunk_candidates
    ]
    available_indices = [
        index for index, candidates in enumerate(ranked_chunks) if candidates
    ]
    if not available_indices:
        return []

    coverage_count = min(limit, len(available_indices))
    if coverage_count == 1:
        coverage_indices = [available_indices[len(available_indices) // 2]]
    else:
        denominator = coverage_count - 1
        coverage_indices = [
            available_indices[
                (position * (len(available_indices) - 1) + denominator // 2)
                // denominator
            ]
            for position in range(coverage_count)
        ]

    coverage_set = set(coverage_indices)
    remaining_indices = [
        index for index in available_indices if index not in coverage_set
    ]
    remaining_indices.sort(
        key=lambda index: (
            -min(abs(index - covered) for covered in coverage_indices),
            index,
        )
    )
    chunk_order = coverage_indices + remaining_indices

    selected: list[ClipCandidate] = []
    maximum_chunk_size = max((len(candidates) for candidates in ranked_chunks), default=0)
    for candidate_index in range(maximum_chunk_size):
        for chunk_index in chunk_order:
            candidates = ranked_chunks[chunk_index]
            if candidate_index >= len(candidates):
                continue
            candidate = candidates[candidate_index]
            if all(_overlap_ratio(candidate, existing) < 0.6 for existing in selected):
                selected.append(candidate)
                if len(selected) >= limit:
                    return selected
    return selected


def extract_json_payload(text: str) -> dict[str, Any] | list[Any]:
    """Extract the first valid JSON object or array, tolerating Markdown prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise AnalysisError("The model response did not contain valid JSON.")


def _candidate_items(payload: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    for key in ("clips", "candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise AnalysisError("The model JSON must contain a 'clips' array.")


def validate_clip_candidates(
    items: Sequence[Any],
    video_duration: float,
    min_duration: float,
    max_duration: float,
    transcript_segments: Sequence[TranscriptSegment] | None = None,
    allowed_start: float = 0.0,
    allowed_end: float | None = None,
    require_segment_boundaries: bool = False,
    require_standalone: bool = False,
    allow_empty: bool = False,
) -> list[ClipCandidate]:
    if allow_empty and not items:
        return []

    valid: list[ClipCandidate] = []
    validation_messages: list[str] = []

    for index, item in enumerate(items):
        try:
            candidate = ClipCandidate.model_validate(item)
        except ValidationError as exc:
            validation_messages.append(f"clip {index + 1}: {exc.errors()[0]['msg']}")
            continue

        if transcript_segments and require_segment_boundaries:
            nearest_start = min(
                transcript_segments,
                key=lambda segment: abs(candidate.start - segment.start),
            ).start
            nearest_end = min(
                transcript_segments,
                key=lambda segment: abs(candidate.end - segment.end),
            ).end
            if (
                abs(candidate.start - nearest_start) > _BOUNDARY_TOLERANCE_SECONDS
                or abs(candidate.end - nearest_end) > _BOUNDARY_TOLERANCE_SECONDS
            ):
                validation_messages.append(
                    f"clip {index + 1}: timestamps do not align with transcript boundaries"
                )
                continue
            candidate = candidate.model_copy(
                update={"start": nearest_start, "end": nearest_end}
            )
            if candidate.end <= candidate.start:
                validation_messages.append(
                    f"clip {index + 1}: aligned timestamps have no duration"
                )
                continue

        if candidate.end > video_duration + 0.01:
            validation_messages.append(f"clip {index + 1}: end exceeds video duration")
            continue
        if candidate.start < allowed_start - 0.01 or (
            allowed_end is not None and candidate.end > allowed_end + 0.01
        ):
            validation_messages.append(f"clip {index + 1}: timestamps are outside this transcript chunk")
            continue
        if not min_duration <= candidate.duration <= max_duration:
            validation_messages.append(
                f"clip {index + 1}: duration {candidate.duration:.1f}s is outside "
                f"{min_duration:.1f}-{max_duration:.1f}s"
            )
            continue
        if require_standalone and not (
            candidate.standalone
            and candidate.topic
            and candidate.opening_context
            and candidate.closing_resolution
        ):
            validation_messages.append(
                f"clip {index + 1}: standalone review evidence is missing"
            )
            continue
        if transcript_segments and not any(
            segment.start < candidate.end and segment.end > candidate.start
            for segment in transcript_segments
        ):
            validation_messages.append(f"clip {index + 1}: no transcript overlap")
            continue
        valid.append(candidate)

    if not valid:
        details = "; ".join(validation_messages[:5]) or "no candidates were provided"
        raise AnalysisError(f"The model returned no usable clips: {details}")
    return valid


def validate_standalone_reviews(
    items: Sequence[Any],
    video_duration: float,
    min_duration: float,
    max_duration: float,
    transcript_segments: Sequence[TranscriptSegment],
    proposed_candidates: Sequence[ClipCandidate],
    allow_empty: bool = False,
) -> list[ClipCandidate]:
    if allow_empty and not items:
        return []

    reviewed: list[dict[str, Any]] = []
    validation_messages: list[str] = []
    for index, item in enumerate(items):
        try:
            review = StandaloneClipReview.model_validate(item)
        except ValidationError as exc:
            validation_messages.append(f"clip {index + 1}: {exc.errors()[0]['msg']}")
            continue
        matches_proposal_context = any(
            review.start >= proposal.start - _REVIEW_CONTEXT_PADDING_SECONDS - 0.01
            and review.end <= proposal.end + _REVIEW_CONTEXT_PADDING_SECONDS + 0.01
            and review.start < proposal.end
            and review.end > proposal.start
            for proposal in proposed_candidates
        )
        if not matches_proposal_context:
            validation_messages.append(
                f"clip {index + 1}: reviewed timestamps are outside proposed context"
            )
            continue
        reviewed.append(review.model_dump())

    if not reviewed:
        details = "; ".join(validation_messages[:5]) or "no reviews were provided"
        raise AnalysisError(f"The model approved no standalone clips: {details}")

    return validate_clip_candidates(
        reviewed,
        video_duration=video_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        transcript_segments=transcript_segments,
        require_segment_boundaries=True,
        require_standalone=True,
    )


def _format_candidate_review_context(
    candidate: ClipCandidate,
    index: int,
    transcript_segments: Sequence[TranscriptSegment],
) -> str:
    context_start = max(0.0, candidate.start - _REVIEW_CONTEXT_PADDING_SECONDS)
    context_end = candidate.end + _REVIEW_CONTEXT_PADDING_SECONDS
    lines = [
        "CANDIDATE "
        + f"{index}: "
        + candidate.model_dump_json(
            include={"title", "start", "end", "score", "hook", "reason"}
        ),
        "TIMESTAMPED CONTEXT:",
    ]
    for segment in transcript_segments:
        if segment.end <= context_start or segment.start >= context_end:
            continue
        if segment.end <= candidate.start + 0.01:
            position = "BEFORE"
        elif segment.start >= candidate.end - 0.01:
            position = "AFTER"
        else:
            position = "IN CLIP"
        lines.append(f"{position} {_format_transcript_line(segment)}")
    return "\n".join(lines)


def _batch_review_candidates(
    candidates: Sequence[ClipCandidate],
    transcript_segments: Sequence[TranscriptSegment],
    max_characters: int,
) -> list[list[ClipCandidate]]:
    """Group candidates by the exact context size sent for final review."""
    if max_characters <= 0:
        raise AnalysisError(
            "Final-review chunk_max_characters must be greater than zero."
        )

    batches: list[list[ClipCandidate]] = []
    current: list[ClipCandidate] = []
    current_size = 0
    for candidate in candidates:
        context = _format_candidate_review_context(
            candidate,
            len(current) + 1,
            transcript_segments,
        )
        separator_size = 2 if current else 0
        if current and current_size + separator_size + len(context) > max_characters:
            batches.append(current)
            current = []
            current_size = 0
            context = _format_candidate_review_context(
                candidate,
                1,
                transcript_segments,
            )
            separator_size = 0

        if len(context) > max_characters:
            raise AnalysisError(
                "A single final-review candidate context for "
                + f"'{candidate.title}' formats to {len(context)} characters, exceeding "
                + f"chunk_max_characters={max_characters}. Increase "
                + "chunk_max_characters or shorten the surrounding transcript segments."
            )

        current.append(candidate)
        current_size += separator_size + len(context)

    if current:
        batches.append(current)
    return batches


def _overlap_ratio(first: ClipCandidate, second: ClipCandidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    return overlap / min(first.duration, second.duration)


def rank_candidates(
    candidates: Sequence[ClipCandidate],
    target_duration: float,
) -> list[ClipCandidate]:
    """Prefer complete clips nearest the target, then use editorial score."""
    return sorted(
        candidates,
        key=lambda candidate: (
            abs(target_duration - candidate.duration),
            -candidate.score,
            candidate.start,
        ),
    )


def deduplicate_candidates(
    candidates: Sequence[ClipCandidate],
    threshold: float = 0.6,
    target_duration: float | None = None,
) -> list[ClipCandidate]:
    retained: list[ClipCandidate] = []
    ordered = (
        rank_candidates(candidates, target_duration)
        if target_duration is not None
        else sorted(candidates, key=lambda item: item.score, reverse=True)
    )
    for candidate in ordered:
        if all(_overlap_ratio(candidate, existing) < threshold for existing in retained):
            retained.append(candidate)
    return retained


def _first_response_choice(response: Any) -> Any:
    try:
        choices = response.get("choices") if isinstance(response, dict) else response.choices
        if not choices:
            raise AnalysisError("LiteLLM returned no completion choices.")
        return choices[0]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise AnalysisError("LiteLLM returned an unexpected response shape.") from exc


def _response_message(response: Any) -> Any:
    choice = _first_response_choice(response)
    message = choice.get("message") if isinstance(choice, dict) else choice.message
    if message is None:
        raise AnalysisError("LiteLLM returned no response message.")
    return message


def _response_finish_reason(response: Any) -> str | None:
    choice = _first_response_choice(response)
    value = choice.get("finish_reason") if isinstance(choice, dict) else choice.finish_reason
    return str(value) if value else None


def _response_content(response: Any) -> str:
    message = _response_message(response)
    content = message.get("content") if isinstance(message, dict) else message.content

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            elif not isinstance(part, dict):
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    text_parts.append(part_text)
        if text_parts:
            return "\n".join(text_parts)
    raise AnalysisError("LiteLLM returned an empty text response.")


def _response_tool_payload(response: Any) -> dict[str, Any] | list[Any] | None:
    message = _response_message(response)
    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict) else message.tool_calls
    )
    if not tool_calls:
        return None

    for tool_call in tool_calls:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else tool_call.function
        )
        if not function:
            continue
        name = function.get("name") if isinstance(function, dict) else function.name
        if name != _CLIP_TOOL_NAME:
            continue
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else function.arguments
        )
        if isinstance(arguments, (dict, list)):
            return arguments
        if isinstance(arguments, str):
            return extract_json_payload(arguments)
        raise AnalysisError("The clip-selection tool returned invalid arguments.")
    return None


def _clip_selection_tool(
    max_candidates: int,
    item_schema: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _CLIP_TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "clips": {
                        "type": "array",
                        "maxItems": max_candidates,
                        "items": item_schema,
                    }
                },
                "required": ["clips"],
                "additionalProperties": False,
            },
        },
    }


def _model_request_options(
    model: str,
    requested_count: int,
    item_schema: dict[str, Any],
    tool_description: str,
) -> dict[str, Any]:
    if model not in _STRUCTURED_TOOL_MODELS:
        return {}

    extra_body: dict[str, Any] | None = None
    tool_choice: dict[str, Any] = {
        "type": "function",
        "function": {"name": _CLIP_TOOL_NAME},
    }
    if model == _NVIDIA_NIM_NEMOTRON_ULTRA_MODEL:
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "medium_effort": True,
                "force_nonempty_content": True,
            }
        }
    elif model == _OPENROUTER_NEMOTRON_ULTRA_MODEL:
        extra_body = {
            "reasoning": {
                "max_tokens": _NEMOTRON_REASONING_TOKENS,
                "exclude": True,
            }
        }
    elif model == _ANTHROPIC_DEFAULT_MODEL:
        tool_choice = {"type": "tool", "name": _CLIP_TOOL_NAME}

    request_options: dict[str, Any] = {
        "tools": [
            _clip_selection_tool(
                requested_count,
                item_schema,
                tool_description,
            )
        ],
        "tool_choice": tool_choice,
    }
    if extra_body is not None:
        request_options["extra_body"] = extra_body
    return request_options


class TranscriptAnalyzer:
    def __init__(
        self,
        completion_fn: CompletionFunction = completion,
        *,
        api_key: str | None = None,
        provider: LLMProvider | str = LLMProvider.NVIDIA,
        codex_binary: str = "codex",
        codex_timeout_seconds: int = 300,
        codex_client: CodexCLIClient | None = None,
    ) -> None:
        self._completion = completion_fn
        self._api_key = api_key.strip() if api_key else None
        self._provider = normalize_llm_provider(provider)
        self._codex_client = (
            codex_client
            if self._provider == LLMProvider.CODEX and codex_client is not None
            else (
                CodexCLIClient(
                    codex_binary,
                    timeout_seconds=codex_timeout_seconds,
                )
                if self._provider == LLMProvider.CODEX
                else None
            )
        )

    def analysis_backend_id(self, model: str) -> str:
        if self._provider != LLMProvider.CODEX:
            return model
        if self._codex_client is None:
            raise AnalysisError("The Codex CLI analysis client is unavailable.")
        try:
            return self._codex_client.cache_identity(model)
        except CodexCLIError as exc:
            raise AnalysisError(f"Could not initialize Codex CLI analysis: {exc}") from exc

    def find_clips(
        self,
        transcript: TranscriptDocument,
        model: str,
        clip_count: int | None,
        min_duration: float,
        max_duration: float,
        *,
        cache_dir: Path | None = None,
        force: bool = False,
        chunk_max_characters: int = 45_000,
        chunk_overlap_seconds: float = 60.0,
        max_concurrency: int = 2,
        request_max_attempts: int = 3,
        progress_callback: AnalysisChunkProgressCallback | None = None,
    ) -> list[ClipCandidate]:
        if max_duration > MAX_CLIP_DURATION_SECONDS:
            raise AnalysisError(
                f"Clip duration cannot exceed {MAX_CLIP_DURATION_SECONDS} seconds."
            )
        if clip_count is not None and not 1 <= clip_count <= MAX_CLIP_COUNT:
            raise AnalysisError(
                f"Clip count must be between 1 and {MAX_CLIP_COUNT}, or None for auto."
            )
        if chunk_overlap_seconds < max_duration:
            raise AnalysisError(
                "Transcript chunk overlap cannot be shorter than the maximum clip duration."
            )

        chunks = chunk_transcript(
            transcript.segments,
            max_characters=chunk_max_characters,
            overlap_seconds=chunk_overlap_seconds,
        )
        if not chunks:
            raise AnalysisError("The transcript is empty.")

        per_chunk = (
            MAX_CLIP_COUNT
            if clip_count is None
            else min(max(clip_count * 2, 4), MAX_CLIP_COUNT)
        )
        chunk_results = self._analyze_chunks(
            chunks=chunks,
            video_duration=transcript.duration_seconds,
            model=model,
            requested_count=per_chunk,
            min_duration=min_duration,
            max_duration=max_duration,
            cache_dir=cache_dir,
            force=force,
            max_concurrency=max_concurrency,
            request_max_attempts=request_max_attempts,
            progress_callback=progress_callback,
            analysis_backend=self.analysis_backend_id(model),
        )

        if all(not candidates for candidates in chunk_results):
            raise AnalysisError("No transcript chunk produced a usable clip candidate.")

        review_pool_limit = (
            MAX_CLIP_COUNT
            if clip_count is None
            else min(max(clip_count * 2, clip_count), MAX_CLIP_COUNT)
        )
        review_pool = select_diverse_review_pool(
            chunk_results,
            limit=review_pool_limit,
            target_duration=max_duration,
        )
        if not review_pool:
            raise AnalysisError("No distinct clip candidates survived validation.")

        review_batches = _batch_review_candidates(
            review_pool,
            transcript.segments,
            max_characters=chunk_max_characters,
        )
        verified: list[ClipCandidate] = []
        for batch in review_batches:
            verified.extend(
                self._verify_standalone_candidates(
                    candidates=batch,
                    transcript=transcript,
                    model=model,
                    requested_count=len(batch),
                    min_duration=min_duration,
                    max_duration=max_duration,
                    request_max_attempts=request_max_attempts,
                    allow_empty=True,
                )
            )

        if not verified:
            raise AnalysisError(
                "The final continuity review approved no clips across all batches."
            )

        final_candidates = deduplicate_candidates(
            verified,
            target_duration=max_duration,
        )
        if clip_count is None:
            return final_candidates
        return final_candidates[:clip_count]

    def _analyze_chunks(
        self,
        *,
        chunks: Sequence[Sequence[TranscriptSegment]],
        video_duration: float,
        model: str,
        requested_count: int,
        min_duration: float,
        max_duration: float,
        cache_dir: Path | None,
        force: bool,
        max_concurrency: int,
        request_max_attempts: int,
        progress_callback: AnalysisChunkProgressCallback | None,
        analysis_backend: str | None = None,
    ) -> list[list[ClipCandidate]]:
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        results: list[list[ClipCandidate] | None] = [None] * len(chunks)

        def analyze(index: int) -> tuple[int, list[ClipCandidate]]:
            return (
                index,
                self._analyze_or_load_chunk(
                    index=index,
                    chunk=chunks[index],
                    video_duration=video_duration,
                    model=model,
                    requested_count=requested_count,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    cache_dir=cache_dir,
                    force=force,
                    request_max_attempts=request_max_attempts,
                    analysis_backend=analysis_backend or model,
                ),
            )

        completed = 0
        worker_count = max(1, min(max_concurrency, len(chunks)))
        if worker_count == 1:
            for index in range(len(chunks)):
                result_index, candidates = analyze(index)
                results[result_index] = candidates
                completed += 1
                self._notify_chunk_progress(progress_callback, completed, len(chunks))
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="clip-analysis",
            ) as executor:
                next_index = 0
                pending = set()
                while next_index < worker_count:
                    pending.add(executor.submit(analyze, next_index))
                    next_index += 1

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result_index, candidates = future.result()
                        results[result_index] = candidates
                        completed += 1
                        self._notify_chunk_progress(
                            progress_callback,
                            completed,
                            len(chunks),
                        )

                    while next_index < len(chunks) and len(pending) < worker_count:
                        pending.add(executor.submit(analyze, next_index))
                        next_index += 1

        if any(result is None for result in results):
            raise AnalysisError("One or more transcript chunks were not analyzed.")

        if cache_dir:
            retained_names = {
                f"chunk-{index + 1:03d}.json" for index in range(len(chunks))
            }
            for path in cache_dir.glob("chunk-*.json"):
                if path.name not in retained_names:
                    path.unlink(missing_ok=True)

        return [result for result in results if result is not None]

    @staticmethod
    def _notify_chunk_progress(
        callback: AnalysisChunkProgressCallback | None,
        current: int,
        total: int,
    ) -> None:
        if not callback:
            return
        try:
            callback(current, total)
        except Exception:
            LOGGER.warning("Analysis chunk progress callback failed", exc_info=True)

    def _analyze_or_load_chunk(
        self,
        *,
        index: int,
        chunk: Sequence[TranscriptSegment],
        video_duration: float,
        model: str,
        requested_count: int,
        min_duration: float,
        max_duration: float,
        cache_dir: Path | None,
        force: bool,
        request_max_attempts: int,
        analysis_backend: str,
    ) -> list[ClipCandidate]:
        fingerprint = _chunk_fingerprint(
            chunk,
            analysis_backend,
            requested_count,
            min_duration,
            max_duration,
            video_duration,
        )
        cache_path = cache_dir / f"chunk-{index + 1:03d}.json" if cache_dir else None
        if cache_path and cache_path.is_file() and not force:
            try:
                cached = ChunkAnalysisDocument.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
                if (
                    cached.schema_version == _CHUNK_ANALYSIS_SCHEMA_VERSION
                    and cached.fingerprint == fingerprint
                ):
                    validated = validate_clip_candidates(
                        [candidate.model_dump() for candidate in cached.candidates],
                        video_duration=video_duration,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        transcript_segments=chunk,
                        allowed_start=chunk[0].start,
                        allowed_end=chunk[-1].end,
                        allow_empty=True,
                    )
                    if len(validated) <= requested_count:
                        return validated
            except (AnalysisError, OSError, ValueError):
                LOGGER.warning("Ignoring invalid chunk analysis cache at %s", cache_path)

        candidates = self._analyze_chunk(
            chunk=chunk,
            video_duration=video_duration,
            model=model,
            requested_count=requested_count,
            min_duration=min_duration,
            max_duration=max_duration,
            request_max_attempts=request_max_attempts,
        )
        if cache_path:
            document = ChunkAnalysisDocument(
                schema_version=_CHUNK_ANALYSIS_SCHEMA_VERSION,
                fingerprint=fingerprint,
                candidates=candidates,
            )
            atomic_write_text(cache_path, document.model_dump_json(indent=2))
        return candidates

    def _request_candidates(
        self,
        *,
        prompt: str,
        system_message: str,
        model: str,
        requested_count: int,
        item_schema: dict[str, Any],
        tool_description: str,
        validator: CandidateValidator,
        temperature: float,
        request_max_attempts: int,
    ) -> list[ClipCandidate]:
        last_error: AnalysisError | None = None
        previous_content: str | None = None
        for attempt in range(2):
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            if attempt:
                if previous_content:
                    messages.append(
                        {"role": "assistant", "content": previous_content}
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do not plan again. Submit corrected data now, using the tool "
                            + "when available or plain JSON otherwise. Follow every schema, "
                            + f"timestamp, and editorial constraint. Previous error: {last_error}"
                        ),
                    }
                )

            finish_reason: str | None = None
            response_content: str | None = None
            try:
                if self._provider == LLMProvider.CODEX:
                    if self._codex_client is None:
                        raise CodexCLIError(
                            "The Codex CLI analysis client is unavailable."
                        )
                    output_schema = _clip_selection_tool(
                        requested_count,
                        item_schema,
                        tool_description,
                    )["function"]["parameters"]
                    codex_result = self._codex_client.request(
                        messages=messages,
                        model=model,
                        output_schema=output_schema,
                        description=tool_description,
                        max_attempts=request_max_attempts,
                    )
                    response_content = codex_result.content
                    return validator(_candidate_items(codex_result.payload))

                request_options: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": 120,
                    "max_tokens": _MAX_ANALYSIS_OUTPUT_TOKENS,
                    "max_retries": max(0, request_max_attempts - 1),
                }
                request_options.update(
                    _model_request_options(
                        model,
                        requested_count,
                        item_schema,
                        tool_description,
                    )
                )
                if self._api_key is not None:
                    request_options["api_key"] = self._api_key
                response = self._completion(**request_options)
                finish_reason = _response_finish_reason(response)
                payload = _response_tool_payload(response)
                if payload is None:
                    response_content = _response_content(response)
                    payload = extract_json_payload(response_content)
                return validator(_candidate_items(payload))
            except AnalysisError as exc:
                previous_content = response_content
                if finish_reason in {"length", "max_tokens"}:
                    last_error = AnalysisError(
                        "The model hit the completion-token limit before returning valid clip data."
                    )
                else:
                    last_error = exc
            except CodexCLIError as exc:
                raise AnalysisError(f"Codex CLI analysis failed: {exc}") from exc
            except Exception as exc:
                request_error = f"LiteLLM request failed for model '{model}': {exc}."
                raise AnalysisError(
                    f"{request_error} Check CLIPPER_LLM_PROVIDER and the active "
                    + "provider's model and credential settings in .env."
                ) from exc

        raise AnalysisError(
            f"The model returned invalid clip data twice: {last_error}"
        )

    def _analyze_chunk(
        self,
        chunk: Sequence[TranscriptSegment],
        video_duration: float,
        model: str,
        requested_count: int,
        min_duration: float,
        max_duration: float,
        request_max_attempts: int,
    ) -> list[ClipCandidate]:
        transcript_text = "\n".join(_format_transcript_line(segment) for segment in chunk)
        prompt = f"""You are the first-pass editor for a timestamped transcript.
Use the submit_clip_candidates tool when available. Otherwise return one compact
JSON object with a "clips" array and no Markdown or commentary.

Find compelling excerpts, but completeness is more important than quantity.
Each candidate must be one coherent thought or topic that a new viewer can fully
understand without anything before or after the clip.

Editorial rules:
- Return every strong candidate you identify in this chunk, up to the
  {requested_count}-candidate safety limit; never add filler.
- Each clip must last between {min_duration:.1f} and {max_duration:.1f} seconds.
- Among naturally complete thoughts, prioritize boundaries that produce a duration
  as close to {max_duration:.1f} seconds as possible.
- Extend only with tightly connected setup, development, and resolution. Never add
  unrelated material merely to make a clip longer.
- Never truncate a longer discussion just to satisfy the duration limit; skip it.
- Start at a displayed transcript segment start where the speaker introduces the
  subject, premise, person, or situation needed to understand the clip.
- Do not start with an unexplained pronoun, answer, continuation, or callback.
- End at a displayed transcript segment end after the conclusion, answer, punchline,
  takeaway, or natural resolution. Never cut a sentence or setup/payoff pair.
- Include only one main thought or tightly connected topic.
- Timestamps must be exact absolute values displayed in this chunk and stay between
  {chunk[0].start:.3f} and {chunk[-1].end:.3f} seconds.
- The hook must reflect the actual opening words, not invented marketing copy.
- Score from 0 to 1, weighting standalone clarity above virality.
- Do not invent dialogue, context, or timestamps.

Every candidate must contain title, start, end, score, hook, and reason. Keep text
fields concise. For plain JSON, begin with {{ and end with }}.

Timestamped transcript:
{transcript_text}
"""

        return self._request_candidates(
            prompt=prompt,
            system_message=(
                "Find complete, independently understandable excerpts. Prefer fewer "
                + "clips over context-dependent or abruptly cut clips. Use the requested "
                + "tool when available; otherwise return strict JSON only."
            ),
            model=model,
            requested_count=requested_count,
            item_schema=ClipCandidate.model_json_schema(),
            tool_description="Submit preliminary complete-thought clip candidates.",
            validator=lambda items: validate_clip_candidates(
                items,
                video_duration=video_duration,
                min_duration=min_duration,
                max_duration=max_duration,
                transcript_segments=chunk,
                allowed_start=chunk[0].start,
                allowed_end=chunk[-1].end,
                allow_empty=True,
            ),
            temperature=0.2,
            request_max_attempts=request_max_attempts,
        )

    def _verify_standalone_candidates(
        self,
        *,
        candidates: Sequence[ClipCandidate],
        transcript: TranscriptDocument,
        model: str,
        requested_count: int,
        min_duration: float,
        max_duration: float,
        request_max_attempts: int,
        allow_empty: bool = False,
    ) -> list[ClipCandidate]:
        review_context = "\n\n".join(
            _format_candidate_review_context(
                candidate,
                index,
                transcript.segments,
            )
            for index, candidate in enumerate(candidates, start=1)
        )
        prompt = f"""You are the final continuity editor for short-form video clips.
Review the proposed clips against the transcript shown before, inside, and after
each boundary. Return only clips that are genuinely standalone. You may adjust
start/end timestamps to exact displayed segment boundaries or reject a proposal.

A clip passes only when all of these are true:
- It covers exactly one complete thought, story beat, explanation, or topic.
- Its opening supplies the subject and setup needed by a viewer with no prior context.
- It does not begin as an unexplained response, continuation, pronoun reference, or
  callback to material outside the clip.
- Its ending contains the answer, conclusion, takeaway, punchline, or natural pause.
- No sentence, setup/payoff pair, or essential explanation continues after the end.
- It lasts between {min_duration:.1f} and {max_duration:.1f} seconds. Among valid
  standalone boundary choices, prefer the one closest to {max_duration:.1f} seconds.
- Include all tightly connected setup, development, and resolution that fit, but
  never add a second topic or unrelated material just to approach the target.
- Reject a thought that needs more than {max_duration:.1f} seconds; never cut it short.
- Start and end exactly match transcript segment boundaries shown below.

For every approved clip, set standalone=true and provide:
- topic: the single topic understood from the clip alone.
- opening_context: why the first spoken lines establish the necessary setup.
- closing_resolution: what resolves or completes the thought at the end.
- title, start, end, score, hook, and reason.

Return every proposal that passes, up to {requested_count} approved clips. Local
ranking will order approved clips and apply an explicit output limit when one was
configured, so do not discard a valid proposal solely to shorten the response.
Fewer clips are correct when proposals fail the standard. Do not invent context or
approve a clip merely because it contains an engaging moment. Use
submit_clip_candidates when available; otherwise return one compact JSON object
with a "clips" array.

PROPOSED CLIPS AND CONTEXT:
{review_context}
"""

        return self._request_candidates(
            prompt=prompt,
            system_message=(
                "Act as a strict continuity editor. Approve only clips that begin with "
                + "sufficient setup and end after a complete resolution. Reject ambiguous "
                + "or abruptly cut excerpts. Return tool data or strict JSON only."
            ),
            model=model,
            requested_count=requested_count,
            item_schema=StandaloneClipReview.model_json_schema(),
            tool_description=(
                "Submit only independently understandable clips that passed final review."
            ),
            validator=lambda items: validate_standalone_reviews(
                items,
                video_duration=transcript.duration_seconds,
                min_duration=min_duration,
                max_duration=max_duration,
                transcript_segments=transcript.segments,
                proposed_candidates=candidates,
                allow_empty=allow_empty,
            ),
            temperature=0.1,
            request_max_attempts=request_max_attempts,
        )
