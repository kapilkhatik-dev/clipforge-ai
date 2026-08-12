from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

import yt_clipper.services.analyzer as analyzer_module
from yt_clipper import ContentType, DEFAULT_ANALYSIS_MODEL, LLMProvider
from yt_clipper.domain.errors import AnalysisError, InsufficientHighlightsError
from yt_clipper.domain.models import (
    ClipCandidate,
    HighlightMoment,
    TranscriptDocument,
    TranscriptOrigin,
    TranscriptSegment,
)
from yt_clipper.services.analyzer import (
    build_highlight_windows,
    select_diverse_highlight_pool,
    validate_highlight_montage,
    validate_highlight_moments,
    validate_highlight_screening_payload,
)
from yt_clipper.services.analyzer import (
    TranscriptAnalyzer,
    chunk_transcript,
    deduplicate_candidates,
    extract_json_payload,
    rank_candidates,
    select_diverse_review_pool,
    validate_clip_candidates,
    validate_standalone_reviews,
)
from yt_clipper.services.codex_cli import CodexCLIResult


class CandidateData(TypedDict):
    title: str
    start: float
    end: float
    score: float
    hook: str
    reason: str


def _candidate(
    start: float, end: float, score: float, title: str = "Clip"
) -> CandidateData:
    return {
        "title": title,
        "start": start,
        "end": end,
        "score": score,
        "hook": "A clear hook",
        "reason": "A complete idea with a payoff",
    }


def _test_transcript() -> TranscriptDocument:
    return TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=60,
        segments=[
            TranscriptSegment(start=0, end=10, text="An earlier unrelated topic."),
            TranscriptSegment(start=10, end=20, text="The speaker introduces the topic."),
            TranscriptSegment(start=20, end=35, text="The thought reaches its conclusion."),
            TranscriptSegment(start=35, end=60, text="A different topic begins."),
        ],
    )


def _reviewed_candidate_json() -> str:
    return (
        '{"clips": [{"title": "Good", "start": 10, "end": 35, '
        '"score": 0.9, "hook": "The speaker introduces the topic", '
        '"reason": "One complete thought with a clear conclusion", '
        '"standalone": true, "topic": "A self-contained topic", '
        '"opening_context": "The opening explicitly introduces the topic", '
        '"closing_resolution": "The final line states the conclusion"}]}'
    )


def _formatted_transcript(segments: list[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{segment.start:.3f} - {segment.end:.3f}] {segment.text}"
        for segment in segments
    )


def _review_batch_fixture() -> tuple[
    TranscriptDocument,
    list[ClipCandidate],
    int,
]:
    transcript = TranscriptDocument(
        video_id="review-batches",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="c" * 64,
        duration_seconds=160,
        segments=[
            TranscriptSegment(
                start=float(index * 20),
                end=float((index + 1) * 20),
                text=f"Context segment {index} states one complete point.",
            )
            for index in range(8)
        ],
    )
    candidates = [
        ClipCandidate(
            **_candidate(
                float(index * 60),
                float(index * 60 + 20),
                0.9 - index * 0.1,
                f"Review {index}",
            )
        )
        for index in range(3)
    ]
    review_limit = max(
        len(
            analyzer_module._format_candidate_review_context(
                candidate,
                1,
                transcript.segments,
            )
        )
        for candidate in candidates
    )
    return transcript, candidates, review_limit


def _standalone_review(candidate: ClipCandidate) -> dict[str, object]:
    return {
        "title": candidate.title,
        "start": candidate.start,
        "end": candidate.end,
        "score": candidate.score,
        "hook": candidate.hook,
        "reason": candidate.reason,
        "standalone": True,
        "topic": f"Topic for {candidate.title}",
        "opening_context": "The opening names the subject.",
        "closing_resolution": "The ending resolves the point.",
    }


def test_one_hour_transcript_chunks_preserve_sixty_second_overlap() -> None:
    segments = [
        TranscriptSegment(
            start=float(index * 10),
            end=float((index + 1) * 10),
            text=(
                f"Segment {index} is a natural ten-second passage where the speaker "
                + "explains one focused idea in clear conversational language with "
                + "enough context to make the point understandable."
            ),
        )
        for index in range(360)
    ]

    chunks = chunk_transcript(
        segments,
        max_characters=8_000,
        overlap_seconds=60,
    )

    assert 8 <= len(chunks) <= 12
    assert chunks[-1][-1].end == 3600
    assert all(len(_formatted_transcript(chunk)) <= 8_000 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-1].end - current[0].start >= 60


def test_chunk_transcript_rejects_a_single_oversized_segment() -> None:
    segment = TranscriptSegment(start=0, end=10, text="x" * 100)

    with pytest.raises(
        AnalysisError,
        match=r"single transcript segment.*max_characters",
    ):
        chunk_transcript([segment], max_characters=50, overlap_seconds=0)


def test_chunk_transcript_rejects_overlap_that_prevents_progress() -> None:
    segments = [
        TranscriptSegment(
            start=float(index * 10),
            end=float((index + 1) * 10),
            text=f"Short segment {index}.",
        )
        for index in range(3)
    ]
    two_segment_limit = len(_formatted_transcript(segments[:2]))

    with pytest.raises(
        AnalysisError,
        match=r"overlap.*leaves no room.*reduce overlap_seconds",
    ):
        chunk_transcript(
            segments,
            max_characters=two_segment_limit,
            overlap_seconds=20,
        )


def test_chunk_transcript_resets_overlap_after_a_long_caption_gap() -> None:
    segments = [
        TranscriptSegment(start=0, end=10, text="Before the long silence."),
        TranscriptSegment(start=100, end=110, text="Speech resumes here."),
        TranscriptSegment(start=110, end=120, text="The resumed thought continues."),
    ]
    two_segment_limit = max(
        len(_formatted_transcript(segments[:2])),
        len(_formatted_transcript(segments[1:])),
    )

    chunks = chunk_transcript(
        segments,
        max_characters=two_segment_limit,
        overlap_seconds=60,
    )

    assert [[segment.start for segment in chunk] for chunk in chunks] == [
        [0.0, 100.0],
        [100.0, 110.0],
    ]
    assert all(
        len(_formatted_transcript(chunk)) <= two_segment_limit for chunk in chunks
    )


def test_chunk_transcript_caps_the_number_of_chunks() -> None:
    segments = [
        TranscriptSegment(
            start=float(index),
            end=float(index + 1),
            text=f"Segment {index}",
        )
        for index in range(65)
    ]
    single_segment_limit = max(
        len(_formatted_transcript([segment])) for segment in segments
    )

    with pytest.raises(AnalysisError, match=r"more than 64 chunks"):
        chunk_transcript(
            segments,
            max_characters=single_segment_limit,
            overlap_seconds=0,
        )


def test_extracts_json_from_markdown_response() -> None:
    payload = extract_json_payload('Result:\n```json\n{"clips": []}\n```')
    assert payload == {"clips": []}


def test_filters_duration_and_video_bounds() -> None:
    segments = [TranscriptSegment(start=0, end=100, text="Transcript")]
    valid = validate_clip_candidates(
        [
            _candidate(10, 40, 0.9, "Valid"),
            _candidate(20, 25, 0.8, "Too short"),
            _candidate(90, 110, 0.7, "Out of bounds"),
        ],
        video_duration=100,
        min_duration=20,
        max_duration=45,
        transcript_segments=segments,
    )
    assert [candidate.title for candidate in valid] == ["Valid"]


def test_first_pass_allows_only_a_genuinely_empty_candidate_array() -> None:
    assert (
        validate_clip_candidates(
            [],
            video_duration=60,
            min_duration=20,
            max_duration=40,
            allow_empty=True,
        )
        == []
    )

    with pytest.raises(AnalysisError, match="no usable clips"):
        validate_clip_candidates(
            [_candidate(0, 5, 0.9, "Invalid")],
            video_duration=60,
            min_duration=20,
            max_duration=40,
            allow_empty=True,
        )


def test_final_review_batch_allows_only_a_genuinely_empty_array() -> None:
    transcript, candidates, _ = _review_batch_fixture()

    assert (
        validate_standalone_reviews(
            [],
            video_duration=transcript.duration_seconds,
            min_duration=20,
            max_duration=20,
            transcript_segments=transcript.segments,
            proposed_candidates=candidates,
            allow_empty=True,
        )
        == []
    )

    with pytest.raises(AnalysisError, match="approved no standalone clips"):
        validate_standalone_reviews(
            [{"title": "Incomplete review"}],
            video_duration=transcript.duration_seconds,
            min_duration=20,
            max_duration=20,
            transcript_segments=transcript.segments,
            proposed_candidates=candidates,
            allow_empty=True,
        )


def test_requires_review_evidence_and_transcript_boundaries() -> None:
    transcript = _test_transcript()
    with pytest.raises(AnalysisError, match="review evidence"):
        _ = validate_clip_candidates(
            [_candidate(10, 35, 0.9, "Unreviewed")],
            video_duration=60,
            min_duration=20,
            max_duration=40,
            transcript_segments=transcript.segments,
            require_segment_boundaries=True,
            require_standalone=True,
        )

    reviewed = {
        **_candidate(11, 35, 0.9, "Misaligned"),
        "standalone": True,
        "topic": "A topic",
        "opening_context": "The opening supplies enough context",
        "closing_resolution": "The ending completes the thought",
    }

    with pytest.raises(AnalysisError, match="transcript boundaries"):
        _ = validate_clip_candidates(
            [reviewed],
            video_duration=60,
            min_duration=20,
            max_duration=40,
            transcript_segments=transcript.segments,
            require_segment_boundaries=True,
            require_standalone=True,
        )

    reviewed["start"] = 10.2
    valid = validate_clip_candidates(
        [reviewed],
        video_duration=60,
        min_duration=20,
        max_duration=40,
        transcript_segments=transcript.segments,
        require_segment_boundaries=True,
        require_standalone=True,
    )
    assert valid[0].start == 10
    assert valid[0].standalone is True


def test_duration_ranking_prefers_clips_closest_to_sixty_seconds() -> None:
    candidates = [
        ClipCandidate(**_candidate(0, 40, 0.95, "Shorter")),
        ClipCandidate(**_candidate(70, 128, 0.8, "Near target")),
        ClipCandidate(**_candidate(140, 190, 0.9, "Middle")),
    ]

    ranked = rank_candidates(candidates, target_duration=60)

    assert [candidate.title for candidate in ranked] == [
        "Near target",
        "Middle",
        "Shorter",
    ]


def test_duration_priority_keeps_longer_overlapping_candidate() -> None:
    candidates = [
        ClipCandidate(**_candidate(0, 58, 0.8, "Complete long clip")),
        ClipCandidate(**_candidate(10, 40, 0.99, "Short excerpt")),
    ]

    retained = deduplicate_candidates(candidates, target_duration=60)

    assert [candidate.title for candidate in retained] == ["Complete long clip"]


def test_review_pool_stratifies_a_limited_prefix_across_the_timeline() -> None:
    chunk_candidates = []
    for index in range(8):
        start = float(index * 100)
        chunk_candidates.append(
            [
                ClipCandidate(
                    **_candidate(start + 5, start + 25, 0.99, f"Short {index}")
                ),
                ClipCandidate(
                    **_candidate(start, start + 58, 0.8, f"Complete {index}")
                ),
            ]
        )

    selected = select_diverse_review_pool(
        chunk_candidates,
        limit=2,
        target_duration=60,
    )

    assert [candidate.title for candidate in selected] == [
        "Complete 0",
        "Complete 7",
    ]


def test_review_pool_stratifies_only_across_nonempty_chunks() -> None:
    chunk_candidates: list[list[ClipCandidate]] = [
        [],
        [ClipCandidate(**_candidate(100, 158, 0.9, "First available"))],
        [],
        [],
        [ClipCandidate(**_candidate(400, 458, 0.9, "Last available"))],
        [],
    ]

    selected = select_diverse_review_pool(
        chunk_candidates,
        limit=2,
        target_duration=60,
    )

    assert [candidate.title for candidate in selected] == [
        "First available",
        "Last available",
    ]


def test_deduplication_keeps_higher_scoring_overlap() -> None:
    candidates = [
        ClipCandidate(**_candidate(10, 40, 0.7, "Lower")),
        ClipCandidate(**_candidate(12, 42, 0.95, "Higher")),
        ClipCandidate(**_candidate(60, 90, 0.8, "Separate")),
    ]
    retained = deduplicate_candidates(candidates)
    assert [candidate.title for candidate in retained] == ["Higher", "Separate"]


def test_analyzer_retries_invalid_json_once() -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": "not json"}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"clips": [{"title": "Good", "start": 10, "end": 35, "score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"message": {"content": _reviewed_candidate_json()}}
                ]
            },
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    result = TranscriptAnalyzer(completion_fn=fake_completion).find_clips(
        _test_transcript(),
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=40,
    )

    assert len(calls) == 3
    assert calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": "not json",
    }
    assert "BEFORE [0.000 - 10.000]" in calls[2]["messages"][-1]["content"]
    assert "AFTER [35.000 - 60.000]" in calls[2]["messages"][-1]["content"]
    assert result[0].title == "Good"
    assert result[0].standalone is True


def test_builds_four_second_transcript_backed_highlight_windows() -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=10,
        segments=[
            TranscriptSegment(start=0, end=3, text="First moment"),
            TranscriptSegment(start=4, end=8, text="Second moment"),
            TranscriptSegment(start=8, end=10, text="Final moment"),
        ],
    )

    windows = build_highlight_windows(transcript, window_seconds=4)

    assert [(item["start"], item["end"]) for item in windows] == [
        (0.0, 4.0),
        (4.0, 8.0),
        (8.0, 10),
    ]
    assert windows[1]["text"] == "Second moment"


def test_highlight_windows_slice_long_untimed_caption_text() -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=8,
        segments=[
            TranscriptSegment(
                start=0,
                end=8,
                text="one two three four five six seven eight",
            )
        ],
    )

    windows = build_highlight_windows(transcript, window_seconds=4)

    assert windows[0]["text"] == "one two three four"
    assert windows[1]["text"] == "five six seven eight"


def test_highlight_validation_rejects_invented_windows_and_bounds_montage() -> None:
    windows = [
        {"start": 0.0, "end": 4.0},
        {"start": 8.0, "end": 12.0},
        {"start": 20.0, "end": 24.0},
    ]
    moments = validate_highlight_moments(
        [
            {"start": 8, "end": 12, "score": 0.9, "hook": "B", "reason": "B"},
            {"start": 1, "end": 5, "score": 1, "hook": "Bad", "reason": "Bad"},
            {"start": 0, "end": 4, "score": 0.8, "hook": "A", "reason": "A"},
            {"start": 20, "end": 24, "score": 0.7, "hook": "C", "reason": "C"},
        ],
        allowed_windows=windows,
    )

    montage = validate_highlight_montage(
        {
            "title": "Best moments",
            "summary": "A varied highlight reel.",
            "moments": [moment.model_dump() for moment in moments],
        },
        proposed_moments=moments,
        max_duration=8,
        max_moments=2,
    )

    assert [(item.start, item.end) for item in montage.moments] == [(8, 12), (0, 4)]
    assert montage.duration == 8


def test_highlight_screening_payload_requires_an_explicit_moments_array() -> None:
    windows = [{"start": 0.0, "end": 4.0}]

    assert validate_highlight_screening_payload(
        {"moments": []},
        allowed_windows=windows,
    ) == []
    with pytest.raises(AnalysisError, match="only a 'moments' array"):
        validate_highlight_screening_payload({}, allowed_windows=windows)
    with pytest.raises(AnalysisError, match="must be an array"):
        validate_highlight_screening_payload(
            {"moments": "none"},  # type: ignore[dict-item]
            allowed_windows=windows,
        )


def test_highlight_review_pool_preserves_quality_across_the_timeline() -> None:
    batches = [
        [
            HighlightMoment(
                start=batch * 40 + offset * 4,
                end=batch * 40 + offset * 4 + 4,
                score=0.95 if offset == 1 else 0.8,
                hook=f"Batch {batch} moment {offset}",
                reason="Strong moment",
            )
            for offset in range(3)
        ]
        for batch in range(8)
    ]

    selected = select_diverse_highlight_pool(batches, limit=4)

    assert len(selected) == 4
    assert all(moment.score == 0.95 for moment in selected)
    assert sorted(moment.start for moment in selected) == [4, 84, 164, 244]


def test_highlight_montage_uses_batched_screening_and_global_review() -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=16,
        segments=[
            TranscriptSegment(start=index, end=index + 1, text=f"Moment {index}")
            for index in range(16)
        ],
    )
    responses = iter(
        [
            {
                "choices": [{"message": {"content": json.dumps({"moments": [
                    {"start": 0, "end": 4, "score": 0.8, "hook": "A", "reason": "A"},
                    {"start": 4, "end": 8, "score": 0.9, "hook": "B", "reason": "B"},
                ]})}}]
            },
            {
                "choices": [{"message": {"content": json.dumps({"moments": [
                    {"start": 8, "end": 12, "score": 0.95, "hook": "C", "reason": "C"},
                    {"start": 12, "end": 16, "score": 0.7, "hook": "D", "reason": "D"},
                ]})}}]
            },
            {
                "choices": [{"message": {"content": json.dumps({
                    "title": "The Best Bits",
                    "summary": "All the strongest moments.",
                    "moments": [
                        {"start": 8, "end": 12, "score": 0.95, "hook": "C", "reason": "C"},
                        {"start": 4, "end": 8, "score": 0.9, "hook": "B", "reason": "B"},
                    ],
                })}}]
            },
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(responses)

    montage = TranscriptAnalyzer(completion_fn=fake_completion).find_highlight_montage(
        transcript,
        model="test/model",
        window_seconds=4,
        max_duration=12,
        max_moments=3,
        batch_windows=2,
    )

    assert len(calls) == 3
    assert montage.title == "The Best Bits"
    assert [(item.start, item.end) for item in montage.moments] == [(8, 12), (4, 8)]
    first_messages = calls[0]["messages"]
    second_messages = calls[1]["messages"]
    assert isinstance(first_messages, list)
    assert isinstance(second_messages, list)
    first_prompt = first_messages[1]["content"]
    second_prompt = second_messages[1]["content"]
    assert '"selectable_window": {"window_id": 1' in first_prompt
    assert '"context_before": null' in first_prompt
    assert '"context_after": "Moment 4 Moment 5 Moment 6 Moment 7"' in first_prompt
    assert '"context_before": "Moment 4 Moment 5 Moment 6 Moment 7"' in second_prompt


@pytest.mark.parametrize(
    "bad_payload",
    [{}, {"moments": "not-an-array"}, []],
    ids=["missing-moments", "non-list-moments", "non-object-payload"],
)
def test_highlight_screening_retries_malformed_payload_and_accepts_empty_array(
    bad_payload: object,
) -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=16,
        segments=[
            TranscriptSegment(start=index, end=index + 1, text=f"Moment {index}")
            for index in range(16)
        ],
    )
    moments = [
        {"start": 8, "end": 12, "score": 0.9, "hook": "C", "reason": "C"},
        {"start": 12, "end": 16, "score": 0.8, "hook": "D", "reason": "D"},
    ]
    responses = iter(
        [
            {"choices": [{"message": {"content": json.dumps(bad_payload)}}]},
            {"choices": [{"message": {"content": '{"moments": []}'}}]},
            {
                "choices": [
                    {"message": {"content": json.dumps({"moments": moments})}}
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "title": "Best moments",
                                    "summary": "Two strong moments.",
                                    "moments": moments,
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(responses)

    montage = TranscriptAnalyzer(completion_fn=fake_completion).find_highlight_montage(
        transcript,
        model="test/model",
        batch_windows=2,
    )

    assert montage.title == "Best moments"
    assert len(calls) == 4
    retry_messages = calls[1]["messages"]
    assert isinstance(retry_messages, list)
    assert "Previous error:" in retry_messages[-1]["content"]


@pytest.mark.parametrize(
    ("model", "provider", "expected_extra_body"),
    [
        (
            DEFAULT_ANALYSIS_MODEL,
            LLMProvider.NVIDIA,
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "medium_effort": True,
                    "force_nonempty_content": True,
                }
            },
        ),
        (
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            LLMProvider.OPENROUTER,
            {"reasoning": {"max_tokens": 2_000, "exclude": True}},
        ),
        ("openai/gpt-4.1-mini", LLMProvider.OPENAI, None),
        ("anthropic/claude-sonnet-4-6", LLMProvider.ANTHROPIC, None),
    ],
    ids=["nvidia-nim", "openrouter", "openai", "anthropic"],
)
def test_highlight_montage_uses_provider_structured_tool_contracts(
    model: str,
    provider: LLMProvider,
    expected_extra_body: dict[str, object] | None,
) -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=8,
        segments=[TranscriptSegment(start=0, end=8, text="Two excellent moments")],
    )
    moments = [
        {"start": 0, "end": 4, "score": 0.9, "hook": "A", "reason": "A"},
        {"start": 4, "end": 8, "score": 0.8, "hook": "B", "reason": "B"},
    ]
    payloads = [
        {"moments": moments},
        {
            "title": "Best moments",
            "summary": "Two strong moments.",
            "moments": moments,
        },
    ]
    tool_names = ["submit_highlight_moments", "submit_highlight_montage"]
    calls: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> object:
        index = len(calls)
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tool_names[index],
                                    "arguments": json.dumps(payloads[index]),
                                }
                            }
                        ],
                    },
                }
            ]
        }

    montage = TranscriptAnalyzer(
        completion_fn=fake_completion,
        api_key="shared-test-key",
        provider=provider,
    ).find_highlight_montage(transcript, model=model)

    assert montage.title == "Best moments"
    assert len(calls) == 2
    tools: list[list[dict[str, Any]]] = []
    for call in calls:
        tool = call["tools"]
        assert isinstance(tool, list)
        tools.append(cast(list[dict[str, Any]], tool))
    assert [tool[0]["function"]["name"] for tool in tools] == tool_names
    assert "moments" in tools[0][0]["function"]["parameters"]["properties"]
    assert "title" in tools[1][0]["function"]["parameters"]["properties"]
    if expected_extra_body is None:
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == expected_extra_body
    if provider == LLMProvider.ANTHROPIC:
        assert calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit_highlight_moments",
        }
    else:
        tool_choice = calls[0]["tool_choice"]
        assert isinstance(tool_choice, dict)
        assert tool_choice["function"]["name"] == "submit_highlight_moments"


def test_terminal_insufficient_final_selection_preserves_exception_type() -> None:
    transcript = TranscriptDocument(
        video_id="video",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="a" * 64,
        duration_seconds=8,
        segments=[TranscriptSegment(start=0, end=8, text="Two moments")],
    )
    first = {"start": 0, "end": 4, "score": 0.9, "hook": "A", "reason": "A"}
    second = {"start": 4, "end": 8, "score": 0.8, "hook": "B", "reason": "B"}
    invalid_montage = {
        "title": "Duplicate",
        "summary": "The same moment twice.",
        "moments": [first, first],
    }
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"moments": [first, second]})
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": json.dumps(invalid_montage)}}]},
            {"choices": [{"message": {"content": json.dumps(invalid_montage)}}]},
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_completion(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(responses)

    with pytest.raises(InsufficientHighlightsError, match="at least two"):
        TranscriptAnalyzer(completion_fn=fake_completion).find_highlight_montage(
            transcript,
            model="test/model",
        )

    assert len(calls) == 3


def test_comedy_content_type_guides_both_existing_analysis_stages() -> None:
    generation = (
        '{"clips": [{"title": "Good", "start": 10, "end": 35, '
        '"score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
    )
    responses = iter(
        [
            {"choices": [{"message": {"content": generation}}]},
            {"choices": [{"message": {"content": _reviewed_candidate_json()}}]},
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    result = TranscriptAnalyzer(completion_fn=fake_completion).find_clips(
        _test_transcript(),
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=40,
        content_type=ContentType.COMEDY,
    )

    assert result[0].title == "Good"
    assert len(calls) == 2
    for call in calls:
        system_message = call["messages"][0]["content"]
        prompt = call["messages"][1]["content"]
        assert "configured content type is 'comedy'" in system_message
        assert "explicitly categorized as comedy" in prompt
        assert "setup, escalation, punchline" in prompt
        assert "isolated laughter or reactions" in prompt


def test_auto_content_type_infers_genre_without_a_classification_call() -> None:
    generation = (
        '{"clips": [{"title": "Good", "start": 10, "end": 35, '
        '"score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
    )
    responses = iter(
        [
            {"choices": [{"message": {"content": generation}}]},
            {"choices": [{"message": {"content": _reviewed_candidate_json()}}]},
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    _ = TranscriptAnalyzer(completion_fn=fake_completion).find_clips(
        _test_transcript(),
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=40,
    )

    assert len(calls) == 2
    assert all(
        "Infer the dominant content type" in call["messages"][1]["content"]
        for call in calls
    )


def test_codex_provider_uses_structured_cli_results_without_litellm() -> None:
    generation = {
        "clips": [
            {
                "title": "Good",
                "start": 10,
                "end": 35,
                "score": 0.9,
                "hook": "The speaker introduces the topic",
                "reason": "A complete idea with a payoff",
            }
        ]
    }
    review = json.loads(_reviewed_candidate_json())
    responses = iter([generation, review])
    calls = []

    class FakeCodexClient:
        def cache_identity(self, model: str) -> str:
            assert model == "codex/default"
            return "codex-cli-test:default"

        def request(self, **kwargs):
            calls.append(kwargs)
            payload = next(responses)
            return CodexCLIResult(
                payload=payload,
                content=json.dumps(payload),
            )

    def fail_litellm(**_kwargs):
        raise AssertionError("Codex provider must not call LiteLLM")

    result = TranscriptAnalyzer(
        completion_fn=fail_litellm,
        provider=LLMProvider.CODEX,
        codex_client=FakeCodexClient(),  # pyright: ignore[reportArgumentType]
    ).find_clips(
        _test_transcript(),
        model="codex/default",
        clip_count=1,
        min_duration=20,
        max_duration=40,
    )

    assert len(calls) == 2
    assert calls[0]["model"] == "codex/default"
    assert calls[0]["max_attempts"] == 3
    assert calls[0]["output_schema"]["required"] == ["clips"]
    assert calls[0]["output_schema"]["properties"]["clips"]["maxItems"] == 4
    assert result[0].title == "Good"
    assert result[0].standalone is True


def test_final_review_uses_multiple_bounded_batches(monkeypatch) -> None:
    transcript, candidates, review_limit = _review_batch_fixture()
    review_prompts = []

    def fake_completion(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        review_prompts.append(prompt)
        reviewed = [
            _standalone_review(candidate)
            for candidate in candidates
            if f'"title":"{candidate.title}"' in prompt
        ]
        return {
            "choices": [
                {"message": {"content": json.dumps({"clips": reviewed})}}
            ]
        }

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    monkeypatch.setattr(
        analyzer,
        "_analyze_chunks",
        lambda **kwargs: [candidates],
    )

    result = analyzer.find_clips(
        transcript,
        model="test/model",
        clip_count=2,
        min_duration=20,
        max_duration=20,
        chunk_max_characters=review_limit,
        chunk_overlap_seconds=20,
    )

    assert len(review_prompts) == len(candidates)
    for prompt in review_prompts:
        marker = "PROPOSED CLIPS AND CONTEXT:\n"
        assert marker in prompt
        review_context = prompt.partition(marker)[2].rstrip("\n")
        assert len(review_context) <= review_limit
        assert review_context.count("CANDIDATE ") == 1
    assert [candidate.title for candidate in result] == ["Review 0", "Review 1"]


def test_automatic_clip_count_returns_every_approved_candidate(monkeypatch) -> None:
    transcript, candidates, _ = _review_batch_fixture()
    analysis_settings = {}

    def fake_completion(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        reviewed = [
            _standalone_review(candidate)
            for candidate in candidates
            if f'"title":"{candidate.title}"' in prompt
        ]
        return {
            "choices": [
                {"message": {"content": json.dumps({"clips": reviewed})}}
            ]
        }

    def fake_analyze_chunks(**kwargs):
        analysis_settings.update(kwargs)
        return [candidates]

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    monkeypatch.setattr(analyzer, "_analyze_chunks", fake_analyze_chunks)

    result = analyzer.find_clips(
        transcript,
        model="test/model",
        clip_count=None,
        min_duration=20,
        max_duration=20,
        chunk_overlap_seconds=20,
    )

    assert analysis_settings["requested_count"] == 20
    assert [candidate.title for candidate in result] == [
        "Review 0",
        "Review 1",
        "Review 2",
    ]


def test_final_review_rejects_a_single_oversized_context(monkeypatch) -> None:
    transcript, candidates, _ = _review_batch_fixture()
    candidate = candidates[0]
    context_size = len(
        analyzer_module._format_candidate_review_context(
            candidate,
            1,
            transcript.segments,
        )
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"clips": []}'}}]}

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    monkeypatch.setattr(
        analyzer,
        "_analyze_chunks",
        lambda **kwargs: [[candidate]],
    )

    with pytest.raises(
        AnalysisError,
        match=r"single final-review candidate context.*chunk_max_characters",
    ):
        analyzer.find_clips(
            transcript,
            model="test/model",
            clip_count=1,
            min_duration=20,
            max_duration=20,
            chunk_max_characters=context_size - 1,
            chunk_overlap_seconds=20,
        )

    assert calls == []


def test_final_review_fails_once_after_all_batches_return_empty(monkeypatch) -> None:
    transcript, candidates, review_limit = _review_batch_fixture()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"clips": []}'}}]}

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    monkeypatch.setattr(
        analyzer,
        "_analyze_chunks",
        lambda **kwargs: [candidates],
    )

    with pytest.raises(
        AnalysisError,
        match="approved no clips across all batches",
    ):
        analyzer.find_clips(
            transcript,
            model="test/model",
            clip_count=2,
            min_duration=20,
            max_duration=20,
            chunk_max_characters=review_limit,
            chunk_overlap_seconds=20,
        )

    assert len(calls) == len(candidates)


def test_bounds_parallel_chunk_analysis_and_preserves_order(monkeypatch) -> None:
    chunks = [
        [
            TranscriptSegment(
                start=float(index * 30),
                end=float((index + 1) * 30),
                text=f"Complete topic {index}",
            )
        ]
        for index in range(3)
    ]
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    maximum_in_flight = 0
    in_flight = set()

    class TrackingThreadPoolExecutor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            nonlocal maximum_in_flight
            future = super().submit(fn, *args, **kwargs)
            with lock:
                completed_futures = {
                    pending for pending in in_flight if pending.done()
                }
                in_flight.difference_update(completed_futures)
                if not future.done():
                    in_flight.add(future)
                maximum_in_flight = max(maximum_in_flight, len(in_flight))
            return future

    monkeypatch.setattr(
        analyzer_module,
        "ThreadPoolExecutor",
        TrackingThreadPoolExecutor,
    )

    def fake_completion(**kwargs):
        nonlocal active, maximum_active
        prompt = kwargs["messages"][1]["content"]
        match = re.search(
            r"stay between\s+([0-9.]+)\s+and\s+([0-9.]+)",
            prompt,
        )
        assert match
        start = float(match.group(1))
        end = float(match.group(2))
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.06 if start == 0 else 0.01)
        with lock:
            active -= 1
        content = (
            '{"clips": [{"title": "Chunk '
            + str(int(start // 30))
            + '", "start": '
            + str(start)
            + ', "end": '
            + str(end)
            + ', "score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
        )
        return {"choices": [{"message": {"content": content}}]}

    progress = []
    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    results = analyzer._analyze_chunks(
        chunks=chunks,
        video_duration=90,
        model="test/model",
        requested_count=1,
        min_duration=20,
        max_duration=40,
        cache_dir=None,
        force=False,
        max_concurrency=2,
        request_max_attempts=1,
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert maximum_active == 2
    assert maximum_in_flight == 2
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [candidates[0].title for candidates in results] == [
        "Chunk 0",
        "Chunk 1",
        "Chunk 2",
    ]


def test_empty_first_pass_chunk_is_cached_and_aggregation_continues(
    tmp_path: Path,
) -> None:
    transcript = TranscriptDocument(
        video_id="multi-chunk",
        language="en",
        requested_language="en",
        origin=TranscriptOrigin.MANUAL,
        source_fingerprint="b" * 64,
        duration_seconds=400,
        segments=[
            TranscriptSegment(
                start=float(index * 20),
                end=float((index + 1) * 20),
                text=f"Transcript segment {index} develops the discussion.",
            )
            for index in range(20)
        ],
    )
    chunk_limit = 500
    chunks = chunk_transcript(
        transcript.segments,
        max_characters=chunk_limit,
        overlap_seconds=20,
    )
    generation = (
        '{"clips": [{"title": "Final point", "start": 380, "end": 400, '
        '"score": 0.9, "hook": "A complete final point", '
        '"reason": "The final point is fully resolved"}]}'
    )
    review = (
        '{"clips": [{"title": "Final point", "start": 380, "end": 400, '
        '"score": 0.9, "hook": "A complete final point", '
        '"reason": "The final point is fully resolved", "standalone": true, '
        '"topic": "The final point", "opening_context": "The subject is explicit", '
        '"closing_resolution": "The point is resolved"}]}'
    )
    responses = iter(
        [
            *[
                {"choices": [{"message": {"content": '{"clips": []}'}}]}
                for _ in chunks[:-1]
            ],
            {"choices": [{"message": {"content": generation}}]},
            {"choices": [{"message": {"content": review}}]},
            {"choices": [{"message": {"content": review}}]},
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    cache_dir = tmp_path / "empty-chunk-cache"
    first = analyzer.find_clips(
        transcript=transcript,
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=20,
        cache_dir=cache_dir,
        chunk_max_characters=chunk_limit,
        chunk_overlap_seconds=20,
        max_concurrency=1,
    )
    second = analyzer.find_clips(
        transcript=transcript,
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=20,
        cache_dir=cache_dir,
        chunk_max_characters=chunk_limit,
        chunk_overlap_seconds=20,
        max_concurrency=1,
    )

    assert first == second
    assert [candidate.title for candidate in first] == ["Final point"]
    assert len(calls) == len(chunks) + 2
    assert len(chunks) > 1
    for index in range(len(chunks)):
        assert (cache_dir / f"chunk-{index + 1:03d}.json").is_file()


def test_reuses_successful_chunk_analysis_cache(tmp_path: Path) -> None:
    generation = (
        '{"clips": [{"title": "Good", "start": 10, "end": 35, '
        '"score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
    )
    responses = iter(
        [
            {"choices": [{"message": {"content": generation}}]},
            {"choices": [{"message": {"content": _reviewed_candidate_json()}}]},
            {"choices": [{"message": {"content": _reviewed_candidate_json()}}]},
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    analyzer = TranscriptAnalyzer(completion_fn=fake_completion)
    cache_dir = tmp_path / "chunks"

    first = analyzer.find_clips(
        transcript=_test_transcript(),
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=40,
        cache_dir=cache_dir,
    )
    second = analyzer.find_clips(
        transcript=_test_transcript(),
        model="test/model",
        clip_count=1,
        min_duration=20,
        max_duration=40,
        cache_dir=cache_dir,
    )

    assert first == second
    assert len(calls) == 3
    assert (tmp_path / "chunks" / "chunk-001.json").is_file()


@pytest.mark.parametrize(
    ("model", "expected_extra_body"),
    [
        (
            DEFAULT_ANALYSIS_MODEL,
            {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "medium_effort": True,
                    "force_nonempty_content": True,
                }
            },
        ),
        (
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            {"reasoning": {"max_tokens": 2_000, "exclude": True}},
        ),
        ("openai/gpt-4.1-mini", None),
        ("anthropic/claude-sonnet-4-6", None),
    ],
    ids=["nvidia-nim", "openrouter", "openai", "anthropic"],
)
def test_supported_provider_routes_use_typed_tool_calls(
    model: str,
    expected_extra_body: dict[str, object] | None,
) -> None:
    planning_text = "I need to plan the candidates before returning JSON."
    generation_arguments = (
        '{"clips": [{"title": "Good", "start": 10, "end": 35, '
        '"score": 0.9, "hook": "Hook", "reason": "Reason"}]}'
    )
    responses = iter(
        [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": planning_text},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_clip_candidates",
                                        "arguments": generation_arguments,
                                    }
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_clip_candidates",
                                        "arguments": _reviewed_candidate_json(),
                                    }
                                }
                            ],
                        },
                    }
                ]
            },
        ]
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    if model.startswith("openrouter/"):
        provider = LLMProvider.OPENROUTER
    elif model.startswith("openai/"):
        provider = LLMProvider.OPENAI
    elif model.startswith("anthropic/"):
        provider = LLMProvider.ANTHROPIC
    else:
        provider = LLMProvider.NVIDIA

    result = TranscriptAnalyzer(
        completion_fn=fake_completion,
        api_key="shared-test-key",
        provider=provider,
    ).find_clips(
        _test_transcript(),
        model=model,
        clip_count=1,
        min_duration=20,
        max_duration=40,
    )

    assert len(calls) == 3
    assert calls[0]["max_tokens"] == 10_000
    assert all(call["api_key"] == "shared-test-key" for call in calls)
    if expected_extra_body is None:
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == expected_extra_body
    if model.startswith("anthropic/"):
        assert calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit_clip_candidates",
        }
    else:
        assert calls[0]["tool_choice"]["function"]["name"] == (
            "submit_clip_candidates"
        )
    assert calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": planning_text,
    }
    assert "completion-token limit" in calls[1]["messages"][-1]["content"]
    review_schema = calls[2]["tools"][0]["function"]["parameters"]["properties"]["clips"]["items"]
    assert "standalone" in review_schema["required"]
    assert result[0].title == "Good"
    assert result[0].standalone is True
