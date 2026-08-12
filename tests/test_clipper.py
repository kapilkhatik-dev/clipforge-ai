from __future__ import annotations

from yt_clipper.domain.models import TranscriptSegment, TranscriptWord, VideoLayout
from yt_clipper.services.renderer import (
    build_ass_document,
    build_caption_cues,
    build_thumbnail_ass_document,
    build_thumbnail_filter_graph,
    build_vertical_poster_ass_document,
    build_vertical_poster_filter_graph,
    build_video_filter_graph,
    clip_transcript_segments,
    slugify,
)


def test_clip_segments_shift_and_clamp_timestamps() -> None:
    source = [
        TranscriptSegment(
            start=8,
            end=15,
            text="Opening words",
            words=[
                TranscriptWord(start=9, end=10, text="Opening"),
                TranscriptWord(start=10, end=11, text="words"),
            ],
        ),
        TranscriptSegment(start=16, end=20, text="Outside"),
    ]

    clipped = clip_transcript_segments(source, clip_start=10, clip_end=16)

    assert len(clipped) == 1
    assert (clipped[0].start, clipped[0].end) == (0.0, 5.0)
    assert [(word.start, word.end, word.text) for word in clipped[0].words] == [
        (0.0, 1.0, "words")
    ]


def test_partial_untimed_segment_uses_only_visible_words() -> None:
    source = [TranscriptSegment(start=0, end=10, text="one two three four")]

    clipped = clip_transcript_segments(source, clip_start=5, clip_end=10)

    assert clipped[0].text == "three four"


def test_ass_document_escapes_override_tags_and_wraps() -> None:
    segment = TranscriptSegment(
        start=0,
        end=4,
        text="This caption has enough words {not-a-tag} to wrap safely",
    )

    document = build_ass_document([segment])

    assert "PlayResX: 1080" in document
    assert "{not-a-tag}" not in document
    assert "(not-a-tag)" in document
    assert "\\N" in document
    assert "Dialogue: 0,0:00:00.00," in document
    assert ",0:00:04.00,Clip" in document


def test_ass_document_uses_sharp_bold_yellow_caption_style() -> None:
    document = build_ass_document(
        [TranscriptSegment(start=0, end=2, text="Readable caption")]
    )

    style = next(
        line for line in document.splitlines() if line.startswith("Style: Clip,")
    )
    assert "Arial,80" in style
    assert "&H0017DFFF,&H0017DFFF,&H00141414,&H50000000" in style
    assert ",-1,0,0,0,100,100,0.4,0,1,5,2.5,2,84,84,300,1" in style


def test_overlapping_duplicate_captions_produce_one_active_timeline() -> None:
    segments = [
        TranscriptSegment(start=0, end=3, text="Repeated caption"),
        TranscriptSegment(start=0.5, end=3.5, text=" repeated   CAPTION "),
        TranscriptSegment(start=2, end=4, text="Fresh caption"),
    ]

    cues = build_caption_cues(segments)
    document = build_ass_document(segments)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 2, "Repeated caption"),
        (2, 4, "Fresh caption"),
    ]
    assert all(current.end <= following.start for current, following in zip(cues, cues[1:]))
    assert document.count("Dialogue:") == 2
    assert document.casefold().count("repeated caption") == 1


def test_same_start_caption_collision_keeps_only_the_richer_text() -> None:
    cues = build_caption_cues(
        [
            TranscriptSegment(start=0, end=3, text="Short"),
            TranscriptSegment(start=0, end=4, text="A more complete caption"),
        ]
    )

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 4, "A more complete caption")
    ]


def test_thumbnail_title_is_safe_balanced_and_professionally_styled() -> None:
    document = build_thumbnail_ass_document(
        "A Surprisingly Complete {Thumbnail} Title That Needs Balanced Lines"
    )

    assert "{Thumbnail}" not in document
    assert "(Thumbnail)" in document
    assert document.count(r"\N") == 2
    assert "Style: Title,Arial," in document
    assert "&H00FFFFFF,&H00FFFFFF,&H00101010" in document
    assert "Dialogue: 0,0:00:00.00,0:00:10.00,Title" in document


def test_thumbnail_filter_uses_hd_artwork_with_readability_panel() -> None:
    graph = build_thumbnail_filter_graph()

    assert "scale=1280:720" in graph
    assert "flags=lanczos" in graph
    assert graph.count("drawbox=") == 2
    assert "black@0.72" in graph
    assert "0xFFDF17@1" in graph
    assert "subtitles=filename='thumbnail-title.ass'" in graph


def test_vertical_poster_title_is_balanced_for_nine_by_sixteen() -> None:
    document = build_vertical_poster_ass_document(
        "The Paranormal Experiences Were Ghosting"
    )

    assert "PlayResX: 1080" in document
    assert "PlayResY: 1920" in document
    assert document.count(r"\N") == 2
    assert "Style: PosterTitle,Arial," in document
    assert ",1,4,2.5,7,84,84,1037,1" in document


def test_vertical_poster_uses_blurred_artwork_and_safe_title_panel() -> None:
    graph = build_vertical_poster_filter_graph()

    assert "scale=1080:1920" in graph
    assert "gblur=sigma=42" in graph
    assert "overlay=(W-w)/2:0" in graph
    assert "drawbox=x=0:y=806" in graph
    assert "drawbox=x=84:y=898" in graph
    assert "black@0.76" in graph
    assert "0xFFDF17@1" in graph
    assert "subtitles=filename='poster-title.ass'[poster]" in graph


def test_slugify_produces_safe_deterministic_name() -> None:
    assert slugify("Why This Works?!") == "why-this-works"


def test_fill_crop_layout_uses_sharp_full_screen_video() -> None:
    graph = build_video_filter_graph(VideoLayout.FILL_CROP, 1080, 1920)

    assert "force_original_aspect_ratio=increase" in graph
    assert "crop=1080:1920" in graph
    assert "gblur" not in graph
    assert "overlay" not in graph
    assert "subtitles=filename='captions.ass'" in graph
    assert graph.count("subtitles=") == 1


def test_fit_blur_layout_preserves_existing_composition() -> None:
    graph = build_video_filter_graph(VideoLayout.FIT_BLUR, 1080, 1920)

    assert "gblur=sigma=24" in graph
    assert "force_original_aspect_ratio=decrease" in graph
    assert "overlay=(W-w)/2:(H-h)/2" in graph
    assert graph.count("subtitles=") == 1
