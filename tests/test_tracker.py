"""The live monitor's sighting tracker on its own: frames of matches in, the sightings to write and
announce out, with time passed in so every rule is exact (#16, #31)."""

import datetime
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from ryuk.detector import Box, Image
from ryuk.recognition import ModelKey
from ryuk.watchlist.live import Candidate, LiveFace, Match, NoMatch, Recognition, TooSmall
from ryuk.watchlist.tracker import (
    CROP_MARGIN,
    Ended,
    LiveSighting,
    Opened,
    SightingChange,
    SightingTracker,
    Updated,
    cut_crop,
)

MODEL = ModelKey(network="arcface", provider="coreml", weights_sha256="a" * 64)
OTHER_MODEL = ModelKey(network="sface", provider="cpu", weights_sha256="b" * 64)
THRESHOLD = 0.5
T0 = datetime.datetime(2026, 9, 29, 12, 0, tzinfo=datetime.UTC)
BOX = Box(10, 20, 40, 40)


def at(ms: float) -> datetime.datetime:
    return T0 + datetime.timedelta(milliseconds=ms)


def ids() -> Iterator[str]:
    return (f"s{n}" for n in range(1, 1000))


def tracker() -> SightingTracker:
    return SightingTracker(new_id=ids().__next__)


def image(fill: int = 0) -> Image:
    """A frame whose every pixel is `fill`, so a crop shows which frame it was cut from."""
    return np.full((120, 160, 3), fill, dtype=np.uint8)


def match(
    person: str,
    score: float,
    runner_up: tuple[str, float] | None = None,
    box: Box = BOX,
) -> Match:
    return Match(
        box,
        Candidate(person, person.title(), score),
        None if runner_up is None else Candidate(runner_up[0], runner_up[0].title(), runner_up[1]),
    )


def recognition(*faces: LiveFace, model: ModelKey = MODEL) -> Recognition:
    return Recognition(model, THRESHOLD, faces)


def feed(
    tracked: SightingTracker, frames: Sequence[tuple[float, Sequence[LiveFace]]]
) -> list[SightingChange]:
    """Every change from frames of `(ms, faces)`, in order, each frame filled with its index."""
    changes: list[SightingChange] = []
    for index, (ms, faces) in enumerate(frames):
        changes += tracked.observe(image(index), recognition(*faces), at(ms))
    return changes


def ada(score: float = 0.8) -> list[LiveFace]:
    return [match("ada", score)]


NOBODY: list[LiveFace] = []


def opened(tracked: SightingTracker, start_ms: float = 0, score: float = 0.8) -> LiveSighting:
    """Ada's sighting, confirmed by three matched frames 100 ms apart from `start_ms`."""
    changes = feed(tracked, [(start_ms + n * 100, ada(score)) for n in range(3)])
    [change] = changes
    assert isinstance(change, Opened)
    return change.sighting


# Confirmation


def test_a_single_fluke_frame_never_opens_a_sighting() -> None:
    tracked = tracker()

    changes = feed(tracked, [(0, ada()), (100, NOBODY), (200, NOBODY), (5000, NOBODY)])

    assert changes == []
    assert tracked.sighting_id("ada") is None


def test_two_matched_frames_out_of_two_are_not_enough() -> None:
    tracked = tracker()

    assert feed(tracked, [(0, ada()), (100, ada())]) == []


def test_three_matched_frames_in_a_row_open_a_sighting_at_the_third() -> None:
    tracked = tracker()

    changes = feed(tracked, [(0, ada(0.7)), (100, ada(0.9)), (200, ada(0.8))])

    [change] = changes
    assert change == Opened(
        LiveSighting(
            id="s1",
            person_id="ada",
            model=MODEL,
            threshold=THRESHOLD,
            started_at=at(0),
            last_seen_at=at(200),
            ended_at=None,
            best_score=0.9,
            best_crop=image(),
            runner_up_person_id=None,
            runner_up_score=None,
        )
    )
    assert tracked.sighting_id("ada") == "s1"


def test_three_of_six_frames_is_half_and_opens() -> None:
    tracked = tracker()
    frames = [(n * 80, ada() if n % 2 else NOBODY) for n in range(6)]

    changes = feed(tracked, frames)

    [change] = changes
    assert isinstance(change, Opened)
    assert change.sighting.started_at == at(80)
    assert change.sighting.last_seen_at == at(400)


def test_two_of_five_frames_do_not_open() -> None:
    tracked = tracker()
    frames = [(n * 80, ada() if n in {1, 3} else NOBODY) for n in range(5)]

    assert feed(tracked, frames) == []


def test_three_of_seven_frames_are_under_half_and_do_not_open() -> None:
    tracked = tracker()
    frames = [(n * 60, ada() if n >= 4 else NOBODY) for n in range(7)]

    assert feed(tracked, frames) == []


def test_every_frame_counts_even_one_with_no_faces_or_only_faces_too_small() -> None:
    tracked = tracker()
    small: list[LiveFace] = [TooSmall(BOX)]
    no_match: list[LiveFace] = [NoMatch(BOX, 0.2)]
    frames = [(0, NOBODY), (50, small), (100, no_match), (150, NOBODY), (200, ada()), (250, ada())]

    # Three matches out of seven frames: under half.
    assert feed(tracked, [*frames, (300, ada())]) == []


def test_frames_older_than_the_500_ms_window_leave_it() -> None:
    tracked = tracker()

    # At 500 ms the window is (0, 500]: the frame at 0 ms has left it.
    assert feed(tracked, [(0, ada()), (250, ada()), (500, ada())]) == []
    [change] = tracked.observe(image(), recognition(*ada()), at(600))

    assert isinstance(change, Opened)
    assert change.sighting.started_at == at(250)


def test_several_faces_of_one_person_in_a_frame_count_once_with_the_highest_score() -> None:
    tracked = tracker()
    twice = [match("ada", 0.6), match("ada", 0.95, box=Box(60, 20, 40, 40))]

    assert feed(tracked, [(0, twice), (100, twice)]) == []
    [change] = tracked.observe(image(9), recognition(match("ada", 0.7)), at(200))

    assert isinstance(change, Opened)
    assert change.sighting.best_score == 0.95
    assert np.array_equal(change.sighting.best_crop, cut_crop(image(0), Box(60, 20, 40, 40)))


def test_of_two_equal_faces_of_one_person_in_a_frame_the_first_is_kept() -> None:
    tracked = tracker()
    twins = [match("ada", 0.9, box=Box(60, 20, 40, 40)), match("ada", 0.9)]

    [change] = feed(tracked, [(0, twins), (100, twins), (200, twins)])

    assert np.array_equal(change.sighting.best_crop, cut_crop(image(0), Box(60, 20, 40, 40)))


def test_the_best_match_of_the_confirming_window_gives_the_crop_and_its_runner_up() -> None:
    tracked = tracker()

    changes = feed(
        tracked,
        [
            (0, [match("ada", 0.7, runner_up=("bob", 0.4))]),
            (100, [match("ada", 0.9, runner_up=("cy", 0.3))]),
            (200, [match("ada", 0.9, runner_up=("bob", 0.45))]),
        ],
    )

    [change] = changes
    sighting = change.sighting
    assert (sighting.best_score, sighting.runner_up_person_id) == (0.9, "cy")
    assert sighting.runner_up_score == 0.3
    assert np.array_equal(sighting.best_crop, cut_crop(image(1), BOX))


def test_several_persons_are_tracked_apart_in_one_frame() -> None:
    tracked = tracker()
    both = [match("ada", 0.8), match("bob", 0.7, box=Box(80, 20, 40, 40))]

    changes = feed(tracked, [(0, both), (100, both), (200, both)])

    assert [(type(c), c.sighting.person_id, c.sighting.id) for c in changes] == [
        (Opened, "ada", "s1"),
        (Opened, "bob", "s2"),
    ]


# Best crop and runner-up after opening


def test_the_crop_is_replaced_only_by_a_strictly_higher_score_and_the_runner_up_follows() -> None:
    tracked = tracker()
    opened(tracked, score=0.8)

    tracked.observe(image(50), recognition(match("ada", 0.8, runner_up=("bob", 0.4))), at(300))
    tracked.observe(image(60), recognition(match("ada", 0.85, runner_up=("cy", 0.2))), at(400))
    tracked.observe(image(70), recognition(match("ada", 0.85, runner_up=("bob", 0.5))), at(500))
    tracked.observe(image(80), recognition(match("ada", 0.6, runner_up=("bob", 0.5))), at(600))
    [ended] = tracked.end_all()

    sighting = ended.sighting
    assert sighting.best_score == 0.85
    assert (sighting.runner_up_person_id, sighting.runner_up_score) == ("cy", 0.2)
    assert np.array_equal(sighting.best_crop, cut_crop(image(60), BOX))


def test_a_best_frame_with_no_runner_up_clears_the_runner_up() -> None:
    tracked = tracker()
    feed(tracked, [(n * 100, [match("ada", 0.7, runner_up=("bob", 0.4))]) for n in range(3)])

    tracked.observe(image(), recognition(match("ada", 0.9)), at(300))
    [ended] = tracked.end_all()

    assert (ended.sighting.runner_up_person_id, ended.sighting.runner_up_score) == (None, None)


# The gap


def test_a_sighting_ends_on_a_tick_exactly_3_s_after_the_last_match() -> None:
    tracked = tracker()
    sighting = opened(tracked)

    assert tracked.tick(at(200 + 2999)) == []
    [change] = tracked.tick(at(200 + 3000))

    assert change == Ended(
        LiveSighting(
            id=sighting.id,
            person_id="ada",
            model=MODEL,
            threshold=THRESHOLD,
            started_at=at(0),
            last_seen_at=at(200),
            ended_at=at(200),
            best_score=0.8,
            best_crop=sighting.best_crop,
            runner_up_person_id=None,
            runner_up_score=None,
        ),
        new_best=False,
    )
    assert tracked.sighting_id("ada") is None
    assert tracked.tick(at(10_000)) == []


def test_a_frame_without_the_person_ends_their_sighting_after_the_gap() -> None:
    tracked = tracker()
    opened(tracked)

    [change] = tracked.observe(image(), recognition(), at(3200))

    assert isinstance(change, Ended)
    assert change.sighting.ended_at == at(200)


def test_a_below_threshold_frame_does_not_extend_a_sighting() -> None:
    tracked = tracker()
    opened(tracked)

    for ms in range(300, 3200, 100):
        tracked.observe(image(), recognition(NoMatch(BOX, 0.4)), at(ms))
    [change] = tracked.tick(at(3200))

    assert isinstance(change, Ended)
    assert change.sighting.last_seen_at == at(200)


def test_a_match_keeps_a_sighting_open_past_3_s_from_its_start() -> None:
    tracked = tracker()
    opened(tracked)

    [update] = tracked.observe(image(), recognition(*ada()), at(3000))

    assert isinstance(update, Updated)
    assert update.sighting.last_seen_at == at(3000)
    assert tracked.tick(at(5999)) == []
    [change] = tracked.tick(at(6000))
    assert isinstance(change, Ended)
    assert change.sighting.ended_at == at(3000)


def test_a_match_exactly_3_s_after_the_last_one_comes_after_the_end() -> None:
    tracked = tracker()
    first = opened(tracked)

    [change] = tracked.observe(image(), recognition(*ada()), at(3200))

    assert isinstance(change, Ended)
    assert (change.sighting.id, change.sighting.ended_at) == (first.id, at(200))
    assert tracked.sighting_id("ada") is None


def test_a_person_seen_again_after_their_sighting_ended_opens_a_new_one_only_once_confirmed() -> (
    None
):
    tracked = tracker()
    first = opened(tracked)
    tracked.tick(at(4000))

    assert feed(tracked, [(5000, ada()), (5100, ada())]) == []
    [change] = tracked.observe(image(), recognition(*ada()), at(5200))

    assert isinstance(change, Opened)
    assert change.sighting.id != first.id
    assert change.sighting.started_at == at(5000)
    assert tracked.sighting_id("ada") == change.sighting.id


# Writes


def test_changes_are_written_at_most_once_a_second() -> None:
    tracked = tracker()
    opened(tracked)  # written at 200 ms

    within = feed(tracked, [(ms, ada()) for ms in range(300, 1200, 100)])
    [first] = tracked.observe(image(), recognition(*ada()), at(1200))
    again = feed(tracked, [(ms, ada()) for ms in range(1300, 2200, 100)])
    [second] = tracked.observe(image(), recognition(*ada()), at(2200))

    assert within == []
    assert isinstance(first, Updated)
    assert first.sighting.last_seen_at == at(1200)
    assert again == []
    assert isinstance(second, Updated)
    assert second.sighting.last_seen_at == at(2200)


def test_held_changes_are_written_on_a_tick_once_frames_stop() -> None:
    tracked = tracker()
    opened(tracked)
    tracked.observe(image(), recognition(*ada(0.95)), at(300))

    assert tracked.tick(at(1199)) == []
    [change] = tracked.tick(at(1200))

    assert isinstance(change, Updated)
    assert (change.sighting.last_seen_at, change.sighting.best_score) == (at(300), 0.95)
    assert change.sighting.ended_at is None
    assert tracked.tick(at(2500)) == []


def test_nothing_is_written_while_nothing_changed() -> None:
    tracked = tracker()
    opened(tracked)

    assert feed(tracked, [(ms, NOBODY) for ms in range(300, 3200, 100)]) == []


def test_a_tick_alone_ends_a_sighting_when_frames_stop_and_writes_the_changes_still_held() -> None:
    tracked = tracker()
    opened(tracked)
    tracked.observe(image(), recognition(*ada(0.95)), at(700))

    [change] = tracked.tick(at(3700))

    assert isinstance(change, Ended)
    assert (change.sighting.last_seen_at, change.sighting.ended_at) == (at(700), at(700))
    assert change.sighting.best_score == 0.95


# Explicit ends


def test_ending_a_person_ends_only_their_sighting_at_its_last_seen_time() -> None:
    tracked = tracker()
    both = [match("ada", 0.8), match("bob", 0.7, box=Box(80, 20, 40, 40))]
    feed(tracked, [(0, both), (100, both), (200, both)])

    [change] = tracked.end_person("ada")

    assert isinstance(change, Ended)
    assert (change.sighting.person_id, change.sighting.ended_at) == ("ada", at(200))
    assert tracked.sighting_id("ada") is None
    assert tracked.sighting_id("bob") == "s2"
    assert tracked.end_person("ada") == []
    assert tracked.end_person("nobody") == []


def test_ending_a_person_forgets_their_matches_awaiting_confirmation() -> None:
    tracked = tracker()
    feed(tracked, [(0, ada()), (100, ada())])

    assert tracked.end_person("ada") == []
    assert tracked.observe(image(), recognition(*ada()), at(200)) == []


def test_ending_all_ends_every_open_sighting_and_forgets_the_window() -> None:
    tracked = tracker()
    both = [match("ada", 0.8), match("bob", 0.7, box=Box(80, 20, 40, 40))]
    feed(tracked, [(0, both), (100, both), (200, both), (300, [match("cy", 0.9)])])
    feed(tracked, [(400, [match("cy", 0.9)])])

    changes = tracked.end_all()

    assert [(type(c), c.sighting.person_id, c.sighting.ended_at) for c in changes] == [
        (Ended, "ada", at(200)),
        (Ended, "bob", at(200)),
    ]
    assert tracked.end_all() == []
    # Cy's two matches were forgotten, so one more is not a confirmation.
    assert tracked.observe(image(), recognition(match("cy", 0.9)), at(500)) == []


def test_a_frame_judged_by_another_model_ends_every_sighting_first() -> None:
    tracked = tracker()
    opened(tracked)
    feed(tracked, [(250, [match("bob", 0.9)]), (260, [match("bob", 0.9)])])

    [change] = tracked.observe(image(), recognition(match("bob", 0.9), model=OTHER_MODEL), at(300))

    assert isinstance(change, Ended)
    assert change.sighting.person_id == "ada"
    # Bob's matches under the old model were forgotten: two more under the new one are needed.
    assert (
        tracked.observe(image(), recognition(match("bob", 0.9), model=OTHER_MODEL), at(400)) == []
    )
    [reopened] = tracked.observe(
        image(), recognition(match("bob", 0.9), model=OTHER_MODEL), at(500)
    )
    assert isinstance(reopened, Opened)
    assert reopened.sighting.model == OTHER_MODEL
    assert reopened.sighting.started_at == at(300)


def test_a_purged_runner_up_is_cleared_with_its_score_kept() -> None:
    tracked = tracker()
    feed(tracked, [(n * 100, [match("ada", 0.8, runner_up=("bob", 0.4))]) for n in range(3)])
    feed(tracked, [(n * 100, [match("cy", 0.8, runner_up=("bob", 0.3))]) for n in (3, 4)])

    tracked.clear_runner_up("bob")
    [cy] = tracked.observe(image(), recognition(match("cy", 0.7, runner_up=("dee", 0.1))), at(500))
    [ada_ended] = tracked.end_person("ada")

    assert (ada_ended.sighting.runner_up_person_id, ada_ended.sighting.runner_up_score) == (
        None,
        0.4,
    )
    assert isinstance(cy, Opened)
    assert (cy.sighting.runner_up_person_id, cy.sighting.runner_up_score) == (None, 0.3)


def test_clearing_a_runner_up_is_held_not_written() -> None:
    tracked = tracker()
    feed(tracked, [(n * 100, [match("ada", 0.8, runner_up=("bob", 0.4))]) for n in range(3)])

    tracked.clear_runner_up("bob")

    # The purge already cleared the stored runner-up, and it is never sent live.
    assert tracked.tick(at(1500)) == []


def test_time_never_runs_backwards_for_the_tracker() -> None:
    tracked = tracker()
    opened(tracked)

    tracked.observe(image(), recognition(*ada()), at(100))
    [change] = tracked.end_all()

    assert (change.sighting.last_seen_at, change.sighting.ended_at) == (at(200), at(200))


# The crop


def test_the_crop_is_the_box_widened_by_the_margin_on_each_side() -> None:
    frame = np.arange(120 * 160 * 3, dtype=np.uint32).astype(np.uint8).reshape(120, 160, 3)

    crop = cut_crop(frame, Box(40, 30, 40, 40))

    pad = int(40 * CROP_MARGIN)
    assert crop.shape == (40 + 2 * pad, 40 + 2 * pad, 3)
    assert np.array_equal(crop, frame[30 - pad : 70 + pad, 40 - pad : 80 + pad])


def test_the_crop_is_clipped_to_the_frame() -> None:
    frame = image(7)

    crop = cut_crop(frame, Box(-10.5, 100, 40, 40))

    assert crop.shape == (30, 40, 3)


def test_the_crop_is_a_read_only_copy_not_a_view_of_the_frame() -> None:
    frame = image(7)

    crop = cut_crop(frame, BOX)
    frame[:] = 0

    assert not np.shares_memory(crop, frame)
    assert (crop == 7).all()
    with pytest.raises(ValueError, match="read-only"):
        crop[0, 0, 0] = 1


def test_a_box_wholly_outside_the_frame_has_no_crop() -> None:
    with pytest.raises(ValueError, match="outside"):
        cut_crop(image(), Box(500, 500, 40, 40))


def test_clearing_a_runner_up_leaves_every_other_runner_up() -> None:
    tracked = tracker()
    feed(tracked, [(n * 100, [match("ada", 0.8, runner_up=("dee", 0.4))]) for n in range(3)])
    feed(tracked, [(n * 100, [match("cy", 0.8, runner_up=("dee", 0.3))]) for n in (3, 4)])

    tracked.clear_runner_up("bob")
    [cy] = tracked.observe(image(), recognition(match("cy", 0.7)), at(500))
    [ada_ended] = tracked.end_person("ada")

    assert isinstance(cy, Opened)
    assert (cy.sighting.runner_up_person_id, cy.sighting.runner_up_score) == ("dee", 0.3)
    assert ada_ended.sighting.runner_up_person_id == "dee"


def test_sighting_ids_are_opaque_and_new_each_time() -> None:
    tracked = SightingTracker()
    first = opened(tracked)
    tracked.tick(at(4000))
    [second] = feed(tracked, [(5000 + n * 100, ada()) for n in range(3)])

    assert len(first.id) == 32
    assert int(first.id, 16) >= 0
    assert second.sighting.id != first.id


def test_a_write_says_whether_it_carries_a_new_best_match() -> None:
    tracked = tracker()
    opened(tracked, score=0.8)

    tracked.observe(image(), recognition(*ada(0.8)), at(300))
    [same] = tracked.tick(at(1200))
    tracked.observe(image(), recognition(*ada(0.9)), at(1300))
    [better] = tracked.tick(at(2200))
    tracked.observe(image(), recognition(*ada(0.85)), at(2300))
    [ended] = tracked.end_all()

    assert isinstance(same, Updated)
    assert not same.new_best
    assert isinstance(better, Updated)
    assert better.new_best
    assert isinstance(ended, Ended)
    assert not ended.new_best


def test_an_end_carries_a_new_best_match_not_yet_written() -> None:
    tracked = tracker()
    opened(tracked, score=0.8)
    tracked.observe(image(), recognition(*ada(0.9)), at(300))

    [ended] = tracked.end_person("ada")

    assert isinstance(ended, Ended)
    assert ended.new_best


# Snapshots, so a caller can undo what a failed write decided


def test_restoring_a_snapshot_undoes_an_opening() -> None:
    tracked = tracker()
    feed(tracked, [(0, ada()), (100, ada())])
    before = tracked.snapshot()

    [first] = tracked.observe(image(), recognition(*ada()), at(200))
    tracked.restore(before)

    assert tracked.sighting_id("ada") is None
    [retried] = tracked.observe(image(), recognition(*ada()), at(250))
    assert isinstance(first, Opened)
    assert isinstance(retried, Opened)
    assert retried.sighting.started_at == at(0)
    assert retried.sighting.last_seen_at == at(250)


def test_restoring_a_snapshot_undoes_held_changes_and_an_end() -> None:
    tracked = tracker()
    sighting = opened(tracked, score=0.8)
    before = tracked.snapshot()

    tracked.observe(image(), recognition(*ada(0.9)), at(300))
    tracked.end_all()
    tracked.restore(before)

    assert tracked.sighting_id("ada") == sighting.id
    assert tracked.tick(at(1300)) == []  # nothing held once more
    [ended] = tracked.tick(at(3200))
    assert isinstance(ended, Ended)
    assert (ended.sighting.best_score, ended.sighting.last_seen_at) == (0.8, at(200))


def test_restoring_a_snapshot_undoes_a_cleared_runner_up_and_a_model_change() -> None:
    tracked = tracker()
    feed(tracked, [(n * 100, [match("ada", 0.8, runner_up=("bob", 0.4))]) for n in range(3)])
    before = tracked.snapshot()

    tracked.clear_runner_up("bob")
    tracked.observe(image(), recognition(model=OTHER_MODEL), at(300))
    tracked.restore(before)

    [ended] = tracked.end_person("ada")
    assert ended.sighting.runner_up_person_id == "bob"


def test_a_snapshot_can_be_restored_more_than_once() -> None:
    tracked = tracker()
    feed(tracked, [(0, ada()), (100, ada())])
    before = tracked.snapshot()

    for ms in (200, 210):
        [change] = tracked.observe(image(), recognition(*ada()), at(ms))
        assert isinstance(change, Opened)
        tracked.restore(before)

    assert tracked.sighting_id("ada") is None
