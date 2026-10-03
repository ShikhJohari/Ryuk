"""#9's rule for the first active model, judged on synthetic contenders."""

from ryuk.evaluation.active import Contender, first_active_model
from ryuk.evaluation.results import Interval, LfwGate, Rate, RecognitionModelId
from ryuk.recognition import Network

GATE = LfwGate(accuracy="scored-pairs", scored_pairs=5917, pairs=6000, tolerance_points=0.5)


def _id(network: Network) -> RecognitionModelId:
    return RecognitionModelId(
        network=network, provider="cpu", weights_sha256="a" * 64, dimension=512
    )


def _contender(
    network: Network,
    tpir: tuple[float, float, float],
    *,
    gap: float | None = 0.1,
    fpir: float = 0.012,
    ms: float = 10.0,
) -> Contender:
    value, low, high = tpir
    return Contender(
        model=_id(network),
        lfw_gap_points=gap,
        test_tpir=Rate(value=value, ci=Interval(low=low, high=high)),
        test_fpir=fpir,
        ms_per_face=ms,
    )


def test_the_highest_test_tpir_wins_when_no_other_interval_overlaps_it() -> None:
    chosen = first_active_model(
        [
            _contender("sface", (0.80, 0.78, 0.82), ms=5),
            _contender("arcface", (0.95, 0.94, 0.96), ms=20),
        ],
        GATE,
    )

    assert chosen.model == _id("arcface")
    assert "highest test TPIR" in chosen.reason
    assert [c.eligible for c in chosen.eligibility] == [True, True]


def test_overlapping_intervals_go_to_the_faster_model() -> None:
    chosen = first_active_model(
        [
            _contender("arcface", (0.95, 0.93, 0.97), ms=20),
            _contender("sface", (0.94, 0.92, 0.96), ms=5),
            _contender("facenet", (0.90, 0.88, 0.92), ms=1),
        ],
        GATE,
    )

    # FaceNet is fastest but its interval does not reach ArcFace's.
    assert chosen.model == _id("sface")
    assert "overlaps" in chosen.reason
    assert "faster" in chosen.reason


def test_a_model_off_its_published_lfw_figure_is_not_eligible() -> None:
    chosen = first_active_model(
        [
            _contender("arcface", (0.99, 0.98, 1.0), gap=-0.6),
            _contender("sface", (0.80, 0.78, 0.82)),
        ],
        GATE,
    )

    assert chosen.model == _id("sface")
    assert not chosen.eligibility[0].reproduces_lfw
    assert not chosen.eligibility[0].eligible


def test_a_model_lfw_did_not_score_is_not_eligible() -> None:
    chosen = first_active_model([_contender("sface", (0.8, 0.78, 0.82), gap=None)], GATE)

    assert chosen.model is None


def test_test_fpir_over_two_percent_or_over_30_ms_per_face_is_not_eligible() -> None:
    chosen = first_active_model(
        [
            _contender("arcface", (0.99, 0.98, 1.0), fpir=0.021),
            _contender("facenet", (0.97, 0.96, 0.98), ms=30.5),
            _contender("sface", (0.80, 0.78, 0.82), fpir=0.02, ms=30.0),
        ],
        GATE,
    )

    assert chosen.model == _id("sface")
    fpir, speed = chosen.eligibility[0], chosen.eligibility[1]
    assert (fpir.fpir_within_limit, fpir.fast_enough) == (False, True)
    assert (speed.fpir_within_limit, speed.fast_enough) == (True, False)


def test_no_eligible_model_leaves_the_monitor_without_one_and_says_why() -> None:
    chosen = first_active_model([_contender("arcface", (0.99, 0.98, 1.0), fpir=0.05)], GATE)

    assert chosen.model is None
    assert chosen.reason.startswith("No model is eligible")


def test_the_gate_recorded_is_the_gate_applied() -> None:
    contenders = [_contender("sface", (0.80, 0.78, 0.82), gap=-0.1)]
    strict = GATE.model_copy(update={"tolerance_points": 0.05})

    assert first_active_model(contenders, GATE).model == _id("sface")
    chosen = first_active_model(contenders, strict)
    assert chosen.model is None
    assert chosen.lfw_gate == strict
    assert "within 0.05 points" in chosen.reason
