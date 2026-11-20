"""The evaluation page's view accounts for every field of the results (#74): each is served, or
named in `SERVES` as left out with a reason, so a field added to the results cannot be dropped
from `GET /api/evaluation` unnoticed."""

from collections.abc import Iterator, Mapping
from typing import TypeAliasType, get_args, get_origin

from pydantic import BaseModel

from ryuk.api.evaluation_models import SERVES, EvaluationReport, Serves
from ryuk.evaluation.results import Results


def models_in(annotation: object) -> Iterator[type[BaseModel]]:
    """Every model a field's type holds, through aliases, optionals, lists, tuples and mappings."""
    if isinstance(annotation, TypeAliasType):
        yield from models_in(annotation.__value__)
    elif get_origin(annotation) is not None:
        for argument in get_args(annotation):
            yield from models_in(argument)
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation


def unaccounted(
    measured: type[BaseModel],
    view: type[BaseModel],
    serves: Mapping[type[BaseModel], Serves],
) -> list[str]:
    """What `serves` gets wrong, walking from `measured` served as `view`: every results field
    neither served nor left out, and every entry naming a field that does not exist."""
    problems = dict.fromkeys(_dead_entries(serves))
    seen: set[tuple[type[BaseModel], type[BaseModel]]] = set()

    def walk(measured: type[BaseModel], view: type[BaseModel], account: Serves, where: str) -> None:
        for name, info in measured.model_fields.items():
            if name in account.left_out:
                continue
            if name in account.inline:
                inlined = account.inline[name]
                for nested in models_in(info.annotation):
                    if nested is inlined.source or nested in inlined.also:
                        walk(nested, view, inlined, f"{where}.inline[{name!r}]")
                    else:
                        problems[
                            f"{measured.__name__}.{name} holds {nested.__name__}, but "
                            f"{where}.inline[{name!r}] serves {inlined.source.__name__}"
                        ] = None
                continue
            served_as = account.renamed.get(name, name)
            if served_as not in view.model_fields:
                problems[
                    f"{measured.__name__}.{name} is neither served nor left out: add it to the "
                    f"view's {view.__name__}, or name it in {where}.left_out with a reason"
                ] = None
                continue
            served = list(models_in(view.model_fields[served_as].annotation))
            for nested in models_in(info.annotation):
                target = next(
                    (
                        candidate
                        for candidate in served
                        if candidate in serves
                        and nested in (serves[candidate].source, *serves[candidate].also)
                    ),
                    None,
                )
                if target is None:
                    problems[
                        f"{measured.__name__}.{name} holds {nested.__name__}, which no view "
                        f"class in {view.__name__}.{served_as} serves: name it as one's source "
                        f"in SERVES, or name {name} in {where}.left_out with a reason"
                    ] = None
                elif (nested, target) not in seen:
                    seen.add((nested, target))
                    walk(nested, target, serves[target], f"SERVES[{target.__name__}]")

    if view in serves:
        walk(measured, view, serves[view], f"SERVES[{view.__name__}]")
    else:
        problems[f"SERVES has no entry for {view.__name__}"] = None
    return list(problems)


def _dead_entries(serves: Mapping[type[BaseModel], Serves]) -> Iterator[str]:
    def entries(view: type[BaseModel], account: Serves, where: str) -> Iterator[str]:
        named = [*account.left_out, *account.renamed, *account.inline]
        for source in (account.source, *account.also):
            for name in named:
                if name not in source.model_fields:
                    yield f"{where} names {source.__name__}.{name}, which is not a field"
        for name, served_as in account.renamed.items():
            if served_as not in view.model_fields:
                yield f"{where} serves {name} as {view.__name__}.{served_as}, which is not a field"
        for name, inlined in account.inline.items():
            yield from entries(view, inlined, f"{where}.inline[{name!r}]")

    for view, account in serves.items():
        yield from entries(view, account, f"SERVES[{view.__name__}]")


def test_every_results_field_is_served_or_left_out_with_a_reason() -> None:
    assert unaccounted(Results, EvaluationReport, SERVES) == []


def test_every_reason_is_written() -> None:
    def reasons(account: Serves) -> Iterator[str]:
        yield from account.left_out.values()
        for inlined in account.inline.values():
            yield from reasons(inlined)

    assert all(reason.strip() for account in SERVES.values() for reason in reasons(account))


class Secret(BaseModel):
    seed: int


class Point(BaseModel):
    x: float
    y: float


class Track(BaseModel):
    name: str
    points: tuple[Point, ...]
    by_label: dict[str, Point | None]
    secret: Secret


class Measured(BaseModel):
    tracks: list[Track] | None


class ServedPoint(BaseModel):
    x: float
    y: float


class ServedTrack(BaseModel):
    name: str
    points: list[ServedPoint]
    by_label: dict[str, ServedPoint | None]


class View(BaseModel):
    tracks: list[ServedTrack] | None


ACCOUNTED: Mapping[type[BaseModel], Serves] = {
    View: Serves(Measured),
    ServedTrack: Serves(Track, left_out={"secret": "a seed, for rebuilding"}),
    ServedPoint: Serves(Point),
}


def test_the_check_passes_a_view_that_accounts_for_every_field() -> None:
    assert unaccounted(Measured, View, ACCOUNTED) == []


def test_a_field_nobody_accounted_for_fails_naming_its_class_and_field() -> None:
    class Point(BaseModel):
        x: float
        y: float
        z: float

    class Track(BaseModel):
        name: str
        points: tuple[Point, ...]
        by_label: dict[str, Point | None]
        secret: Secret

    class Measured(BaseModel):
        tracks: list[Track] | None

    serves: Mapping[type[BaseModel], Serves] = {
        View: Serves(Measured),
        ServedTrack: Serves(Track, left_out={"secret": "a seed, for rebuilding"}),
        ServedPoint: Serves(Point),
    }

    assert unaccounted(Measured, View, serves) == [
        "Point.z is neither served nor left out: add it to the view's ServedPoint, or name it in "
        "SERVES[ServedPoint].left_out with a reason"
    ]


def test_a_field_under_a_field_left_out_is_not_reported() -> None:
    class Secret(BaseModel):
        seed: int
        salt: str

    class Track(BaseModel):
        name: str
        points: tuple[Point, ...]
        by_label: dict[str, Point | None]
        secret: Secret

    class Measured(BaseModel):
        tracks: list[Track] | None

    serves: Mapping[type[BaseModel], Serves] = {
        View: Serves(Measured),
        ServedTrack: Serves(Track, left_out={"secret": "a seed and salt, for rebuilding"}),
        ServedPoint: Serves(Point),
    }

    assert unaccounted(Measured, View, serves) == []


def test_a_field_served_inline_or_renamed_is_accounted_for() -> None:
    class Flat(BaseModel):
        name: str
        x: float
        seed_count: int

    class FlatView(BaseModel):
        track: Flat

    class Holder(BaseModel):
        track: Track

    serves: Mapping[type[BaseModel], Serves] = {
        FlatView: Serves(Holder),
        Flat: Serves(
            Track,
            left_out={"points": "drawn elsewhere", "by_label": "drawn elsewhere"},
            inline={"secret": Serves(Secret, renamed={"seed": "seed_count"})},
        ),
    }

    assert unaccounted(Holder, FlatView, serves) == []


def test_an_entry_naming_a_field_that_does_not_exist_fails() -> None:
    serves: Mapping[type[BaseModel], Serves] = {
        **ACCOUNTED,
        ServedTrack: Serves(
            Track,
            left_out={"secret": "a seed, for rebuilding", "colour": "renamed long ago"},
            renamed={"name": "title"},
        ),
    }

    assert unaccounted(Measured, View, serves) == [
        "SERVES[ServedTrack] names Track.colour, which is not a field",
        "SERVES[ServedTrack] serves name as ServedTrack.title, which is not a field",
        "Track.name is neither served nor left out: add it to the view's ServedTrack, or name it "
        "in SERVES[ServedTrack].left_out with a reason",
    ]


def test_a_nested_model_no_view_class_serves_fails() -> None:
    serves: Mapping[type[BaseModel], Serves] = {View: Serves(Measured), ServedPoint: Serves(Point)}

    assert unaccounted(Measured, View, serves) == [
        "Measured.tracks holds Track, which no view class in View.tracks serves: name it as one's "
        "source in SERVES, or name tracks in SERVES[View].left_out with a reason"
    ]
