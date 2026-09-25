"""The summary models' lookups, which the figures, notebook and report read the summary through."""

import pytest

from ryuk.eda.summary import DRAW_SPLITS
from summaries import celeba_draw, eda_summary


def test_each_draw_is_one_celeba_split() -> None:
    assert list(DRAW_SPLITS.items()) == [("validation", "valid"), ("test", "test")]


def test_a_draw_is_found_by_name() -> None:
    summary = eda_summary()

    assert summary.draw("validation") is summary.celeba[0]
    assert summary.draw("test") is summary.celeba[1]


def test_a_missing_draw_is_a_key_error() -> None:
    summary = eda_summary().model_copy(update={"celeba": [celeba_draw("test")]})

    with pytest.raises(KeyError, match="no validation draw"):
        summary.draw("validation")


def test_a_group_is_found_by_attribute_and_value() -> None:
    draw = eda_summary().draw("validation")

    male, not_male = draw.group("Male", True), draw.group("Male", False)

    assert (male.attribute, male.value, male.images) == ("Male", True, 80)
    assert (not_male.attribute, not_male.value, not_male.images) == ("Male", False, 100)


def test_a_missing_group_is_a_key_error() -> None:
    draw = eda_summary().draw("test")
    draw = draw.model_copy(update={"groups": [g for g in draw.groups if g.value]})

    with pytest.raises(KeyError, match="test draw has no Blurry=False group"):
        draw.group("Blurry", False)


def test_an_attributes_prevalence_is_found_by_attribute() -> None:
    draw = eda_summary().draw("validation")

    assert draw.prevalence("Young").majority_with == 7
    with pytest.raises(KeyError, match="no prevalence for Blurry"):
        draw.model_copy(update={"attributes": []}).prevalence("Blurry")


@pytest.mark.parametrize(
    ("images", "identities"), [(0, 45), (1, 45), (2, 15), (3, 7), (7, 3), (8, 1), (40, 1), (41, 0)]
)
def test_identities_with_at_least_so_many_images(images: int, identities: int) -> None:
    # LFW here: 30 identities with 1 image, 8 with 2, 4 with 3, 2 with 7 and 1 with 40.
    assert eda_summary().lfw.images_per_identity.at_least(images) == identities
