"""Drawing the CelebA gallery and held-out identities (#9, #27), on synthetic identity lists."""

import pytest

from ryuk.evaluation.draws import (
    ENROLLED_PER_IDENTITY,
    MATED_PROBES_PER_IDENTITY,
    NON_MATED_PROBES_PER_IDENTITY,
    make_draw,
)


def _usable(counts: dict[int, int]) -> dict[int, list[str]]:
    """Identity -> that many usable image names, unique across identities."""
    return {
        identity: [f"{identity:04d}-{n:02d}.png" for n in range(count)]
        for identity, count in counts.items()
    }


# Identities 1-6 can be enrolled (at least 20 usable images); 7-9 cannot; 10 has none usable.
USABLE = _usable({1: 20, 2: 25, 3: 31, 4: 20, 5: 22, 6: 40, 7: 19, 8: 3, 9: 12, 10: 0})


def test_every_gallery_identity_has_five_enrolled_photos_and_fifteen_mated_probes() -> None:
    draw = make_draw("validation", USABLE, seed=7, gallery_size=4)

    assert len(draw.gallery) == 4
    for identity in draw.gallery:
        assert len(identity.enrolled) == ENROLLED_PER_IDENTITY == 5
        assert len(identity.probes) == MATED_PROBES_PER_IDENTITY == 15
        assert not set(identity.enrolled) & set(identity.probes)
        assert set(identity.enrolled) | set(identity.probes) <= set(USABLE[identity.identity])


def test_only_identities_with_twenty_usable_images_can_be_enrolled() -> None:
    for seed in range(20):
        draw = make_draw("validation", USABLE, seed=seed, gallery_size=6)

        assert [g.identity for g in draw.gallery] == [1, 2, 3, 4, 5, 6]


def test_every_other_identity_with_a_usable_image_is_held_out_with_up_to_ten_probes() -> None:
    draw = make_draw("test", USABLE, seed=3, gallery_size=4)

    gallery = {g.identity for g in draw.gallery}
    held_out = {h.identity: h.probes for h in draw.held_out}
    # The two gallery candidates not drawn are held out too, as is every identity under 20.
    assert set(held_out) == {1, 2, 3, 4, 5, 6, 7, 8, 9} - gallery
    assert 10 not in held_out
    assert len(held_out[7]) == NON_MATED_PROBES_PER_IDENTITY == 10
    assert sorted(held_out[8]) == USABLE[8]
    for identity, probes in held_out.items():
        assert len(set(probes)) == len(probes)
        assert set(probes) <= set(USABLE[identity])


def test_a_draw_is_fixed_by_its_seed() -> None:
    first = make_draw("validation", USABLE, seed=11, gallery_size=3)
    again = make_draw(
        "validation",
        {k: list(reversed(v)) for k, v in reversed(USABLE.items())},
        seed=11,
        gallery_size=3,
    )
    galleries = {
        tuple(g.identity for g in make_draw("validation", USABLE, seed=s, gallery_size=3).gallery)
        for s in range(10)
    }

    assert again == first
    assert again.selection_sha256 == first.selection_sha256
    assert len(galleries) > 1


def test_the_selection_hash_changes_with_any_chosen_image() -> None:
    draw = make_draw("validation", USABLE, seed=11, gallery_size=3)
    other = make_draw("validation", USABLE, seed=12, gallery_size=3)

    assert len(draw.selection_sha256) == 64
    assert draw.selection_sha256 != other.selection_sha256


def test_the_images_list_every_enrolled_photo_and_probe_once() -> None:
    draw = make_draw("validation", USABLE, seed=5, gallery_size=4)

    images = draw.images()
    assert len(images) == len(set(images)) == 4 * 20 + sum(len(h.probes) for h in draw.held_out)


def test_too_few_identities_to_fill_the_gallery_is_an_error() -> None:
    with pytest.raises(ValueError, match="6 identities have at least 20 usable images, 7 needed"):
        make_draw("validation", USABLE, seed=0, gallery_size=7)
