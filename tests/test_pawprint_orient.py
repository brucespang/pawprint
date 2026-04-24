"""Tests for pawprint._orient_for_print and pawprint._resolve_reverse.

These cover the orientation transform applied to the binarized image
before it goes to the encoder, and the markdown-vs-image default for
`--reverse`. We construct a small distinctive 2D bool image (a "ones in
the top-left corner" marker) so each transform's effect is unambiguous.
"""
import numpy as np
import pytest

from pawprint import _orient_for_print, _resolve_reverse


def _marker_image() -> np.ndarray:
    """4x6 bool image with a single True at (row=0, col=0).

    Asymmetric on both axes so identity, fliplr, flipud, and rot180 each
    produce a distinct array (the True ends up at a different corner).
    """
    img = np.zeros((4, 6), dtype=bool)
    img[0, 0] = True
    return img


def test_top_first_returns_image_unchanged():
    img = _marker_image()
    out = _orient_for_print(img, top_first=True, reverse=False)
    assert np.array_equal(out, img)


def test_top_first_and_reverse_both_send_raw():
    """Both flags map to the same "no transform" path - the printer's
    natural-orientation strip is the source image itself."""
    img = _marker_image()
    a = _orient_for_print(img, top_first=True, reverse=False)
    b = _orient_for_print(img, top_first=False, reverse=True)
    c = _orient_for_print(img, top_first=True, reverse=True)
    assert np.array_equal(a, img)
    assert np.array_equal(b, img)
    assert np.array_equal(c, img)


def test_default_image_path_is_rot180():
    """Legacy behavior: raw images get rot-180 so the user can tear the
    strip off and rotate it 180 degrees in-hand to read it title-up."""
    img = _marker_image()
    out = _orient_for_print(img, top_first=False, reverse=False)
    assert np.array_equal(out, np.rot90(img, k=2))
    assert not np.array_equal(out, img), (
        "legacy default must differ from the no-transform reverse path"
    )


def test_resolve_reverse_explicit_true_wins():
    assert _resolve_reverse("foo.png", True) is True
    assert _resolve_reverse("foo.md", True) is True


def test_resolve_reverse_explicit_false_wins():
    assert _resolve_reverse("foo.png", False) is False
    assert _resolve_reverse("foo.md", False) is False


@pytest.mark.parametrize("name", ["a.md", "A.MD", "notes.markdown", "x.MarkDown"])
def test_resolve_reverse_defaults_on_for_markdown(name):
    assert _resolve_reverse(name, None) is True


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.bmp", "a.tiff", "a"])
def test_resolve_reverse_defaults_off_for_raw_images(name):
    assert _resolve_reverse(name, None) is False
