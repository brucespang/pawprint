"""Tests for the image-encoding helpers in catprinter.cmds."""
import io

import cv2
import numpy as np
import pytest

from catprinter import cmds
from catprinter.img import read_img, read_img_grayscale


def test_encode_1bpp_row_all_white():
    row = np.zeros(cmds.PRINTER_WIDTH_PIXELS, dtype=bool)
    encoded = cmds.encode_1bpp_row(row)
    assert bytes(encoded) == b"\x00" * cmds.PRINTER_WIDTH_BYTES


def test_encode_1bpp_row_all_black():
    row = np.ones(cmds.PRINTER_WIDTH_PIXELS, dtype=bool)
    encoded = cmds.encode_1bpp_row(row)
    assert bytes(encoded) == b"\xff" * cmds.PRINTER_WIDTH_BYTES


def test_encode_1bpp_row_lsb_is_leftmost_pixel():
    # Per PROTOCOL.md: LSB of each byte is the LEFTMOST pixel of its 8-pixel chunk.
    row = np.zeros(cmds.PRINTER_WIDTH_PIXELS, dtype=bool)
    row[0] = True  # leftmost pixel of the row -> LSB of byte 0
    encoded = cmds.encode_1bpp_row(row)
    expected = bytearray(cmds.PRINTER_WIDTH_BYTES)
    expected[0] = 0x01
    assert bytes(encoded) == bytes(expected)


def test_encode_1bpp_row_msb_is_rightmost_in_chunk():
    row = np.zeros(cmds.PRINTER_WIDTH_PIXELS, dtype=bool)
    row[7] = True  # rightmost pixel of the first 8-pixel chunk -> MSB of byte 0
    encoded = cmds.encode_1bpp_row(row)
    expected = bytearray(cmds.PRINTER_WIDTH_BYTES)
    expected[0] = 0x80
    assert bytes(encoded) == bytes(expected)


def test_encode_1bpp_row_wrong_width_raises():
    row = np.zeros(cmds.PRINTER_WIDTH_PIXELS - 1, dtype=bool)
    with pytest.raises(ValueError):
        cmds.encode_1bpp_row(row)


def test_prepare_image_data_buffer_pads_to_min():
    # 10 lines * 48 bytes = 480, well below the 4320-byte minimum -> must pad.
    img = np.zeros((10, cmds.PRINTER_WIDTH_PIXELS), dtype=bool)
    buf = cmds.prepare_image_data_buffer(img)
    assert len(buf) == cmds.MIN_DATA_BYTES
    assert all(b == 0 for b in buf)


def test_prepare_image_data_buffer_no_pad_when_already_long():
    # 100 lines * 48 = 4800 > MIN_DATA_BYTES, no padding expected.
    n_lines = 100
    img = np.zeros((n_lines, cmds.PRINTER_WIDTH_PIXELS), dtype=bool)
    buf = cmds.prepare_image_data_buffer(img)
    assert len(buf) == n_lines * cmds.PRINTER_WIDTH_BYTES


def test_prepare_image_data_buffer_wrong_width_raises():
    img = np.zeros((10, cmds.PRINTER_WIDTH_PIXELS - 1), dtype=bool)
    with pytest.raises(ValueError):
        cmds.prepare_image_data_buffer(img)


# --- stdin (`-`) support --------------------------------------------------


class _BytesStdin:
    """Minimal sys.stdin stub: exposes a `.buffer` with raw bytes."""

    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", arr)
    assert ok
    return encoded.tobytes()


def test_read_img_grayscale_from_stdin(monkeypatch):
    src = np.full((20, 30), 200, dtype=np.uint8)
    src[:, 15:] = 50
    png = _png_bytes(src)
    monkeypatch.setattr("sys.stdin", _BytesStdin(png))
    out = read_img_grayscale("-")
    assert out.shape == (20, 30)
    assert out.dtype == np.uint8
    assert out[0, 0] >= 150
    assert out[0, -1] <= 100


def test_read_img_grayscale_stdin_empty_raises(monkeypatch):
    monkeypatch.setattr("sys.stdin", _BytesStdin(b""))
    with pytest.raises(RuntimeError, match="No image data"):
        read_img_grayscale("-")


def test_read_img_grayscale_stdin_undecodable_raises(monkeypatch):
    monkeypatch.setattr("sys.stdin", _BytesStdin(b"not an image"))
    with pytest.raises(RuntimeError, match="Could not decode image"):
        read_img_grayscale("-")


def test_dither_png_in_place_binarizes(tmp_path):
    """`dither_png_in_place` should turn a midtone PNG into ~1-bit pixels."""
    from catprinter.img import dither_png_in_place

    src = tmp_path / "grey.png"
    h, w = 32, 64
    grey = np.full((h, w), 128, dtype=np.uint8)
    cv2.imwrite(str(src), grey)

    dither_png_in_place(src, "floyd-steinberg")

    out = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    assert out is not None
    assert out.shape == (h, w), "dimensions must be preserved"
    # After dithering a uniform grey field, almost every pixel should be
    # either 0 or 255 - no broad band of in-betweens.
    midtone = np.count_nonzero((out > 16) & (out < 239))
    assert midtone == 0, f"expected fully-binarized output, got {midtone} midtones"


def test_dither_png_in_place_none_is_noop(tmp_path):
    """`algo='none'` should leave the file's bytes untouched."""
    from catprinter.img import dither_png_in_place

    src = tmp_path / "grey.png"
    cv2.imwrite(str(src), np.full((8, 8), 128, dtype=np.uint8))
    before = src.read_bytes()
    dither_png_in_place(src, "none")
    after = src.read_bytes()
    assert before == after


def test_read_img_from_stdin_full_pipeline(monkeypatch):
    # Solid-grey image at the printer's exact width: with `none` binarization
    # this should round-trip into a bool array of shape (h, PRINTER_WIDTH).
    h = 5
    src = np.full((h, cmds.PRINTER_WIDTH_PIXELS), 255, dtype=np.uint8)
    monkeypatch.setattr("sys.stdin", _BytesStdin(_png_bytes(src)))
    out = read_img("-", cmds.PRINTER_WIDTH_PIXELS, "none")
    assert out.shape == (h, cmds.PRINTER_WIDTH_PIXELS)
    assert out.dtype == bool
