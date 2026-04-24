"""Tests for the image-encoding helpers in catprinter.cmds."""
import numpy as np
import pytest

from catprinter import cmds


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
