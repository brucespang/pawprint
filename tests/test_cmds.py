"""Tests for catprinter.cmds: command builders, CRC, response parsers."""
import pytest

from catprinter import cmds


# --- CRC8 ---


def test_crc8_empty_is_zero():
    assert cmds.calculate_crc8(b"") == 0x00


def test_crc8_single_zero_byte():
    # CRC of \x00 with init=0 must be 0
    assert cmds.calculate_crc8(b"\x00") == 0x00


def test_crc8_known_vectors():
    # These are computed via the CRC8 lookup table embedded in cmds.py.
    # Pinning them here so we notice if the table or algorithm ever drifts.
    assert cmds.calculate_crc8(b"\x5d") == 0x94
    assert cmds.calculate_crc8(b"\x01\x02\x03") == 0x48


# --- Packet structure helpers ---


def _assert_envelope(packet: bytes, expected_cmd_id: int, expected_payload: bytes):
    """Assert preamble, cmd id, fixed byte, length LE, payload, CRC, footer."""
    assert packet[0] == 0x22, f"preamble[0]: {packet[0]:#x}"
    assert packet[1] == 0x21, f"preamble[1]: {packet[1]:#x}"
    assert packet[2] == expected_cmd_id, f"cmd id: {packet[2]:#x}"
    assert packet[3] == 0x00, f"fixed: {packet[3]:#x}"
    payload_len = int.from_bytes(packet[4:6], "little")
    assert payload_len == len(expected_payload)
    assert bytes(packet[6 : 6 + payload_len]) == expected_payload
    assert packet[6 + payload_len] == cmds.calculate_crc8(expected_payload)
    assert packet[6 + payload_len + 1] == 0xFF, "footer must be 0xFF"
    assert len(packet) == 8 + payload_len


# --- Command builders ---


def test_cmd_get_status():
    pkt = bytes(cmds.cmd_get_status())
    _assert_envelope(pkt, cmds.CommandIDs.GET_STATUS, b"\x00")


def test_cmd_set_intensity_in_range():
    pkt = bytes(cmds.cmd_set_intensity(0x5D))
    _assert_envelope(pkt, cmds.CommandIDs.PRINT_INTENSITY, bytes([0x5D]))


def test_cmd_set_intensity_clamps_low():
    pkt = bytes(cmds.cmd_set_intensity(-5))
    _assert_envelope(pkt, cmds.CommandIDs.PRINT_INTENSITY, bytes([0x00]))


def test_cmd_set_intensity_clamps_high():
    pkt = bytes(cmds.cmd_set_intensity(999))
    _assert_envelope(pkt, cmds.CommandIDs.PRINT_INTENSITY, bytes([0xFF]))


def test_cmd_print_request_monochrome():
    pkt = bytes(cmds.cmd_print_request(100, cmds.PrintModes.MONOCHROME))
    expected_payload = bytes([100, 0x00, 0x30, cmds.PrintModes.MONOCHROME])
    _assert_envelope(pkt, cmds.CommandIDs.PRINT, expected_payload)


def test_cmd_print_request_large_line_count():
    # 384 lines exercises the high byte of the LE length
    pkt = bytes(cmds.cmd_print_request(384, cmds.PrintModes.MONOCHROME))
    expected_payload = bytes([0x80, 0x01, 0x30, cmds.PrintModes.MONOCHROME])
    _assert_envelope(pkt, cmds.CommandIDs.PRINT, expected_payload)


def test_cmd_flush():
    pkt = bytes(cmds.cmd_flush())
    _assert_envelope(pkt, cmds.CommandIDs.PRINT_DATA_FLUSH, b"\x00")


def test_cmd_get_battery():
    pkt = bytes(cmds.cmd_get_battery())
    _assert_envelope(pkt, cmds.CommandIDs.BATTERY_LEVEL, b"\x00")


def test_cmd_get_version():
    pkt = bytes(cmds.cmd_get_version())
    _assert_envelope(pkt, cmds.CommandIDs.GET_VERSION, b"\x00")


def test_cmd_cancel():
    pkt = bytes(cmds.cmd_cancel())
    _assert_envelope(pkt, cmds.CommandIDs.CANCEL_PRINT, b"\x00")


# --- parse_status ---


def _build_status_payload(
    state: int = 0x00,
    battery: int = 75,
    temperature: int = 20,
    is_ok: bool = True,
    error_code: int = 0,
    extra: bytes = b"",
) -> bytes:
    """Assemble an A1 status payload with the documented byte indices."""
    p = bytearray(14)
    p[6] = state
    p[9] = battery
    p[10] = temperature
    p[12] = 0 if is_ok else 1
    p[13] = error_code
    return bytes(p) + extra


def test_parse_status_ok_standby():
    payload = _build_status_payload(state=0x00, battery=80, temperature=22)
    s = cmds.parse_status(payload)
    assert s.is_ok is True
    assert s.is_standby is True
    assert s.state == cmds.PrinterStates.STANDBY
    assert s.battery_pct == 80
    assert s.temperature == 22
    assert s.error_code is None


def test_parse_status_ok_printing():
    payload = _build_status_payload(state=cmds.PrinterStates.PRINTING)
    s = cmds.parse_status(payload)
    assert s.is_ok is True
    assert s.is_standby is False
    assert s.state == cmds.PrinterStates.PRINTING


def test_parse_status_no_paper():
    payload = _build_status_payload(
        is_ok=False, error_code=cmds.PrinterErrorCodes.NO_PAPER_1
    )
    s = cmds.parse_status(payload)
    assert s.is_ok is False
    assert s.error_code == cmds.PrinterErrorCodes.NO_PAPER_1


def test_parse_status_overheated():
    payload = _build_status_payload(
        is_ok=False, error_code=cmds.PrinterErrorCodes.OVERHEATED
    )
    s = cmds.parse_status(payload)
    assert s.is_ok is False
    assert s.error_code == cmds.PrinterErrorCodes.OVERHEATED


def test_parse_status_low_battery():
    payload = _build_status_payload(
        is_ok=False, error_code=cmds.PrinterErrorCodes.LOW_BATTERY
    )
    s = cmds.parse_status(payload)
    assert s.is_ok is False
    assert s.error_code == cmds.PrinterErrorCodes.LOW_BATTERY


def test_parse_status_extremely_short_payload_returns_not_ok():
    # Less than 5 bytes can't carry even the short-format state/battery/temp;
    # we should still get a structured response rather than crash.
    s = cmds.parse_status(b"\x00\x01\x02")
    assert s.is_ok is False
    assert s.error_code is None
    assert s.raw == b"\x00\x01\x02"


def test_parse_status_short_format_standby_observed_in_wild():
    # MXW01 firmware 1.9.3.1.1 emits a 10-byte status payload instead of the
    # documented 13+. Layout (confirmed by comparing Standby vs Printing):
    #   byte 0: state, byte 3: battery %, byte 4: temperature °C
    payload = bytes.fromhex("0000006410000000c400")
    s = cmds.parse_status(payload)
    assert s.is_ok is True
    assert s.state == cmds.PrinterStates.STANDBY
    assert s.is_standby is True
    assert s.battery_pct == 100
    assert s.temperature == 16
    assert s.error_code is None


def test_parse_status_short_format_printing_observed_in_wild():
    # Captured during an actual print job - state byte (0) is 0x02 (Printing),
    # battery has sagged under load, head temp has risen.
    payload = bytes.fromhex("0200005215000000c400")
    s = cmds.parse_status(payload)
    assert s.is_ok is True
    assert s.state == cmds.PrinterStates.PRINTING
    assert s.is_standby is False
    assert s.battery_pct == 0x52
    assert s.temperature == 0x15


def test_parse_status_describe_does_not_crash():
    payload = _build_status_payload()
    assert "Standby" in cmds.parse_status(payload).describe()


# --- error_for_code ---


def test_error_for_code_no_paper_alternate():
    err = cmds.error_for_code(cmds.PrinterErrorCodes.NO_PAPER_9)
    assert isinstance(err, cmds.NoPaperError)


def test_error_for_code_overheated():
    err = cmds.error_for_code(cmds.PrinterErrorCodes.OVERHEATED)
    assert isinstance(err, cmds.OverheatedError)


def test_error_for_code_low_battery():
    err = cmds.error_for_code(cmds.PrinterErrorCodes.LOW_BATTERY)
    assert isinstance(err, cmds.LowBatteryError)


def test_error_for_code_unknown_falls_back_to_generic():
    err = cmds.error_for_code(0x42)
    assert isinstance(err, cmds.PrinterError)
    assert err.error_code == 0x42


def test_printer_error_subclasses_inherit_from_runtime_error():
    # CLI catches RuntimeError as well as PrinterError; make sure the chain holds.
    assert issubclass(cmds.NoPaperError, cmds.PrinterError)
    assert issubclass(cmds.PrinterError, RuntimeError)


# --- parse_battery ---


def test_parse_battery_simple():
    assert cmds.parse_battery(b"\x55") == 0x55


def test_parse_battery_empty_raises():
    with pytest.raises(ValueError):
        cmds.parse_battery(b"")


# --- parse_version ---


def test_parse_version_with_nul_and_type_byte():
    payload = b"1.2.3\x00\xff\x01"
    v = cmds.parse_version(payload)
    assert v.version == "1.2.3"
    assert v.type_byte == 0x01


def test_parse_version_without_nul_terminator():
    payload = b"V0.9"
    v = cmds.parse_version(payload)
    assert v.version == "V0.9"
    # No NUL terminator means we can't reliably split off a type byte.
    assert v.type_byte is None
