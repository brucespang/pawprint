"""Tests for catprinter.ble flow helpers using a fake BleakClient.

Covers the do_print / do_status / do_cancel sequences end-to-end without any
real Bluetooth hardware, plus regression tests for the wait_for race and the
state-guard / cancel-on-cleanup behaviors.
"""
import asyncio
from typing import Any, Callable, List, Optional, Tuple

import pytest

from catprinter import cmds, ble


# --- Fake BLE plumbing -------------------------------------------------------


class FakeChar:
    def __init__(self, char_uuid: str):
        self.uuid = char_uuid


class FakeClient:
    """Records writes and dispatches injected notifications to a registered receiver."""

    def __init__(self):
        self.writes: List[Tuple[str, bytes, bool]] = []
        self.is_connected = True
        self.mtu_size = 256
        self.write_handlers: List[Callable[[str, bytes], None]] = []

    async def write_gatt_char(self, char_uuid: str, data: Any, response: bool = False):
        # Bleak accepts bytes / bytearray; coerce to bytes for stable assertions.
        data_bytes = bytes(data)
        self.writes.append((char_uuid, data_bytes, response))
        # Run any registered "when this command is written, respond like so" hooks.
        for handler in list(self.write_handlers):
            handler(char_uuid, data_bytes)


def make_session() -> Tuple[ble.PrinterSession, FakeClient, Callable[[int, bytes], None]]:
    """Build a PrinterSession backed by a FakeClient.

    Returns (session, fake_client, inject_notification(cmd_id, payload)).
    The inject helper builds a well-formed notification packet and feeds it
    through the same notification_receiver_factory used by production code.
    """
    notification_state = ble._new_notification_state()
    receiver = ble.notification_receiver_factory(notification_state)

    client = FakeClient()
    session = ble.PrinterSession(
        client=client,  # type: ignore[arg-type]
        control_char=FakeChar(cmds.CONTROL_WRITE_UUID),
        notify_char=FakeChar(cmds.NOTIFY_UUID),
        data_char=FakeChar(cmds.DATA_WRITE_UUID),
        notify_char_uuid=cmds.NOTIFY_UUID,
        notification_state=notification_state,
    )

    def inject(cmd_id: int, payload: bytes) -> None:
        # The production receiver expects the same envelope we use to send.
        packet = bytes(cmds.create_command(cmd_id, payload))
        receiver(0, bytearray(packet))

    return session, client, inject


def _writes_to_control(client: FakeClient) -> List[bytes]:
    return [data for (uuid, data, _) in client.writes if uuid == cmds.CONTROL_WRITE_UUID]


def _writes_to_data(client: FakeClient) -> List[bytes]:
    return [data for (uuid, data, _) in client.writes if uuid == cmds.DATA_WRITE_UUID]


def _control_cmd_ids(client: FakeClient) -> List[int]:
    """Sequence of command IDs sent on the control characteristic."""
    return [w[2] for w in _writes_to_control(client)]


def _build_status_payload(
    state: int = cmds.PrinterStates.STANDBY,
    battery: int = 80,
    temperature: int = 22,
    is_ok: bool = True,
    error_code: int = 0,
) -> bytes:
    p = bytearray(14)
    p[6] = state
    p[9] = battery
    p[10] = temperature
    p[12] = 0 if is_ok else 1
    p[13] = error_code
    return bytes(p)


# --- send_and_wait race regression ------------------------------------------


async def test_send_and_wait_handles_response_landing_before_lock():
    """Regression for the wait_for_notification pop-after-send race.

    Simulates the case where the printer response is fully processed before
    the waiter manages to acquire the condition lock (e.g. the receiver
    callback drained on a thread / earlier event-loop tick). In the OLD
    `wait_for_notification`, the pop-after-acquire would discard that
    payload and then hang waiting for a second response that never comes.
    In the new `send_and_wait`, the slot is armed *before* the write, so a
    landed payload is preserved.

    To make the race deterministic we bypass the create_task'd notification
    handler and write directly into `received` from inside write_gatt_char.
    """
    session, client, _inject = make_session()

    expected_cmd = cmds.CommandIDs.GET_STATUS
    response_payload = _build_status_payload()

    def respond(_char_uuid, data):
        if len(data) >= 3 and data[2] == expected_cmd:
            session.notification_state["received"][expected_cmd] = response_payload
    client.write_handlers.append(respond)

    payload = await asyncio.wait_for(
        ble.send_and_wait(
            session, cmds.cmd_get_status(), expected_cmd, timeout=1.0
        ),
        timeout=2.0,
    )
    assert payload == response_payload


# --- do_status / do_version / do_battery -----------------------------------


async def test_do_status_returns_parsed_info():
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) >= 3 and data[2] == cmds.CommandIDs.GET_STATUS:
            inject(cmds.CommandIDs.GET_STATUS, _build_status_payload(battery=42))
    client.write_handlers.append(respond)

    info = await ble.do_status(session)
    assert info.is_ok and info.is_standby
    assert info.battery_pct == 42


async def test_do_status_timeout_raises_printer_error(monkeypatch):
    monkeypatch.setattr(ble, "NOTIFICATION_TIMEOUT_S", 0.05)
    session, _client, _inject = make_session()
    with pytest.raises(cmds.PrinterError):
        await ble.do_status(session)


async def test_do_version_decodes_payload():
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) >= 3 and data[2] == cmds.CommandIDs.GET_VERSION:
            inject(cmds.CommandIDs.GET_VERSION, b"V1.2\x00\x00\x01")
    client.write_handlers.append(respond)

    v = await ble.do_version(session)
    assert v.version == "V1.2"
    assert v.type_byte == 0x01


async def test_do_battery_returns_byte():
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) >= 3 and data[2] == cmds.CommandIDs.BATTERY_LEVEL:
            inject(cmds.CommandIDs.BATTERY_LEVEL, b"\x5a")
    client.write_handlers.append(respond)

    battery = await ble.do_battery(session)
    assert battery == 0x5A


# --- do_cancel -------------------------------------------------------------


async def test_do_cancel_sends_AC_then_A1_and_returns_status():
    session, client, inject = make_session()

    # Pre-cancel A1: report Printing. Post-cancel A1: report Standby.
    a1_responses = iter(
        [
            _build_status_payload(state=cmds.PrinterStates.PRINTING),
            _build_status_payload(state=cmds.PrinterStates.STANDBY),
        ]
    )

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        if data[2] == cmds.CommandIDs.GET_STATUS:
            inject(cmds.CommandIDs.GET_STATUS, next(a1_responses))
        # AC has no response per PROTOCOL.md; nothing to inject.

    client.write_handlers.append(respond)

    post = await ble.do_cancel(session)
    assert post.is_standby

    cmd_ids = _control_cmd_ids(client)
    # Order should be: pre-cancel A1, then AC, then post-cancel A1.
    assert cmd_ids == [
        cmds.CommandIDs.GET_STATUS,
        cmds.CommandIDs.CANCEL_PRINT,
        cmds.CommandIDs.GET_STATUS,
    ]


# --- do_print: happy path --------------------------------------------------


async def test_do_print_happy_path():
    session, client, inject = make_session()

    image_buffer = bytes(cmds.MIN_DATA_BYTES)  # all-white, padded image
    expected_lines = cmds.MIN_DATA_BYTES // cmds.PRINTER_WIDTH_BYTES

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        cmd_id = data[2]
        if cmd_id == cmds.CommandIDs.GET_STATUS:
            inject(cmds.CommandIDs.GET_STATUS, _build_status_payload())
        elif cmd_id == cmds.CommandIDs.PRINT:
            inject(cmds.CommandIDs.PRINT, b"\x00")  # 0 = OK
        elif cmd_id == cmds.CommandIDs.PRINT_DATA_FLUSH:
            # AA arrives spontaneously after the data flush completes.
            inject(cmds.CommandIDs.PRINT_COMPLETE, b"\x00")

    client.write_handlers.append(respond)

    await ble.do_print(session, image_buffer, intensity=0x5D)

    cmd_ids = _control_cmd_ids(client)
    # A2 set-intensity, A1 status, A9 print req, AD flush. No AC because we
    # received AA cleanly.
    assert cmd_ids == [
        cmds.CommandIDs.PRINT_INTENSITY,
        cmds.CommandIDs.GET_STATUS,
        cmds.CommandIDs.PRINT,
        cmds.CommandIDs.PRINT_DATA_FLUSH,
    ]

    data_writes = _writes_to_data(client)
    assert len(data_writes) == expected_lines
    assert all(len(chunk) == cmds.PRINTER_WIDTH_BYTES for chunk in data_writes)
    assert b"".join(data_writes) == image_buffer


# --- do_print: state guard -------------------------------------------------


async def test_do_print_aborts_when_printer_busy():
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        if data[2] == cmds.CommandIDs.GET_STATUS:
            # Printer reports it's already printing -> we must abort before A9.
            inject(
                cmds.CommandIDs.GET_STATUS,
                _build_status_payload(state=cmds.PrinterStates.PRINTING),
            )

    client.write_handlers.append(respond)

    image_buffer = bytes(cmds.MIN_DATA_BYTES)
    with pytest.raises(cmds.PrinterError, match="Standby"):
        await ble.do_print(session, image_buffer, intensity=0x5D)

    # Crucially: A9 was never sent.
    assert cmds.CommandIDs.PRINT not in _control_cmd_ids(client)
    # And no image data was written.
    assert _writes_to_data(client) == []


# --- do_print: error surfacing ---------------------------------------------


async def test_do_print_raises_no_paper_error():
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        if data[2] == cmds.CommandIDs.GET_STATUS:
            inject(
                cmds.CommandIDs.GET_STATUS,
                _build_status_payload(
                    is_ok=False, error_code=cmds.PrinterErrorCodes.NO_PAPER_1
                ),
            )

    client.write_handlers.append(respond)

    image_buffer = bytes(cmds.MIN_DATA_BYTES)
    with pytest.raises(cmds.NoPaperError):
        await ble.do_print(session, image_buffer, intensity=0x5D)
    assert cmds.CommandIDs.PRINT not in _control_cmd_ids(client)


# --- do_print: cancel-on-cleanup ------------------------------------------


async def test_do_print_sends_cancel_when_AA_never_arrives(monkeypatch):
    """If we exit between A9-ack and AA, we must send AC so the printer doesn't
    get stuck waiting for more data."""
    # Make the AA wait time short so the test is fast.
    monkeypatch.setattr(ble, "PRINT_COMPLETE_BASE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ble, "PRINT_COMPLETE_LINES_PER_SEC", 1_000_000.0)
    monkeypatch.setattr(ble, "PACING_DELAY_S", 0.0)

    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        cmd_id = data[2]
        if cmd_id == cmds.CommandIDs.GET_STATUS:
            inject(cmds.CommandIDs.GET_STATUS, _build_status_payload())
        elif cmd_id == cmds.CommandIDs.PRINT:
            inject(cmds.CommandIDs.PRINT, b"\x00")
        # Deliberately do NOT respond to AD with AA; simulate a stuck print.

    client.write_handlers.append(respond)

    image_buffer = bytes(cmds.MIN_DATA_BYTES)
    await ble.do_print(session, image_buffer, intensity=0x5D)

    cmd_ids = _control_cmd_ids(client)
    # The cleanup AC must come after AD.
    assert cmds.CommandIDs.CANCEL_PRINT in cmd_ids
    assert cmd_ids.index(cmds.CommandIDs.CANCEL_PRINT) > cmd_ids.index(
        cmds.CommandIDs.PRINT_DATA_FLUSH
    )


async def test_do_print_no_cleanup_cancel_on_happy_path():
    """Sanity check: the cleanup AC is NOT sent when AA arrives."""
    session, client, inject = make_session()

    def respond(_char_uuid, data):
        if len(data) < 3:
            return
        cmd_id = data[2]
        if cmd_id == cmds.CommandIDs.GET_STATUS:
            inject(cmds.CommandIDs.GET_STATUS, _build_status_payload())
        elif cmd_id == cmds.CommandIDs.PRINT:
            inject(cmds.CommandIDs.PRINT, b"\x00")
        elif cmd_id == cmds.CommandIDs.PRINT_DATA_FLUSH:
            inject(cmds.CommandIDs.PRINT_COMPLETE, b"\x00")

    client.write_handlers.append(respond)

    image_buffer = bytes(cmds.MIN_DATA_BYTES)
    await ble.do_print(session, image_buffer, intensity=0x5D)
    assert cmds.CommandIDs.CANCEL_PRINT not in _control_cmd_ids(client)
