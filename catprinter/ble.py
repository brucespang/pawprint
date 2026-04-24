import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bleak import BleakClient, BleakScanner
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from . import cmds
from . import logger
from .ui import NULL_REPORTER, Reporter, countdown_step

PACING_DELAY_S = 0.015
NOTIFICATION_TIMEOUT_S = 7.0
# 35 lines/s was measured on an MXW01 (1380 lines in 38.9s). It controls two things:
# 1. How long to wait for the printer's AA "print complete" notification after
#    we send the AD flush. Computed as:
#      timeout = PRINT_COMPLETE_BASE_TIMEOUT_S + line_count / PRINT_COMPLETE_LINES_PER_SEC
#    i.e. a fixed slack budget plus the time the print itself should take. The
# 2. The user-facing "Estimated ~Xs" hint shown right before data send.
PRINT_COMPLETE_BASE_TIMEOUT_S = 15.0
PRINT_COMPLETE_LINES_PER_SEC = 35.0
POST_CANCEL_DELAY_S = 0.25

# Discovery: short first attempt so an impatient user can power the printer on
# without waiting too long, then a longer second attempt.
SCAN_FIRST_ATTEMPT_S = 4
SCAN_RETRY_ATTEMPT_S = 6
SCAN_ATTEMPTS = 2
# Kept for compatibility with code that imports SCAN_TIMEOUT_S.
SCAN_TIMEOUT_S = SCAN_FIRST_ATTEMPT_S + SCAN_RETRY_ATTEMPT_S

NotificationState = Dict[str, Any]


@dataclass
class PrinterSession:
    """A connected, notification-enabled BLE session with the printer."""

    client: BleakClient
    control_char: Any
    notify_char: Any
    data_char: Any
    notify_char_uuid: str
    notification_state: NotificationState = field(default_factory=dict)
    reporter: Reporter = field(default=NULL_REPORTER)


# --- Discovery ---


async def do_scan(
    timeout: float = SCAN_TIMEOUT_S,
    name: Optional[str] = None,
    on_found: Optional[Any] = None,
) -> List[BLEDevice]:
    """Scan for nearby MXW01 printers (devices advertising the main service).

    If `name` is given, only devices with that advertisement name are returned;
    otherwise all devices advertising the MXW01 service UUID are returned.

    If `on_found` is provided, it is invoked the FIRST time each matching
    device is seen (BLE advertisements repeat every few hundred ms, so this
    callback is de-duplicated by address). Use it to render results live as
    the scan progresses instead of waiting for the full timeout.
    """
    possible_service_uuids = {
        cmds.MAIN_SERVICE_UUID.lower(),
        cmds.MAIN_SERVICE_UUID_ALT.lower(),
    }
    matches: Dict[str, BLEDevice] = {}

    def detection(device: BLEDevice, ad: AdvertisementData) -> None:
        ad_uuids = {s.lower() for s in (ad.service_uuids or [])}
        service_match = bool(ad_uuids & possible_service_uuids)
        name_match = name is None or device.name == name
        if name is not None and not name_match:
            return
        if not service_match and name is None:
            return
        if device.address in matches:
            return
        matches[device.address] = device
        if on_found is not None:
            try:
                on_found(device)
            except Exception as e:
                logger.debug(f"on_found callback raised: {e}")

    async with BleakScanner(detection_callback=detection):
        await asyncio.sleep(timeout)

    return list(matches.values())


async def scan(name: Optional[str], timeout: int) -> BLEDevice:
    """Scan for a single matching printer; raise if none found."""
    autodiscover = not name
    if autodiscover:
        logger.debug("Auto-discovering printer (MXW01 service)...")
        possible_service_uuids = [
            cmds.MAIN_SERVICE_UUID.lower(),
            cmds.MAIN_SERVICE_UUID_ALT.lower(),
        ]
        filter_fn = lambda d, ad: any(
            uuid in [s.lower() for s in ad.service_uuids]
            for uuid in possible_service_uuids
        )
    else:
        logger.debug(f"Looking for a BLE device named {name!r}...")
        filter_fn = lambda d, ad: d.name == name

    device = await BleakScanner.find_device_by_filter(
        filter_fn,
        timeout=timeout,
    )
    if device is None:
        raise RuntimeError(
            "Unable to find printer; make sure it is turned on and in range"
        )
    logger.debug(f"Found device: name={device.name!r}, address={device.address}")
    return device


async def scan_with_retry(
    name: Optional[str],
    reporter: "Reporter",
    attempts: int = SCAN_ATTEMPTS,
    first_timeout: float = SCAN_FIRST_ATTEMPT_S,
    retry_timeout: float = SCAN_RETRY_ATTEMPT_S,
) -> BLEDevice:
    """Scan with one retry. Updates the reporter step on each attempt."""
    label = f"named {name!r}" if name else "printer"
    last_err: Optional[RuntimeError] = None
    for i in range(1, attempts + 1):
        timeout = first_timeout if i == 1 else retry_timeout

        def fmt(remaining: float, _i: int = i) -> str:
            if _i == 1:
                return f"Discovering {label} ({remaining:.0f}s left)"
            return (
                f"Still no {label}; turn it on if needed... "
                f"(try {_i}/{attempts}, {remaining:.0f}s left)"
            )

        async with countdown_step(reporter, fmt, timeout):
            try:
                return await scan(name, timeout=int(timeout))
            except RuntimeError as e:
                last_err = e
    assert last_err is not None
    raise last_err


async def get_device_address(device: Optional[str]):
    if device:
        with contextlib.suppress(ValueError):
            return str(uuid.UUID(device))
        if device.count(":") == 5 and device.replace(":", "").isalnum():
            return device
        logger.debug(
            f"Treating -d {device!r} as an advertised name (not a UUID/MAC); "
            "scanning for it."
        )

    # Note: this code path no longer drives the user-facing reporter. New
    # callers should use scan_with_retry directly. Kept for any external use.
    return await scan(device, timeout=SCAN_TIMEOUT_S)


# --- Notification plumbing ---


def notification_receiver_factory(notification_state: NotificationState):
    async def update_notification_state(cmd_id: int, payload: bytes):
        async with notification_state["condition"]:
            notification_state["received"][cmd_id] = payload
            notification_state["condition"].notify_all()
            logger.debug(f"Notified waiters for command 0x{cmd_id:02X}")

    def notification_receiver(sender: int, data: bytearray):
        # Basic check for header and minimum possible length (header, cmd, fixed, len)
        if len(data) >= 6 and data[0] == 0x22 and data[1] == 0x21:
            cmd_id = data[2]
            try:
                payload_len = int.from_bytes(data[4:6], "little")
                expected_payload_end_idx = 6 + payload_len  # Index after the payload

                # Check if we received AT LEAST enough data for the declared payload
                if len(data) >= expected_payload_end_idx:
                    payload = bytes(data[6:expected_payload_end_idx])

                    # --- Optional: Perform CRC/Footer check only if data is long enough ---
                    expected_total_len = 8 + payload_len  # Includes CRC and Footer
                    if len(data) >= expected_total_len:
                        crc_received = data[expected_payload_end_idx]
                        footer = data[expected_payload_end_idx + 1]
                        crc_calculated = cmds.calculate_crc8(payload)
                        if crc_received != crc_calculated:
                            logger.debug(
                                f"CRC mismatch for 0x{cmd_id:02X}. Got {crc_received:02X}, expected {crc_calculated:02X}. Payload: {payload.hex()}"
                            )
                        if footer != 0xFF:
                            logger.debug(
                                f"Invalid footer for 0x{cmd_id:02X}. Got {footer:02X}, expected FF."
                            )
                    elif len(data) > expected_payload_end_idx:
                        # Some firmwares (e.g. version 1.9.3.1.1) consistently
                        # omit the trailing CRC + footer. That's expected, log
                        # at debug only.
                        logger.debug(
                            f"Notification 0x{cmd_id:02X} missing CRC/footer (got {len(data)} bytes, expected {expected_total_len}). Skipping integrity check."
                        )
                    # else: Data ends exactly after payload, CRC/Footer definitely missing.

                    # --- Process the payload regardless of CRC/Footer issues ---
                    logger.debug(
                        f"Received notification 0x{cmd_id:02X}: {payload.hex()}"
                    )
                    asyncio.create_task(update_notification_state(cmd_id, payload))

                else:
                    # This means we didn't even get the full payload bytes
                    logger.warning(
                        f"Received notification too short for declared payload. Cmd: 0x{cmd_id:02X}, Declared len: {payload_len}, Actual len: {len(data)}, Needed for payload: {expected_payload_end_idx}"
                    )

            except IndexError:
                logger.error(
                    f"Error parsing notification - IndexError. Data: {data.hex()}"
                )
            except Exception as e:
                logger.error(f"Error parsing notification: {e}. Data: {data.hex()}")
        else:
            logger.debug(
                f"Ignoring unexpected/non-MXW01 notification format or too short: {data.hex()}"
            )

    return notification_receiver


def _new_notification_state() -> NotificationState:
    return {
        "received": {},
        "condition": asyncio.Condition(),
    }


async def arm(session: PrinterSession, expected_cmd_id: int) -> None:
    """Clear any stale notification for `expected_cmd_id`.

    Must be called BEFORE writing the request command, so a fast response is
    not silently discarded by a later pop.
    """
    cond = session.notification_state["condition"]
    async with cond:
        session.notification_state["received"].pop(expected_cmd_id, None)


async def wait_for(
    session: PrinterSession, expected_cmd_id: int, timeout: float
) -> Optional[bytes]:
    """Wait for a notification of `expected_cmd_id`. Caller must have armed first."""
    cond = session.notification_state["condition"]
    received = session.notification_state["received"]
    async with cond:
        try:
            await asyncio.wait_for(
                cond.wait_for(lambda: expected_cmd_id in received),
                timeout=timeout,
            )
            payload = received.pop(expected_cmd_id)
            logger.debug(f"Waited and received notification 0x{expected_cmd_id:02X}")
            return payload
        except asyncio.TimeoutError:
            logger.debug(
                f"Timeout waiting for notification 0x{expected_cmd_id:02X} after {timeout}s"
            )
            return None


async def send_and_wait(
    session: PrinterSession,
    cmd_bytes: bytes,
    expected_cmd_id: int,
    timeout: float,
) -> Optional[bytes]:
    """Arm + write + wait, race-free."""
    await arm(session, expected_cmd_id)
    await session.client.write_gatt_char(
        session.control_char.uuid, cmd_bytes, response=False
    )
    return await wait_for(session, expected_cmd_id, timeout)


# --- Connection lifecycle ---


@asynccontextmanager
async def connected_printer(
    device: Optional[str],
    reporter: Reporter = NULL_REPORTER,
):
    """Resolve the device, connect, find the MXW01 service+characteristics,
    enable notifications, and yield a `PrinterSession`. Cleans up on exit.

    Drives the user-visible `Discovering`/`Connecting` step lines through the
    given reporter, and finishes with a permanent `✓ <name> (<address>)`
    line so subsequent step calls (from do_print etc.) overwrite cleanly.
    """
    autodiscovered = not device
    try:
        # Auto-discovery returns a BLEDevice (with a name); a literal address
        # just round-trips through get_device_address. We need the name when
        # available so the final line reads nicely.
        if autodiscovered:
            ble_device = await scan_with_retry(None, reporter)
            address = ble_device.address
            display_name = ble_device.name or "MXW01"
        else:
            reporter.step(f"Resolving {device}")
            address = await get_device_address(device)
            # If the user passed an address (UUID/MAC), `device` matches
            # `address` modulo case (UUIDs get normalised). In that case we
            # don't have a name to show, so fall back to the model name.
            looks_like_address = device.lower() == str(address).lower()
            display_name = "MXW01" if looks_like_address else device
    except RuntimeError as e:
        reporter.error(str(e))
        raise

    reporter.step(f"Connecting to {display_name}")
    notification_state = _new_notification_state()
    receive_notification = notification_receiver_factory(notification_state)

    notify_char_uuid: Optional[str] = None
    try:
        async with BleakClient(address, timeout=20.0) as client:
            logger.debug(f"Connected: {client.is_connected}; MTU: {client.mtu_size}")

            service = None
            possible_service_uuids = [
                cmds.MAIN_SERVICE_UUID.lower(),
                cmds.MAIN_SERVICE_UUID_ALT.lower(),
            ]
            for s in client.services:
                if s.uuid.lower() in possible_service_uuids:
                    service = s
                    logger.debug(f"Found service: {s.uuid}")
                    break
            if not service:
                raise BleakError(
                    f"Service {cmds.MAIN_SERVICE_UUID} (or alternative) not found."
                )

            control_char = service.get_characteristic(cmds.CONTROL_WRITE_UUID)
            notify_char = service.get_characteristic(cmds.NOTIFY_UUID)
            data_char = service.get_characteristic(cmds.DATA_WRITE_UUID)

            if not all([control_char, notify_char, data_char]):
                missing = [
                    char_uuid
                    for char_uuid, char in [
                        (cmds.CONTROL_WRITE_UUID, control_char),
                        (cmds.NOTIFY_UUID, notify_char),
                        (cmds.DATA_WRITE_UUID, data_char),
                    ]
                    if char is None
                ]
                raise BleakError(f"Missing required characteristics: {missing}")

            notify_char_uuid = notify_char.uuid
            logger.debug(f"Starting notifications on {notify_char.uuid}")
            await client.start_notify(notify_char.uuid, receive_notification)

            session = PrinterSession(
                client=client,
                control_char=control_char,
                notify_char=notify_char,
                data_char=data_char,
                notify_char_uuid=notify_char.uuid,
                notification_state=notification_state,
                reporter=reporter,
            )

            # The headline result of the discovery+connect dance. When the
            # user auto-discovered, append a grey `-d <addr>` hint so the
            # opaque UUID/MAC is recognisable as something they can pass back
            # in next time. When they explicitly passed -d, skip the hint
            # (they already know it).
            hint = f"-d {address}" if autodiscovered else None
            reporter.done(f"Connected to {display_name}", hint=hint)

            try:
                yield session
            finally:
                if client.is_connected and notify_char_uuid:
                    try:
                        logger.debug("Stopping notifications")
                        await client.stop_notify(notify_char_uuid)
                    except Exception as e:
                        logger.debug(f"Error stopping notifications: {e}")
                logger.debug("Disconnecting")
    finally:
        logger.debug("BLE operation finished.")


# --- Per-operation helpers ---


async def do_status(session: PrinterSession) -> cmds.StatusInfo:
    payload = await send_and_wait(
        session,
        cmds.cmd_get_status(),
        cmds.CommandIDs.GET_STATUS,
        NOTIFICATION_TIMEOUT_S,
    )
    if payload is None:
        raise cmds.PrinterError("Timed out waiting for status response (A1)")
    return cmds.parse_status(payload)


async def do_version(session: PrinterSession) -> cmds.VersionInfo:
    payload = await send_and_wait(
        session,
        cmds.cmd_get_version(),
        cmds.CommandIDs.GET_VERSION,
        NOTIFICATION_TIMEOUT_S,
    )
    if payload is None:
        raise cmds.PrinterError("Timed out waiting for version response (B1)")
    return cmds.parse_version(payload)


async def do_battery(session: PrinterSession) -> int:
    payload = await send_and_wait(
        session,
        cmds.cmd_get_battery(),
        cmds.CommandIDs.BATTERY_LEVEL,
        NOTIFICATION_TIMEOUT_S,
    )
    if payload is None:
        raise cmds.PrinterError("Timed out waiting for battery response (AB)")
    return cmds.parse_battery(payload)


async def do_cancel(session: PrinterSession) -> cmds.StatusInfo:
    """Send AC cancel-print, then poll A1 to confirm the printer returned to standby."""
    pre_state: Optional[str] = None
    try:
        pre = await do_status(session)
        pre_state = _format_state(pre)
    except cmds.PrinterError as e:
        logger.debug(f"Could not read pre-cancel status: {e}")

    if pre_state:
        session.reporter.step(f"Cancelling print (was {pre_state})")
    else:
        session.reporter.step("Cancelling print")

    await session.client.write_gatt_char(
        session.control_char.uuid, cmds.cmd_cancel(), response=False
    )
    await asyncio.sleep(POST_CANCEL_DELAY_S)

    post = await do_status(session)
    post_state = _format_state(post)
    if post.is_standby:
        session.reporter.done(f"Cancelled (now {post_state})")
    else:
        session.reporter.warn(
            f"Cancel sent but printer is still {post_state}; you may need to power-cycle it."
        )
    return post


async def do_print(
    session: PrinterSession, image_data_buffer: bytes, intensity: int
) -> Optional[float]:
    """Run the full print sequence: intensity -> status -> A9 -> data -> AD -> AA.

    Aborts before A9 if the printer is not in Standby or is reporting an error,
    and best-effort sends AC if we exit between A9-ack and AA so the printer
    isn't left waiting for more data.

    Returns the wall-clock seconds elapsed from the A9 acknowledgment to the
    AA "print complete" notification (i.e., the end-to-end "actively
    printing" time, including BLE data transfer). Returns None if the AA
    notification never arrived or the print aborted before A9.
    """
    line_count = len(image_data_buffer) // cmds.PRINTER_WIDTH_BYTES
    reporter = session.reporter

    reporter.step(f"Setting intensity 0x{intensity:02X}")
    await session.client.write_gatt_char(
        session.control_char.uuid, cmds.cmd_set_intensity(intensity), response=False
    )
    await asyncio.sleep(0.1)

    reporter.step("Checking printer status")
    status = await do_status(session)
    if not status.is_ok:
        if status.error_code is not None:
            raise cmds.error_for_code(status.error_code, status.raw)
        raise cmds.PrinterError(f"Printer reported not OK: {status.raw.hex()}")
    if status.state != cmds.PrinterStates.STANDBY:
        raise cmds.PrinterError(
            f"Printer is not in Standby (state=0x{status.state:02X}); aborting before A9. "
            "Run `pawprint cancel` or power-cycle the printer and try again."
        )

    print_in_flight = False
    try:
        reporter.step(f"Requesting print of {line_count} lines")
        print_req_payload = await send_and_wait(
            session,
            cmds.cmd_print_request(line_count, cmds.PrintModes.MONOCHROME),
            cmds.CommandIDs.PRINT,
            NOTIFICATION_TIMEOUT_S,
        )
        if print_req_payload is None:
            raise cmds.PrinterError(
                "Timed out waiting for print request acknowledgment (A9)"
            )
        if not (len(print_req_payload) > 0 and print_req_payload[0] == 0):
            raise cmds.PrinterError(
                f"Printer rejected print request (A9): {print_req_payload.hex()}"
            )
        print_in_flight = True
        # Start the wall-clock the moment the printer ack'd A9. This is the
        # cleanest "print started" signal we have; everything before it is
        # setup chatter (intensity/status) that the user doesn't think of as
        # part of "the print taking N seconds".
        print_start_t = asyncio.get_event_loop().time()
        elapsed_s: Optional[float] = None

        # Tell the user roughly how long this is going to take. The estimate
        # uses the same lines/s constant as the AA timeout, so it stays
        # in sync if either is retuned. Shown as a permanent grey detail
        # line so it stays visible while the transient "Sending image..."
        # progress overwrites itself.
        estimated_s = line_count / PRINT_COMPLETE_LINES_PER_SEC
        reporter.detail(f"Estimated ~{estimated_s:.0f}s")

        chunk_size = cmds.PRINTER_WIDTH_BYTES
        num_chunks = (len(image_data_buffer) + chunk_size - 1) // chunk_size
        reporter.step(f"Sending image (0/{num_chunks} lines)")

        for i in range(0, len(image_data_buffer), chunk_size):
            chunk = image_data_buffer[i : i + chunk_size]
            await session.client.write_gatt_char(
                session.data_char.uuid, chunk, response=False
            )
            await asyncio.sleep(PACING_DELAY_S)
            current_chunk_num = i // chunk_size + 1
            # Update reporter every ~5% (or every 16 lines for tiny images).
            update_every = max(1, num_chunks // 20)
            if current_chunk_num % update_every == 0 or current_chunk_num == num_chunks:
                pct = current_chunk_num * 100 // num_chunks
                reporter.step(
                    f"Sending image ({current_chunk_num}/{num_chunks} lines, {pct}%)"
                )

        reporter.step("Flushing buffer")
        await session.client.write_gatt_char(
            session.control_char.uuid, cmds.cmd_flush(), response=False
        )
        await asyncio.sleep(0.1)

        print_timeout_duration = PRINT_COMPLETE_BASE_TIMEOUT_S + (
            line_count / PRINT_COMPLETE_LINES_PER_SEC
        )
        reporter.step("Waiting for printer to finish")
        completion_payload = await wait_for(
            session,
            cmds.CommandIDs.PRINT_COMPLETE,
            print_timeout_duration,
        )

        if completion_payload is None:
            reporter.warn(
                f"No print-complete notification within {print_timeout_duration:.0f}s; "
                "the print may still be running."
            )
        else:
            elapsed_s = asyncio.get_event_loop().time() - print_start_t
            logger.debug(
                f"Print Complete payload: {completion_payload.hex()} "
                f"(elapsed {elapsed_s:.2f}s, {line_count / elapsed_s:.1f} lines/s)"
            )
            print_in_flight = False
            # Headline ✓ line is emitted by the caller (pawprint.cmd_print) so it
            # can include the filename. Just leave a transient step so the
            # disconnect happens under a tidy "Finishing up" hint.
            reporter.step("Finishing up")

        await asyncio.sleep(1.0)
        return elapsed_s
    finally:
        if print_in_flight:
            # Replace whichever step was active with a single concise message.
            reporter.step("Cancelling print")
            try:
                await session.client.write_gatt_char(
                    session.control_char.uuid,
                    cmds.cmd_cancel(),
                    response=False,
                )
                await asyncio.sleep(POST_CANCEL_DELAY_S)
                reporter.done("Cancelled")
            except Exception as e:
                logger.debug(f"Failed to send cleanup cancel: {e}")
                reporter.error("Failed to cancel cleanly")


# --- Helpers ---


_STATE_NAMES = {
    cmds.PrinterStates.STANDBY: "Standby",
    cmds.PrinterStates.PRINTING: "Printing",
}


def _format_state(status: cmds.StatusInfo) -> str:
    if not status.is_ok:
        if status.error_code is not None:
            return f"ERROR (code 0x{status.error_code:02X})"
        return "ERROR (unknown)"
    if status.state is None:
        return "Unknown"
    return _STATE_NAMES.get(status.state, f"Unknown(0x{status.state:02X})")


# --- Backwards-compatible entry point ---


async def run_ble(image_data_buffer: bytes, device: Optional[str], intensity: int):
    """Backwards-compatible wrapper used by print.py.

    The new code paths use `connected_printer` + `do_print` directly; this
    just composes them and swallows expected errors at the top level so the
    existing `print.py` semantics (no traceback for known errors) are
    preserved.
    """
    try:
        async with connected_printer(device) as session:
            await do_print(session, image_data_buffer, intensity)
    except cmds.PrinterError as e:
        logger.error(f"🛑 {e}")
    except BleakError as e:
        logger.error(f"🛑 Bluetooth Error: {e}")
    except asyncio.TimeoutError:
        logger.error("🛑 Connection timed out")
    except RuntimeError:
        # Already logged by connected_printer / get_device_address
        pass
    except Exception as e:
        logger.error(f"🛑 An unexpected error occurred: {e}", exc_info=True)
