#!/usr/bin/env -S uv run --quiet
"""pawprint - command-line interface for the MXW01 thermal cat printer.

The shebang dispatches via `uv run`, which transparently uses (or creates)
the project venv defined by `pyproject.toml` / `uv.lock`. So `./pawprint.py
...` works from anywhere inside the repo without needing to activate
`.venv` first. `--quiet` keeps uv's own "Resolved/Audited" chatter off
stderr so the CLI's output stays clean.
"""
import argparse
import asyncio
import logging
import os
import sys
from typing import Awaitable, Callable, List, Optional

import numpy as np

from bleak.exc import BleakError

from catprinter import logger
from catprinter import cmds
from catprinter.ble import (
    SCAN_TIMEOUT_S,
    PrinterSession,
    connected_printer,
    do_battery,
    do_cancel,
    do_print,
    do_scan,
    do_status,
    do_version,
)
from catprinter.img import read_img, show_preview
from catprinter.md_render import (
    DEFAULT_WIDTH_PX,
    render_md_to_png_async,
)
from catprinter.ui import Reporter, countdown_step


# --- argparse setup ---


def _add_common_args(
    parser: argparse.ArgumentParser, *, suppress_defaults: bool = False
) -> None:
    """Adds -l/--log-level and -d/--device to a parser.

    When `suppress_defaults=True`, omits defaults via argparse.SUPPRESS so that
    the same args declared on the top-level parser are not silently
    overwritten by the subparser's defaults when the user only provides them
    at the top level (e.g. `./pawprint.py -d X status`).
    """
    log_default = argparse.SUPPRESS if suppress_defaults else "warning"
    device_default = argparse.SUPPRESS if suppress_defaults else ""
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["debug", "info", "warn", "warning", "error"],
        default=log_default,
        help=(
            "Logger verbosity. User-facing progress always renders via the "
            "Reporter; this controls the underlying Python logger only "
            "(default: warning). Use `debug` to see protocol-level chatter."
        ),
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=device_default,
        help=(
            "The printer's Bluetooth Low Energy (BLE) address "
            "(MAC address on Linux; UUID on macOS) "
            'or advertisement name (e.g.: "MXW01"). '
            "If omitted, the script will try to auto-discover the printer "
            "based on its advertised BLE services."
        ),
    )


def _add_print_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "filename",
        type=str,
        help=(
            "Image file to print (PNG/JPG/...). If the path ends in .md the "
            "file is rendered to a temporary PNG via the same pipeline as "
            "`render` and then printed; the temp PNG path is logged so you "
            "can inspect what was actually sent."
        ),
    )
    parser.add_argument(
        "-b",
        "--dithering-algo",
        type=str,
        choices=["mean-threshold", "floyd-steinberg", "atkinson", "halftone", "none"],
        default="floyd-steinberg",
        help=(
            f"Which image binarization algorithm to use. If 'none' is used, no "
            f"binarization will be used. In this case the image has to have a "
            f"width of {cmds.PRINTER_WIDTH_PIXELS} px."
        ),
    )
    parser.add_argument(
        "-s",
        "--show-preview",
        action="store_true",
        help=(
            "If set, displays the final image and asks the user for "
            "confirmation before printing."
        ),
    )
    parser.add_argument(
        "-i",
        "--intensity",
        type=lambda x: int(x, 0),
        default=0x5D,
        help=(
            "Print intensity/energy byte (0x00-0xFF, default 0x5D). Higher "
            "values generally produce darker prints. Accepts hex (0xNN) or "
            "decimal."
        ),
    )
    parser.add_argument(
        "--top-first",
        action="store_true",
        help=(
            "Send the image with no orientation transform: row 0 prints "
            "first, byte 0 on the left of the paper. The strip then reads "
            "correctly in printer-natural orientation (no in-hand rotation "
            "after tear-off). Same effect as --reverse; kept for backward "
            "compat with existing scripts."
        ),
    )
    parser.add_argument(
        "--reverse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Print so the FIRST line of the source emerges from the printer "
            "first, and the strip reads correctly *while it streams* (no "
            "need to flip the strip in-hand after tear-off). Defaults to ON "
            "for markdown inputs and OFF for raw images. Pass --no-reverse "
            "to force the legacy rot-180 behavior, where the strip is meant "
            "to be tore off and rotated 180 degrees in-hand to read."
        ),
    )
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Extra CSS file to layer on top of the baked-in default. Only "
            "used when `filename` is a .md path. Can be passed multiple "
            "times; later files win."
        ),
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help=(
            "When rendering markdown, also write the intermediate HTML next "
            "to the temp PNG (handy for CSS debugging)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do everything except connect to the printer. Useful to verify "
            "image / markdown rendering without paper."
        ),
    )


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("filename", type=str, help="Markdown file to render.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output PNG path. Defaults to the input filename with a .png "
            "extension, written next to the source markdown."
        ),
    )
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Extra CSS file to layer on top of the baked-in default. Can be "
            "passed multiple times; files are appended in the given order, "
            "so later files override earlier ones (and both override the "
            "default stylesheet)."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH_PX,
        help=(
            f"Render width in pixels (default {DEFAULT_WIDTH_PX}, the printer "
            f"width). Override only if you want a non-printer output."
        ),
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help=(
            "Also write the intermediate HTML next to the output PNG (handy "
            "for CSS debugging)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk to your MXW01 thermal cat printer over BLE.",
        prog="pawprint",
    )
    _add_common_args(parser)

    # A parent parser with the same common args, so subcommands also accept
    # `-d`/`-l` after the subcommand name (e.g. `./pawprint.py print foo.png
    # -d X`, which is what the print.py shim relies on for backwards compat).
    # Defaults are SUPPRESSed so that omitting -d/-l on the subcommand does
    # NOT overwrite a value that was passed at the top level.
    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common, suppress_defaults=True)

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_print = sub.add_parser(
        "print",
        parents=[common],
        help="Print an image (or render+print a markdown file).",
    )
    _add_print_args(p_print)

    p_render = sub.add_parser(
        "render",
        parents=[common],
        help="Render a markdown file to a PNG (no printer connection).",
    )
    _add_render_args(p_render)

    sub.add_parser(
        "status",
        parents=[common],
        help=(
            "Show printer state, battery, head temperature, and firmware "
            "version in a single connection."
        ),
    )

    p_scan = sub.add_parser(
        "scan",
        parents=[common],
        help="List nearby MXW01 printers (no connect). Useful for finding -d address.",
    )
    p_scan.add_argument(
        "--timeout",
        type=float,
        default=SCAN_TIMEOUT_S,
        help=f"How long to scan, in seconds (default {SCAN_TIMEOUT_S}).",
    )
    p_scan.add_argument(
        "--name",
        type=str,
        default=None,
        help="Only list devices with this advertisement name.",
    )

    sub.add_parser(
        "cancel",
        parents=[common],
        help="Send AC cancel-print, then poll A1 to confirm Standby.",
    )

    return parser


# --- Logger setup ---


def configure_logger(log_level: int) -> None:
    if logger.handlers:
        # Idempotent: avoid double-handlers when pawprint.main is re-entered
        # (e.g. via the print.py shim).
        logger.setLevel(log_level)
        for h in logger.handlers:
            h.setLevel(log_level)
        return
    logger.setLevel(log_level)
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(log_level)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)


# --- Reporter helpers ---


def _format_status_kv(
    reporter: Reporter,
    status: cmds.StatusInfo,
    battery_override: Optional[int] = None,
) -> None:
    """Print status fields as a small grey-labelled kv block.

    Pass `battery_override` (the AB response) to use that as the canonical
    battery percentage instead of the A1 status byte, which is unreliable on
    some firmwares.
    """
    if not status.is_ok:
        if status.error_code is not None:
            reporter.kv("Status", f"ERROR (code 0x{status.error_code:02X})")
        else:
            reporter.kv("Status", "ERROR (unknown)")
        return
    state_name = {
        cmds.PrinterStates.STANDBY: "Standby",
        cmds.PrinterStates.PRINTING: "Printing",
    }.get(
        status.state if status.state is not None else -1,
        f"Unknown(0x{status.state:02X})" if status.state is not None else "Unknown",
    )
    reporter.kv("Status", state_name)
    battery_pct = battery_override if battery_override is not None else status.battery_pct
    if battery_pct is not None:
        reporter.kv("Battery", f"{battery_pct}%")
    if status.temperature is not None:
        # The printer reports head temperature in °C; convert to °F for the
        # display since that's what we (Americans, sigh) think in.
        temp_f = round(status.temperature * 9 / 5 + 32)
        reporter.kv("Temp", f"{temp_f}\u00b0F")


def _format_version(v: cmds.VersionInfo) -> str:
    if v.type_byte is not None:
        return f"{v.version} (type 0x{v.type_byte:02X})"
    return v.version


# --- Command implementations ---


async def cmd_status(args: argparse.Namespace, reporter: Reporter) -> int:
    """Show state + battery + temperature + firmware version.

    Combines A1 (state, head temp), AB (battery percentage - more accurate
    than A1's battery byte under load), and B1 (firmware version) in a
    single connection.
    """
    async with connected_printer(args.device or None, reporter=reporter) as session:
        reporter.step("Reading status")
        status = await do_status(session)
        battery: Optional[int] = None
        try:
            battery = await do_battery(session)
        except cmds.PrinterError as e:
            reporter.warn(f"Could not read battery: {e}")
        _format_status_kv(reporter, status, battery_override=battery)
        try:
            version = await do_version(session)
            reporter.kv("Version", _format_version(version))
        except cmds.PrinterError as e:
            reporter.warn(f"Could not read version: {e}")
        if not status.is_ok and status.error_code is not None:
            raise cmds.error_for_code(status.error_code, status.raw)
    return 0


async def cmd_cancel(args: argparse.Namespace, reporter: Reporter) -> int:
    async with connected_printer(args.device or None, reporter=reporter) as session:
        await do_cancel(session)
    return 0


async def cmd_scan(args: argparse.Namespace, reporter: Reporter) -> int:
    found_count = 0

    def label(remaining: float) -> str:
        suffix = f", {found_count} found" if found_count else ""
        return f"Scanning for MXW01 printers ({remaining:.0f}s left{suffix})"

    def on_found(device) -> None:
        # Detection callback runs on the asyncio loop thread; safe to write
        # to the reporter here. Print the device immediately - the parallel
        # countdown task will redraw the step on its next tick.
        nonlocal found_count
        found_count += 1
        reporter.kv(device.name or "?", device.address, label_width=10)

    async with countdown_step(reporter, label, args.timeout):
        devices = await do_scan(
            timeout=args.timeout, name=args.name, on_found=on_found
        )
    if not devices:
        reporter.error("No matching devices found.")
        return 1
    reporter.done(
        f"Found {len(devices)} device" + ("" if len(devices) == 1 else "s")
    )
    return 0


def _is_markdown_path(path: str) -> bool:
    """Treat anything ending in .md or .markdown as markdown."""
    p = path.lower()
    return p.endswith(".md") or p.endswith(".markdown")


def _orient_for_print(
    img: np.ndarray, *, top_first: bool, reverse: bool
) -> np.ndarray:
    """Apply the orientation transform sent to the printer.

    The MXW01 prints rows in send order (`row 0` first), with byte 0 on
    the left of the paper and bit 0 (LSB) the leftmost pixel of each
    chunk. So an image sent untransformed lands on the strip exactly as
    it appears in the source - in the *printer-natural* strip orientation
    (head-end on one side, first-printed row at one end). The legacy
    default (`rot180`) compensates for the fact that, after tearing the
    strip off, users typically rotate it 180 degrees in-hand to read it
    title-up; the rot180 cancels that mental rotation.

    Modes:

    - `reverse=False, top_first=False` (legacy default for raw images):
      `rot180`. Tear off, rotate the strip 180 degrees in-hand, read
      top-to-bottom in source order.
    - `reverse=True` (new default for markdown): no transform. The strip
      reads correctly in printer-natural orientation, so you can read it
      *as it streams out* - the FIRST line of the document emerges
      first. The strip does NOT need to be rotated after tear-off.
    - `top_first=True`: same as reverse - no transform. Kept as a
      separate flag for backward compat (debug knob for already-rotated
      raw images).
    """
    if top_first or reverse:
        return img
    return np.rot90(img, k=2)


def _resolve_reverse(filename: str, reverse: Optional[bool]) -> bool:
    """Pick a default for `--reverse` when the user didn't say either way.

    Markdown gets the new "first line first" behavior; raw images keep the
    legacy rot-180 behavior so existing scripts/photos don't change.
    """
    if reverse is not None:
        return reverse
    return _is_markdown_path(filename)


async def cmd_print(args: argparse.Namespace, reporter: Reporter) -> int:
    import tempfile
    from pathlib import Path

    filename = args.filename
    if not os.path.exists(filename):
        reporter.error(f"File not found: {filename}")
        return 1

    if _is_markdown_path(filename):
        # Render to a non-deleted temp PNG so the user can inspect what was
        # actually sent to the printer if anything looks off.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", prefix="catprinter-", delete=False
        )
        tmp.close()
        png_path = Path(tmp.name)
        reporter.step(f"Rendering {filename}")
        await render_md_to_png_async(
            Path(filename),
            png_path,
            extra_css_paths=[Path(p) for p in args.style],
            keep_html=args.keep_html,
        )
        reporter.done(f"Rendered {filename} -> {png_path}")
        image_path = str(png_path)
    else:
        image_path = filename

    reporter.step(f"Reading {image_path}")
    bin_img_bool = read_img(
        image_path,
        cmds.PRINTER_WIDTH_PIXELS,
        args.dithering_algo,
    )

    if args.show_preview:
        preview_img_uint8 = (~bin_img_bool).astype(np.uint8) * 255
        show_preview(preview_img_uint8)

    reverse = _resolve_reverse(filename, args.reverse)
    bin_img_bool = _orient_for_print(
        bin_img_bool, top_first=args.top_first, reverse=reverse
    )

    reporter.step("Encoding image")
    image_data_buffer = cmds.prepare_image_data_buffer(bin_img_bool)
    line_count = len(image_data_buffer) // cmds.PRINTER_WIDTH_BYTES

    if args.dry_run:
        reporter.warn(
            f"Dry run: skipping BLE. Would print {line_count} lines "
            f"({len(image_data_buffer)} bytes) from {image_path}."
        )
        return 0

    async with connected_printer(args.device or None, reporter=reporter) as session:
        elapsed_s = await do_print(session, image_data_buffer, args.intensity)
    if elapsed_s is not None:
        reporter.done(
            f"Printed {filename} ({line_count} lines, {elapsed_s:.1f}s)"
        )
    else:
        # AA never arrived (warning was already emitted by do_print); skip
        # the timing rather than print a misleading "0.0s".
        reporter.done(f"Printed {filename} ({line_count} lines)")
    return 0


async def cmd_render(args: argparse.Namespace, reporter: Reporter) -> int:
    from pathlib import Path

    md_path = Path(args.filename)
    if not md_path.is_file():
        reporter.error(f"File not found: {md_path}")
        return 1

    out_path = Path(args.output) if args.output else md_path.with_suffix(".png")
    reporter.step(f"Rendering {md_path} -> {out_path}")
    await render_md_to_png_async(
        md_path,
        out_path,
        extra_css_paths=[Path(p) for p in args.style],
        width_px=args.width,
        keep_html=args.keep_html,
    )
    reporter.done(f"Wrote {out_path}")
    return 0


COMMANDS: dict = {
    "print": cmd_print,
    "render": cmd_render,
    "status": cmd_status,
    "scan": cmd_scan,
    "cancel": cmd_cancel,
}


# --- Entry point ---


_LEVEL_ALIASES = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = _LEVEL_ALIASES[args.log_level.lower()]
    configure_logger(log_level)

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2

    reporter = Reporter()
    try:
        return asyncio.run(handler(args, reporter)) or 0
    except cmds.PrinterError as e:
        reporter.error(str(e))
        return 1
    except BleakError as e:
        reporter.error(f"Bluetooth error: {e}")
        return 1
    except RuntimeError as e:
        # e.g. discovery errors, image preview "Aborted print."
        reporter.error(str(e))
        return 1
    except KeyboardInterrupt:
        # do_print's `finally` already showed "Cancelling"/"Cancelled" if it
        # was running mid-print. Otherwise just leave a small breadcrumb.
        reporter.warn("Interrupted")
        return 130
    except Exception as e:
        reporter.error(f"Unexpected error: {e}")
        logger.debug("Traceback:", exc_info=True)
        return 1
    finally:
        reporter.close()


if __name__ == "__main__":
    sys.exit(main())
