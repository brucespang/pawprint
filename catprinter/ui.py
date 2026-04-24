"""Small UX helpers for terse, pretty CLI output.

The `Reporter` shows transient grey "step" lines that get replaced as work
progresses, plus permanent `done`/`info`/`warn`/`error`/`kv`/`detail` lines
for results that should stick around.

When stderr is not a TTY (piped / redirected), transient lines fall back to
plain newline-terminated output so they still appear in logs.
"""
import asyncio
import contextlib
import os
import sys
from contextlib import asynccontextmanager
from typing import Callable, Optional, TextIO


_RESET = "\033[0m"
_GREY = "\033[2m"           # SGR 2 "dim/faint": readable in dark + light modes
                            # alike (the older `\033[90m` "bright black" was
                            # near-invisible in some dark-mode palettes).
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_CLEAR_LINE = "\r\033[2K"   # carriage return + erase entire line


class Reporter:
    """Drives terse single-line progress output for a CLI command.

    Calls are intended to be cheap; pass `enabled=False` (or use the
    module-level `NULL_REPORTER`) to silence everything.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        color: Optional[bool] = None,
        transient: Optional[bool] = None,
        enabled: bool = True,
    ):
        self.stream = stream if stream is not None else sys.stderr
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = (
            color
            if color is not None
            else (is_tty and os.environ.get("NO_COLOR") is None)
        )
        # Transient (carriage-return-overwriteable) lines need a TTY.
        self.transient = transient if transient is not None else is_tty
        self.enabled = enabled
        self._has_active = False

    # -- internal --------------------------------------------------------

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def _clear_active(self) -> None:
        if self._has_active:
            if self.transient:
                self.stream.write(_CLEAR_LINE)
            else:
                # We already wrote a newline-terminated step; nothing to clear.
                pass
            self.stream.flush()
            self._has_active = False

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    # -- transient progress ---------------------------------------------

    def step(self, msg: str) -> None:
        """Show a transient grey progress line. Replaced on next call."""
        if not self.enabled:
            return
        self._clear_active()
        if self.transient:
            self._write(self._c(_GREY, f"⋯ {msg}"))
            self._has_active = True
        else:
            self._write(self._c(_GREY, f"⋯ {msg}") + "\n")

    # -- permanent lines ------------------------------------------------

    def done(self, msg: str, hint: Optional[str] = None) -> None:
        """Replace any active step with a permanent ✓ line.

        An optional `hint` is appended after the message in grey, useful for
        secondary info like a `-d <addr>` reminder.
        """
        if not self.enabled:
            return
        self._clear_active()
        line = self._c(_GREEN, "✓ ") + msg
        if hint:
            line += "  " + self._c(_GREY, hint)
        self._write(line + "\n")

    def info(self, msg: str) -> None:
        """Print a permanent line (no marker)."""
        if not self.enabled:
            return
        self._clear_active()
        self._write(msg + "\n")

    def detail(self, msg: str, indent: str = "  ") -> None:
        """Print a permanent grey indented line, e.g. supplemental info."""
        if not self.enabled:
            return
        self._clear_active()
        self._write(self._c(_GREY, f"{indent}{msg}") + "\n")

    def kv(self, label: str, value: str, label_width: int = 8) -> None:
        """Print an indented key/value pair with a grey label."""
        if not self.enabled:
            return
        self._clear_active()
        label_str = self._c(_GREY, f"  {label.ljust(label_width)}")
        self._write(f"{label_str}{value}\n")

    def warn(self, msg: str) -> None:
        if not self.enabled:
            return
        self._clear_active()
        self._write(self._c(_YELLOW, "⚠ ") + msg + "\n")

    def error(self, msg: str) -> None:
        if not self.enabled:
            return
        self._clear_active()
        self._write(self._c(_RED, "✗ ") + msg + "\n")

    def close(self) -> None:
        """Erase any lingering transient line, e.g. at exit."""
        self._clear_active()


class _NullReporter:
    """No-op reporter so libraries can call `reporter.step(...)` unconditionally."""

    enabled = False

    def step(self, msg: str) -> None:  # noqa: D401
        pass

    def done(self, msg: str, hint: Optional[str] = None) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def detail(self, msg: str, indent: str = "  ") -> None:
        pass

    def kv(self, label: str, value: str, label_width: int = 8) -> None:
        pass

    def warn(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def close(self) -> None:
        pass


NULL_REPORTER: Reporter = _NullReporter()  # type: ignore[assignment]


@asynccontextmanager
async def countdown_step(
    reporter: Reporter,
    label_fn: Callable[[float], str],
    total_s: float,
    tick_s: float = 1.0,
):
    """Async context manager that periodically updates `reporter.step` with a
    countdown.

    Inside the `async with` block, a background task repeatedly calls
    `reporter.step(label_fn(remaining_seconds))` every `tick_s` seconds until
    `total_s` is reached or the context exits (whichever comes first).

    Other code running inside the block can still call `reporter.kv(...)` or
    `reporter.step(...)` — the next tick will simply overwrite the step line
    with the latest countdown text.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()

    async def _tick_loop() -> None:
        while True:
            elapsed = loop.time() - start
            remaining = max(0.0, total_s - elapsed)
            reporter.step(label_fn(remaining))
            if remaining <= 0:
                return
            await asyncio.sleep(min(tick_s, remaining))

    task = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
