"""Content-type sniffing for stdin input.

`pawprint print -` and `pawprint render -` both accept content via stdin,
where there's no filename to look at. We use magic-byte sniffing (via the
`filetype` package) to decide whether the input is a printable image or a
markdown document.

The strategy is asymmetric on purpose:

- If the bytes start with a known image magic number (PNG/JPG/HEIC/AVIF/
  WebP/...), we call it an image. Crucially, this includes formats OpenCV
  can't decode (HEIC, AVIF, etc.) - we'd rather fail loudly with a clear
  "couldn't decode HEIC" error than silently feed the binary blob into the
  markdown renderer and produce a page of mojibake.
- Otherwise we call it markdown. This is permissive on purpose: real
  markdown files often contain stray non-UTF-8 bytes (smart quotes from
  Word, BOMs, copy-pasted control characters, etc.), and we want them to
  still render as text.

`filetype` only inspects the first ~261 bytes, so this is cheap even for
large piped inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import filetype


InputKind = Literal["image", "markdown"]


@dataclass(frozen=True)
class SniffResult:
    kind: InputKind
    # The detected image extension (e.g. "png", "heic", "avif") when
    # kind == "image". None for markdown. Useful for error messages.
    image_extension: Optional[str] = None


def sniff_stdin_kind(data: bytes) -> SniffResult:
    """Decide whether `data` is a printable image or a markdown document.

    See module docstring for the rationale.
    """
    if not data:
        # Empty input: arbitrary, but markdown gives a clearer downstream
        # error ("No markdown on stdin.") than the image path's
        # "couldn't decode" message.
        return SniffResult(kind="markdown")
    kind = filetype.guess(data)
    if kind is not None and kind.mime.startswith("image/"):
        return SniffResult(kind="image", image_extension=kind.extension)
    return SniffResult(kind="markdown")
