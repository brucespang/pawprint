"""Render Obsidian-flavored markdown to a 384px-wide PNG via Playwright.

Pipeline:

    note.md
      -> md_to_html: parse with markdown-it-py + plugins (front-matter strip,
         GFM tables, task lists), apply two custom inline rules for Obsidian
         `[[wikilinks]]` and `![[image embeds]]`, then rewrite relative image
         srcs to absolute file:// URLs anchored at the source markdown's
         directory.
      -> build_page: wrap the HTML body in templates/page.html, layering
         default.css then any user --style files. CSS `url(...)` references
         are rewritten to absolute file:// URLs anchored at each stylesheet's
         own directory (so default.css can keep relative `../fonts/...` paths
         that work from disk and from `set_content`-injected pages alike).
      -> render_html_to_png: Playwright Chromium, viewport={width: 384},
         device_scale_factor=1, wait for `document.fonts.ready`, full_page
         screenshot.

The screenshot is consumed by the existing print pipeline (read_img + dither
+ binarize) so we don't need to do binarization ourselves.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from catprinter import logger

if TYPE_CHECKING:
    from playwright.async_api import Browser


_PKG_DIR = Path(__file__).parent
TEMPLATES_DIR = _PKG_DIR / "templates"
FONTS_DIR = _PKG_DIR / "fonts"
DEFAULT_CSS_PATH = TEMPLATES_DIR / "default.css"
PAGE_HTML_PATH = TEMPLATES_DIR / "page.html"

DEFAULT_WIDTH_PX = 384


# --- Markdown -> HTML -----------------------------------------------------

# Inline rules. Both run before `link` so they shadow normal `[`/`![`
# handling for the `[[...]]` and `![[...]]` cases.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n|]+)(?:\|([^\[\]\n]+))?\]\]")
_EMBED_RE = re.compile(r"!\[\[([^\[\]\n|]+)(?:\|([^\[\]\n]+))?\]\]")


def _wikilink_rule(state, silent: bool) -> bool:
    pos = state.pos
    src = state.src
    if pos + 4 > len(src) or src[pos] != "[" or src[pos + 1] != "[":
        return False
    m = _WIKILINK_RE.match(src, pos)
    if not m:
        return False
    target = m.group(1).strip()
    label = (m.group(2) or m.group(1)).strip()
    if not silent:
        tok_open = state.push("link_open", "a", 1)
        tok_open.attrs = {"class": "wikilink", "href": "#" + target}
        tok_open.markup = "wikilink"
        tok_open.info = "wikilink"
        tok_text = state.push("text", "", 0)
        tok_text.content = label
        state.push("link_close", "a", -1)
    state.pos = m.end()
    return True


def _embed_rule(state, silent: bool) -> bool:
    pos = state.pos
    src = state.src
    if pos + 5 > len(src) or src[pos] != "!" or src[pos + 1] != "[":
        return False
    m = _EMBED_RE.match(src, pos)
    if not m:
        return False
    target = m.group(1).strip()
    alt = (m.group(2) or m.group(1)).strip()
    if not silent:
        tok = state.push("image", "img", 0)
        tok.attrs = {"src": target, "alt": alt}
        tok.content = alt
        # markdown-it's HTML renderer reads alt text from the children list
        # rather than the `alt` attr, so seed it with a single text child.
        child = Token("text", "", 0)
        child.content = alt
        tok.children = [child]
    state.pos = m.end()
    return True


def _is_external_url(s: str) -> bool:
    """True for things we should leave alone (http://, file://, data:, etc.)."""
    return bool(re.match(r"^[a-z][a-z0-9+\-.]*:", s, re.IGNORECASE))


def _rewrite_image_srcs(tokens: list[Token], base_dir: Path) -> None:
    for tok in tokens:
        if tok.children:
            _rewrite_image_srcs(tok.children, base_dir)
        if tok.type == "image":
            src = tok.attrGet("src")
            if not src or _is_external_url(src):
                continue
            try:
                resolved = (base_dir / src).resolve(strict=False)
                tok.attrSet("src", resolved.as_uri())
            except (OSError, ValueError):
                pass


def _build_md() -> MarkdownIt:
    md = (
        MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": False})
        .enable("table")
        .enable("strikethrough")
        .use(front_matter_plugin)
        .use(tasklists_plugin, enabled=True)
    )
    # Order matters: embed rule must run before wikilink so `![[x]]` doesn't
    # get partially consumed as `!` + `[[x]]`.
    md.inline.ruler.before("link", "obsidian_embed", _embed_rule)
    md.inline.ruler.before("link", "wikilink", _wikilink_rule)
    return md


_MD = _build_md()


def md_to_html(text: str, *, base_dir: Optional[Path] = None) -> str:
    """Parse markdown text to an HTML body fragment.

    `base_dir` anchors relative image URLs (`![alt](pic.png)` and Obsidian
    `![[pic.png]]`). Pass the directory of the source `.md` so embedded
    images resolve correctly. Pass None to leave srcs untouched (useful for
    tests).
    """
    env: dict = {}
    tokens = _MD.parse(text, env)
    if base_dir is not None:
        _rewrite_image_srcs(tokens, Path(base_dir))
    return _MD.renderer.render(tokens, _MD.options, env)


# --- CSS layering ---------------------------------------------------------

# Match `url(...)` with optional surrounding quotes, capturing the URL body.
# Excludes `data:` URIs and absolute schemes via _is_external_url.
_CSS_URL_RE = re.compile(
    r"""url\(\s*
            (?P<q>['"]?)
            (?P<url>[^'"\)]+)
            (?P=q)
        \s*\)""",
    re.VERBOSE,
)


def _rewrite_css_urls(css: str, base_dir: Path) -> str:
    """Rewrite relative `url(...)` references in CSS to absolute file:// URLs.

    Anchors against `base_dir` (the directory of the stylesheet itself).
    External schemes (http://, data:, file://, etc.) are left alone.
    """

    def repl(m: re.Match) -> str:
        url = m.group("url").strip()
        if not url or _is_external_url(url):
            return m.group(0)
        try:
            resolved = (base_dir / url).resolve(strict=False)
        except (OSError, ValueError):
            return m.group(0)
        return f'url("{resolved.as_uri()}")'

    return _CSS_URL_RE.sub(repl, css)


def _read_css(path: Path) -> str:
    css = path.read_text(encoding="utf-8")
    return _rewrite_css_urls(css, path.parent.resolve())


def build_page(
    body_html: str,
    *,
    extra_css_paths: Sequence[Path] = (),
    title: str = "catprinter",
) -> str:
    """Wrap an HTML body in the page template, layering CSS.

    `default.css` is loaded first; each path in `extra_css_paths` is read and
    appended after it (in order). Later rules win, so users can override
    defaults without re-stating them.
    """
    default_css = _read_css(DEFAULT_CSS_PATH)
    user_css_chunks: list[str] = []
    for p in extra_css_paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(f"--style stylesheet not found: {p}")
        user_css_chunks.append(f"/* {p.name} */\n{_read_css(p)}")
    user_css = "\n\n".join(user_css_chunks)

    template = PAGE_HTML_PATH.read_text(encoding="utf-8")
    return template.format(
        title=title,
        default_css=default_css,
        user_css=user_css,
        body_html=body_html,
    )


# --- HTML -> PNG ----------------------------------------------------------


async def render_html_to_png(
    html: str,
    *,
    width_px: int,
    out_path: Path,
    browser: Optional["Browser"] = None,
    html_path: Optional[Path] = None,
) -> Path:
    """Screenshot an HTML page to `out_path` at exactly `width_px` columns.

    If `browser` is provided, the function reuses that running Chromium
    instance (useful for batch / watch modes). Otherwise it launches and
    closes its own.

    If `html_path` is provided, the HTML is written there and the page
    navigates to that `file://` URL (required so embedded `file://` images
    aren't blocked by Chromium's same-origin policy). Otherwise a temp file
    is created next to `out_path` and removed after the screenshot.
    """
    from playwright.async_api import async_playwright

    if browser is not None:
        return await _screenshot_with(
            browser, html, width_px, out_path, html_path
        )

    async with async_playwright() as p:
        b = await p.chromium.launch()
        try:
            return await _screenshot_with(
                b, html, width_px, out_path, html_path
            )
        finally:
            await b.close()


async def _wait_for_images(page) -> None:
    """Wait for every <img> on the page to either load or fail.

    `wait_until="load"` covers most cases, but in CI we sometimes see
    Playwright resolve before the last image's bytes are decoded - belt and
    braces.
    """
    await page.evaluate(
        """() => Promise.all(
            Array.from(document.images).map(img =>
                img.complete
                    ? Promise.resolve()
                    : new Promise(resolve => {
                        img.addEventListener('load', resolve, { once: true });
                        img.addEventListener('error', resolve, { once: true });
                    })
            )
        )"""
    )


async def _screenshot_with(
    browser: "Browser",
    html: str,
    width_px: int,
    out_path: Path,
    html_path: Optional[Path],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # We have to navigate to a real file:// URL (rather than use
    # `page.set_content`, which leaves the page at about:blank) so that
    # `<img src="file:///...">` references inside the HTML are not blocked
    # by Chromium's cross-origin policy.
    cleanup_html = html_path is None
    if html_path is None:
        html_path = out_path.with_suffix(".rendering.html")
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    # Tiny initial height so `full_page=True` grows from the body's actual
    # height rather than padding short content out to the viewport floor.
    page = await browser.new_page(
        viewport={"width": width_px, "height": 1},
        device_scale_factor=1,
    )
    try:
        # `wait_until="load"` waits for the load event, which fires after
        # subresources (images, stylesheets) have loaded. domcontentloaded
        # was firing before images had a chance, leading to broken-image
        # icons in the screenshot.
        await page.goto(html_path.resolve().as_uri(), wait_until="load")
        # @font-face declarations use font-display: block so glyphs are
        # invisible until loaded. Wait for them explicitly so the screenshot
        # is deterministic.
        await page.evaluate("document.fonts.ready")
        await _wait_for_images(page)
        await page.screenshot(
            path=str(out_path),
            full_page=True,
            omit_background=False,
        )
        return out_path
    finally:
        await page.close()
        if cleanup_html:
            try:
                html_path.unlink()
            except OSError:
                pass


def _md_to_page_html(
    md_path: Path,
    extra_css_paths: Sequence[Path],
) -> str:
    text = md_path.read_text(encoding="utf-8")
    body_html = md_to_html(text, base_dir=md_path.parent.resolve())
    return build_page(
        body_html, extra_css_paths=extra_css_paths, title=md_path.stem
    )


async def render_md_to_png_async(
    md_path: Path,
    out_path: Path,
    *,
    extra_css_paths: Sequence[Path] = (),
    width_px: int = DEFAULT_WIDTH_PX,
    keep_html: bool = False,
    browser: Optional["Browser"] = None,
) -> Path:
    """Async one-shot: read a .md file, render it to a PNG.

    Use this when you're already inside an event loop (e.g. from another
    async CLI command). Pass `browser` to reuse a long-lived Chromium.
    """
    md_path = Path(md_path)
    out_path = Path(out_path)
    page_html = _md_to_page_html(md_path, extra_css_paths)
    # When the user asked to keep the HTML, write it directly to its final
    # location and let render_html_to_png navigate to it (rather than write
    # a separate temp file). Otherwise let render_html_to_png manage a
    # temporary file beside the PNG output.
    html_path = out_path.with_suffix(".html") if keep_html else None
    if html_path is not None:
        logger.debug(f"Will keep intermediate HTML at {html_path}")
    await render_html_to_png(
        page_html,
        width_px=width_px,
        out_path=out_path,
        browser=browser,
        html_path=html_path,
    )
    return out_path


def render_md_to_png(
    md_path: Path,
    out_path: Path,
    *,
    extra_css_paths: Sequence[Path] = (),
    width_px: int = DEFAULT_WIDTH_PX,
    keep_html: bool = False,
) -> Path:
    """Synchronous one-shot. Wraps `render_md_to_png_async` with asyncio.run.

    Don't call from inside an existing event loop; use the async variant.
    """
    return asyncio.run(
        render_md_to_png_async(
            md_path,
            out_path,
            extra_css_paths=extra_css_paths,
            width_px=width_px,
            keep_html=keep_html,
        )
    )
