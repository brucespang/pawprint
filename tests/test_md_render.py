"""Tests for the markdown -> PNG rendering pipeline.

The pure tests (md_to_html, build_page CSS layering) run in any environment.
The Playwright integration tests are gated on the chromium browser binary
being installed; they auto-skip if it isn't.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import pytest

from catprinter.md_render import (
    DEFAULT_CSS_PATH,
    FONTS_DIR,
    build_page,
    md_to_html,
)


# --- md_to_html: pure tests, no browser -----------------------------------


def test_basic_inline_marks():
    out = md_to_html("Hello **bold** and *em* and `code`.")
    assert "<strong>bold</strong>" in out
    assert "<em>em</em>" in out
    assert "<code>code</code>" in out


def test_headings():
    out = md_to_html("# H1\n\n## H2\n\n###### H6\n")
    assert "<h1>H1</h1>" in out
    assert "<h2>H2</h2>" in out
    assert "<h6>H6</h6>" in out


def test_fenced_code_block():
    out = md_to_html("```\nhello world\n```\n")
    assert "<pre>" in out
    assert "<code>hello world\n</code>" in out


def test_bullet_and_numbered_lists():
    out = md_to_html("- a\n- b\n\n1. one\n2. two\n")
    assert "<ul>" in out and "<li>a</li>" in out
    assert "<ol>" in out and "<li>one</li>" in out


def test_nested_lists():
    out = md_to_html("- top\n  - nested\n  - also\n")
    # outer list contains an inner <ul>
    assert "<ul>" in out
    assert out.count("<ul>") == 2


def test_task_lists_unchecked_and_checked():
    out = md_to_html("- [ ] todo\n- [x] done\n")
    assert "task-list-item" in out
    assert 'type="checkbox"' in out
    # checked attribute is present for [x]
    checked_index = out.find("checked")
    done_index = out.find("done")
    assert 0 < checked_index < done_index, (
        "checked attribute should appear before 'done' text:\n" + out
    )
    # And NOT for the unchecked one (only one occurrence of `checked`).
    assert out.count("checked=") == 1


def test_front_matter_is_stripped():
    out = md_to_html("---\ntitle: Hi\nfoo: bar\n---\n\n# Header\n")
    assert "title:" not in out
    assert "---" not in out
    assert "<h1>Header</h1>" in out


def test_wikilink_basic():
    out = md_to_html("See [[Alice]] for details.")
    assert '<a class="wikilink" href="#Alice">Alice</a>' in out


def test_wikilink_with_alias():
    out = md_to_html("See [[Page|nice label]] please.")
    assert (
        '<a class="wikilink" href="#Page">nice label</a>' in out
    ), f"got: {out}"


def test_obsidian_image_embed():
    out = md_to_html("![[picture.png]]", base_dir=Path("/tmp"))
    assert "<img" in out
    # Path was rewritten to absolute file:// URL.
    assert "src=" in out
    assert "picture.png" in out
    assert 'src="file://' in out


def test_image_src_rewrite_relative():
    out = md_to_html("![alt](pic.png)", base_dir=Path("/tmp"))
    assert 'src="file:///private/tmp/pic.png"' in out or 'src="file:///tmp/pic.png"' in out


def test_image_src_external_url_unchanged():
    out = md_to_html("![alt](https://example.com/x.png)", base_dir=Path("/tmp"))
    assert 'src="https://example.com/x.png"' in out


def test_table():
    out = md_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in out
    assert "<th>a</th>" in out
    assert "<td>1</td>" in out


def test_strikethrough():
    out = md_to_html("~~gone~~")
    assert "<s>gone</s>" in out


# --- build_page: CSS layering --------------------------------------------


def test_build_page_includes_default_css():
    page = build_page("<p>x</p>")
    assert "@font-face" in page
    # The default stylesheet should appear inline.
    assert "Inter-Regular.woff2" in page
    assert "<p>x</p>" in page


def test_build_page_rewrites_font_urls_to_absolute():
    page = build_page("<p>x</p>")
    expected_uri = (FONTS_DIR / "Inter-Regular.woff2").resolve().as_uri()
    assert expected_uri in page


def test_build_page_appends_user_css(tmp_path: Path):
    user = tmp_path / "user.css"
    user.write_text("body { font-size: 99px; }\n")
    page = build_page("<p>x</p>", extra_css_paths=[user])
    default_css = DEFAULT_CSS_PATH.read_text(encoding="utf-8")
    # User CSS appears AFTER the default CSS (later wins in source order).
    user_idx = page.find("font-size: 99px")
    default_marker = "@font-face"
    default_last_idx = page.rfind(default_marker)
    assert default_last_idx != -1
    assert user_idx > default_last_idx, (
        f"user css ({user_idx}) should appear after last default rule "
        f"({default_last_idx})"
    )
    assert "@font-face" in default_css  # sanity


def test_build_page_multiple_user_files_in_order(tmp_path: Path):
    a = tmp_path / "a.css"
    b = tmp_path / "b.css"
    a.write_text("/* MARK_A */\n")
    b.write_text("/* MARK_B */\n")
    page = build_page("<p>x</p>", extra_css_paths=[a, b])
    a_idx = page.find("MARK_A")
    b_idx = page.find("MARK_B")
    assert 0 < a_idx < b_idx, f"a={a_idx} should precede b={b_idx}"


def test_build_page_user_css_url_rewritten_relative_to_user_dir(tmp_path: Path):
    # A user CSS file referencing a sibling asset should resolve against its
    # own directory, not the package's templates/.
    asset = tmp_path / "bg.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    user = tmp_path / "user.css"
    user.write_text('body { background: url("bg.png"); }\n')
    page = build_page("<p>x</p>", extra_css_paths=[user])
    assert asset.resolve().as_uri() in page


def test_build_page_external_url_left_alone(tmp_path: Path):
    user = tmp_path / "user.css"
    user.write_text('body { background: url("https://example.com/x.png"); }\n')
    page = build_page("<p>x</p>", extra_css_paths=[user])
    assert "https://example.com/x.png" in page


def test_build_page_missing_user_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_page("<p>x</p>", extra_css_paths=[tmp_path / "nope.css"])


# --- Playwright integration tests ----------------------------------------
# These only run if Chromium is installed (`playwright install chromium`).
# They auto-skip otherwise so the rest of the suite stays green.


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            try:
                p.chromium.executable_path  # noqa: B018
                return Path(p.chromium.executable_path).exists()
            except Exception:
                return False
    except Exception:
        return False


_REQUIRES_CHROMIUM = pytest.mark.skipif(
    not _chromium_available(),
    reason="chromium not installed; run `playwright install chromium`",
)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        f.seek(16)
        w, h = struct.unpack(">II", f.read(8))
    return w, h


@_REQUIRES_CHROMIUM
def test_render_md_to_png_width_and_height(tmp_path: Path):
    from catprinter.md_render import render_md_to_png

    src = tmp_path / "x.md"
    src.write_text("# Hello\n\n- [x] Done\n- [ ] Todo\n\nSome **bold** text.\n")
    out = tmp_path / "x.png"
    render_md_to_png(src, out)
    assert out.is_file() and out.stat().st_size > 100
    w, h = _png_dimensions(out)
    assert w == 384
    assert 50 <= h <= 1500, f"unexpected height {h}"


@_REQUIRES_CHROMIUM
def test_render_md_keep_html(tmp_path: Path):
    from catprinter.md_render import render_md_to_png

    src = tmp_path / "x.md"
    src.write_text("# Hello")
    out = tmp_path / "x.png"
    render_md_to_png(src, out, keep_html=True)
    assert out.is_file()
    assert out.with_suffix(".html").is_file()


@_REQUIRES_CHROMIUM
def test_render_html_reuses_browser(tmp_path: Path):
    """A reusable browser passed in should be used for multiple renders."""
    import asyncio

    from catprinter.md_render import (
        build_page,
        md_to_html,
        render_html_to_png,
    )

    async def go():
        from playwright.async_api import async_playwright

        launches = 0

        async with async_playwright() as p:
            real_launch = p.chromium.launch

            async def counting_launch(*a, **kw):
                nonlocal launches
                launches += 1
                return await real_launch(*a, **kw)

            p.chromium.launch = counting_launch  # type: ignore[assignment]
            browser = await counting_launch()
            try:
                html = build_page(md_to_html("# Hi"))
                p1 = tmp_path / "a.png"
                p2 = tmp_path / "b.png"
                await render_html_to_png(
                    html, width_px=384, out_path=p1, browser=browser
                )
                await render_html_to_png(
                    html, width_px=384, out_path=p2, browser=browser
                )
                assert p1.is_file() and p2.is_file()
            finally:
                await browser.close()
        assert launches == 1, f"expected 1 launch, got {launches}"

    asyncio.run(go())


@_REQUIRES_CHROMIUM
def test_print_dry_run_autodetects_md(tmp_path: Path):
    """`pawprint print foo.md --dry-run` should reach the encoder, not BLE."""
    from pawprint import main

    src = tmp_path / "x.md"
    src.write_text("# Hello\n\n- [ ] one\n- [ ] two\n")
    rc = main(["print", str(src), "--dry-run"])
    assert rc == 0
