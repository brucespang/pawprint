**pawprint** is a CLI to print images and markdown files to the MXW01 thermal cat printer, and is a fork of **[jeremy46231/MXW01-catprinter](https://github.com/rbaron/catprinter)**.

It has been tested with the following printers:

- [HUIJUTCHEN Mini Photo Printer](https://www.amazon.com/dp/B0CLNT35KW)

# Setup

Requires Python 3.10+ and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Clone the
repo, then from its root:

```bash
$ uv sync && uv run playwright install chromium
```

# Usage

After `uv sync`, the `pawprint` console script is installed in the project
venv and can be invoked via `uv run pawprint …`. The repo-root
`./pawprint.py` script is equivalent (it self-bootstraps via `uv run` from
its shebang).

```bash
$ uv run pawprint --help
usage: pawprint [-h] [-l {debug,info,warn,warning,error}] [-d DEVICE] <command> ...

Talk to your MXW01 thermal cat printer over BLE.

positional arguments:
  <command>
    print               Print an image (or render+print a markdown file).
    render              Render a markdown file to a PNG (no printer connection).
    status              Show printer state, battery, head temperature, and firmware version in a single connection.
    scan                List nearby MXW01 printers (no connect). Useful for finding -d address.
    cancel              Send AC cancel-print, then poll A1 to confirm Standby.
```

### Common flows

```bash
# Find your printer (no connect, no paper):
$ uv run pawprint scan

# Quick health check:
$ uv run pawprint status

# Print an image:
$ uv run pawprint print my-image.png --intensity 0x5D

# Print an Obsidian-style markdown file (TODO list, note, etc.):
$ uv run pawprint print demo/todo.md

# Render markdown to a PNG without printing (great for previewing styling):
$ uv run pawprint render demo/todo.md -o /tmp/todo.png

# Pipe via stdin with `-`. Content type is auto-detected from magic bytes,
# so `print -` accepts EITHER an image stream (PNG/JPG/...) or markdown:
$ convert photo.heic png:- | uv run pawprint print -
$ echo "# Hi from stdin" | uv run pawprint print -
$ echo "# Hi" | uv run pawprint render - -o /tmp/note.png

# If you know the address, you can skip the search.
$ uv run pawprint status -d <ADDR>

# Cancel an in-progress print
$ uv run pawprint cancel
```

### `print` options

```bash
$ uv run pawprint print --help
usage: pawprint print [-h] [-l {debug,info,warn,warning,error}] [-d DEVICE]
                      [-b {mean-threshold,floyd-steinberg,atkinson,halftone,none}]
                      [-s] [-i INTENSITY] [--top-first] [--reverse | --no-reverse]
                      filename
```

- `-b/--dithering-algo`: image binarization algorithm. Use `none` if your
  image is already 384px wide and 1-bit. For `.md` inputs this controls
  the dither applied to the rendered PNG; the printed strip is then sent
  as-is, no second dither.
- `-s/--show-preview`: opens an OpenCV preview window and asks for
  confirmation before sending the image to the printer.
- `-i/--intensity`: print darkness, `0x00`-`0xFF`. Default `0x5D`.
- `--top-first`: send the image with no orientation transform (row 0 prints
  first, byte 0 on the left of the paper). Same effect as `--reverse`;
  kept for backward compat.
- `--reverse / --no-reverse`: print so the FIRST line of the source emerges
  from the printer first, and the strip reads correctly *as it streams*
  (no need to rotate it in-hand after tear-off). Defaults to ON for `.md`
  inputs and OFF for raw images. With `--no-reverse`, the legacy rot-180
  path applies and the strip is meant to be torn off and rotated 180
  degrees in-hand to read.
- `--style PATH`: extra CSS file to layer on top of the baked-in
  stylesheet (only used when `filename` is `.md`). Repeatable.
- `--keep-html`: when rendering markdown, also write the intermediate HTML
  next to the temp PNG (handy for CSS debugging).
- `--dry-run`: do everything except connect to the printer. Useful to
  verify image / markdown rendering without paper.

### Rendering markdown

`pawprint render` and `pawprint print` parse Obsidian-flavored markdown
and rasterize it to a 384px-wide PNG via headless Chromium. Supported
features: headings, paragraphs, bold/italic, bullet/numbered/nested lists,
GFM task lists (`- [ ]` / `- [x]`), GFM tables, fenced code, blockquotes,
horizontal rules, YAML front-matter (parsed and skipped), Obsidian
`[[wikilinks]]` and `![[image embeds]]`, and embedded local images.

Styling is plain CSS, layered: the baked-in [default
stylesheet](catprinter/templates/default.css) ships in the package and ships
with vendored Inter and JetBrains Mono fonts so renders look the same
everywhere. Append your own CSS with `--style my.css` (repeatable - later
files win):

```bash
$ uv run pawprint render demo/todo.md \
    --style demo/styles/big.css \
    --style demo/styles/center.css
```

A worked example is [demo/styles/big.css](demo/styles/big.css):

```css
body { font-size: 22px; padding: 24px; }
h1, h2 { text-align: center; }
```

There's no TOML config and no per-property CLI flags - if you want to change
something, write CSS. `--keep-html` dumps the post-template HTML next to the
PNG so you can iterate quickly with a real browser's devtools.

By default `pawprint render` writes a **dithered** PNG (Floyd-Steinberg) so
the file you preview matches what the printer's heating elements will
actually fire. Pick a different algorithm with `-b/--dithering-algo`, or
pass `-b none` to keep the raw anti-aliased Chromium screenshot:

```bash
$ uv run pawprint render demo/todo.md -b atkinson
$ uv run pawprint render demo/todo.md -b none -o /tmp/raw.png
```

The `print` command applies the same dither during rendering and then
sends the rendered PNG as-is (no second dither at a slightly different
scale), so what you see in `render` output is what gets printed.

### Markdown demo

[demo/todo.md](demo/todo.md) is a first-print onboarding receipt
that exercises most supported features (headings, task list, blockquote,
bold/italic/strikethrough, inline `code`, fenced code block, GFM table,
ordered list, Obsidian wikilink, embedded image, front-matter). Render or
print it:

```bash
$ uv run pawprint render demo/todo.md         # writes demo/todo.png
$ uv run pawprint print demo/todo.md
```

# Protocol Documentation

The MXW01's BLE protocol is documented in [PROTOCOL.md](./PROTOCOL.md). This
documentation is a work in progress, please contribute!

# Different Algorithms

This script offers several different dithering algorithms to convert images to
monochrome for printing. The following algorithms are available:

**Mean Threshold:**

![Mean threshold](./media/grumpymeanthreshold.png)

**Floyd Steinberg (default):**

![Floyd Steinberg](./media/grumpyfloydsteinbergexample.png)

**Atkinson:**

![Atkinson](./media/grumpyatkinsonexample.png)

**Halftone dithering:**

![Halftone](./media/grumpyhalftone.png)

**None (image must be 384px wide):**

![None](./media/grumpynone.png)
