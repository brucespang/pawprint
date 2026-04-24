![Cat Printer](./media/hackoclock.jpg)

This project allows you to use the MXW01 thermal cat printer with Python. It
provides a command-line interface to print images and markdown files.

It is a fork of **[jeremy46231/MXW01-catprinter](https://github.com/rbaron/catprinter)**

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

The main entry point is `./cli.py`, which exposes several subcommands. Pass
`-d <ADDR>` to skip auto-discovery (the address is logged whenever you scan or
auto-discover) and `-l debug` for verbose output.

```bash
$ ./cli.py --help
usage: cli.py [-h] [-l {debug,info,warn,warning,error}] [-d DEVICE] <command> ...

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
$ ./cli.py scan

# Quick health check:
$ ./cli.py status

# Print an image:
$ ./cli.py print my-image.png --intensity 0x5D

# Print an Obsidian-style markdown file (TODO list, note, etc.):
$ ./cli.py print demo/todo.md

# Render markdown to a PNG without printing (great for previewing styling):
$ ./cli.py render demo/todo.md -o /tmp/todo.png

# If you know the address, you can skip the search.
$ ./cli.py status -d <ADDR>

# Cancel an in-progress print
$ ./cli.py cancel
```

### `print` options

```bash
$ ./cli.py print --help
usage: cli.py print [-h] [-l {debug,info,warn,warning,error}] [-d DEVICE]
                    [-b {mean-threshold,floyd-steinberg,atkinson,halftone,none}]
                    [-s] [-i INTENSITY] [--top-first] [--reverse | --no-reverse]
                    filename
```

- `-b/--dithering-algo`: image binarization algorithm. Use `none` if your
  image is already 384px wide and 1-bit.
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

`cli.py render` and `cli.py print` parse Obsidian-flavored markdown
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
$ ./cli.py render demo/todo.md \
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

### Markdown demo

[demo/todo.md](demo/todo.md) is a first-print onboarding receipt
that exercises most supported features (headings, task list, blockquote,
bold/italic/strikethrough, inline `code`, fenced code block, GFM table,
ordered list, Obsidian wikilink, embedded image, front-matter). Render or
print it:

```bash
$ ./cli.py render demo/todo.md         # writes demo/todo.png
$ ./cli.py print demo/todo.md
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
