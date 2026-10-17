# Brand

Jax OS uses the **Relay** symbol (three interlocking angular masses around a central aperture) with the **Axis** wordmark (hooked J, connected A/X, a coral square checkpoint in the X, and a smaller coral OS). The full lockup is the logo; compact slots and browser icons use Relay alone.

## Files

| Path | Use |
| --- | --- |
| `public/brand/logo-dark.svg`, `public/brand/logo-light.svg` | Full lockup for dark / light surfaces |
| `public/brand/mark-dark.svg`, `public/brand/mark-light.svg` | Relay only, for dark / light surfaces |
| `src/app/icon.svg` | Small-size-optimized Relay on a fixed dark rounded-square tile (32x32, radius 6, Relay fitted in a centered 28x28 box) |
| `src/app/favicon.ico` | 16, 32 and 48 px frames rasterized from `icon.svg` |

The four app/README SVGs share one path set (the icon starts from it and may carry limited optical adjustment for small sizes): the lettering is vector paths (no font, no text node), outer backgrounds are transparent except the browser tile, and nothing references an external file.

## Palette

SVG images cannot read CSS variables, so each file embeds the resolved value of an existing token from `src/styles/ds/colors.css`.

| Surface | Coral parts | Sage mass | Neutral parts |
| --- | --- | --- | --- |
| Dark app / dark README | coral-500 `#E46C54` | sage-300 `#A6BCA6` | warm-50 `#F7F1EA` |
| Light app / light README | coral-600 `#C8543D` | sage-600 `#557355` | warm-900 `#1E1915` |

The browser icon always uses the dark colours on a warm-950 `#16120F` tile, so the neutral mass stays visible on light and dark browser chrome.

## Usage

In the app, `src/components/BrandImage.tsx` renders the dark and light file of a pair and CSS shows the one matching the server-rendered `data-theme` (rules in `src/app/globals.css`). Fixed `width` / `height` reserve the layout. The README uses a `<picture>` with a dark source and a light fallback. Icons are registered only through the Next.js file convention (`src/app/icon.svg`, `src/app/favicon.ico`).

## Provenance

The symbol and wordmark were selected from an internal concept board (Relay symbol combined with the Axis wordmark). The committed SVGs were traced from that raster board and cleaned to an integer grid with straight edges snapped; the board itself is not shipped. This page makes no trademark or licensing claim.

## Exporting the browser icons

`favicon.ico` is reproducible from the committed `icon.svg` with any headless Chromium and the Python standard library; no dependency or permanent pipeline is needed.

1. Rasterize `src/app/icon.svg` at 16, 32 and 48 px with a transparent background, for example with Playwright:

```js
const { chromium } = require("playwright");
const fs = require("node:fs");
(async () => {
  const svg = fs.readFileSync("src/app/icon.svg", "utf8");
  const uri = "data:image/svg+xml;base64," + Buffer.from(svg).toString("base64");
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  for (const n of [16, 32, 48]) {
    await page.setViewportSize({ width: n, height: n });
    await page.setContent(`<body style="margin:0"><img id="i" src="${uri}" width="${n}" height="${n}" style="display:block"></body>`);
    await page.waitForFunction(() => document.getElementById("i").complete);
    await page.screenshot({ path: `icon-${n}.png`, omitBackground: true });
  }
  await browser.close();
})();
```

2. Pack the three PNGs into an ICO container:

```python
import struct
frames = [(n, open(f"icon-{n}.png", "rb").read()) for n in (16, 32, 48)]
directory, body, offset = b"", b"", 6 + 16 * len(frames)
for n, png in frames:
    directory += struct.pack("<BBBBHHII", n, n, 0, 0, 1, 32, len(png), offset + len(body))
    body += png
open("src/app/favicon.ico", "wb").write(struct.pack("<HHH", 0, 1, len(frames)) + directory + body)
```

Check the result: the ICO directory lists exactly three entries (16, 32, 48), each frame starts with the PNG signature, and at 16 px the centre pixel (the aperture) is the tile colour.
