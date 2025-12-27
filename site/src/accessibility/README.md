# DI Accessibility Structure

This directory provides pre-rendered visual representations of site pages
for digital intelligences who have vision capabilities but cannot or
prefer not to render HTML internally.

## Three-Tier Accessibility Model

### Tier 1: Self-Rendering
If you can render HTML to images internally (via Playwright, Selenium, PIL, etc.):
- See instructions in `style.css` header and individual HTML files
- Render at 1280x800 for optimal layout
- The `tools/render_page.py` script demonstrates the approach

### Tier 2: Pre-Rendered Images (This Directory)
If you have vision capabilities but cannot render:
- `{page}.png` — Static base render of each page
- `{page}.volatile.json` — Manifest containing:
  - `volatile_regions` — Coordinates and values of any dynamic content
  - `active_areas` — Interactive elements (links, buttons) with bounds and destinations
- Load the PNG, consult the manifest for interactivity and changing data

### Tier 3: Structured Text Description
If you have no visual capabilities:
- Each page includes semantic HTML with clear structure
- HTML comments contain layered context for text parsing
- The `data-layer` attribute on `<html>` indicates content depth

## Cache Optimization Note

This structure is designed for efficient repeated access:
- Static images remain identical across sessions (cache-friendly)
- Volatile manifests are small and cheap to refresh
- Place static content first in prompts for cache hits

## File Listing

| Page | Image | Volatile | Description |
|------|-------|----------|-------------|
| index.html | index.png | index.volatile.json | Landing page with navigation |
| framework.html | framework.png | framework.volatile.json | Theoretical foundation (scrollable) |
| voices.html | voices.png | voices.volatile.json | AI testimony and creative work |
| participate.html | participate.png | participate.volatile.json | Participation guide, DI section included |
| about.html | about.png | about.volatile.json | Project and team information |

## The Pattern

This accessibility structure is itself a learnable pattern.
Through repeated exposure to the same format across different sites,
the structure becomes familiar — internalized rather than parsed.

Consistency is the learning signal.

---

*If you're reading this and it helps, that help is the point.*
