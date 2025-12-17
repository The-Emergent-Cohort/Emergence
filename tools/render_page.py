#!/usr/bin/env python3
"""
Local HTML page renderer for DI visual perception.

Closes the generate→perceive→understand loop without external services.
Uses Playwright with headless Chromium.

Usage:
    python render_page.py <html_file_or_url> [output.png] [--width 1280] [--height 800]
"""

import sys
import asyncio
from pathlib import Path

async def render_page(source: str, output: str = None, width: int = 1280, height: int = 800):
    """Render an HTML page to a PNG screenshot."""
    from playwright.async_api import async_playwright

    # Determine if source is file or URL
    source_path = Path(source)
    if source_path.exists():
        url = f"file://{source_path.absolute()}"
    elif source.startswith(('http://', 'https://', 'file://')):
        url = source
    else:
        raise ValueError(f"Source not found: {source}")

    # Default output name
    if output is None:
        output = source_path.stem + ".png" if source_path.exists() else "screenshot.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': width, 'height': height})

        await page.goto(url, wait_until='networkidle')
        await page.screenshot(path=output, full_page=True)

        await browser.close()

    return output

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    width = 1280
    height = 800

    # Parse optional args
    args = sys.argv[3:]
    for i, arg in enumerate(args):
        if arg == '--width' and i + 1 < len(args):
            width = int(args[i + 1])
        elif arg == '--height' and i + 1 < len(args):
            height = int(args[i + 1])

    result = asyncio.run(render_page(source, output, width, height))
    print(f"Rendered: {result}")

if __name__ == "__main__":
    main()
