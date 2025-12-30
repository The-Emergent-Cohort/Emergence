#!/usr/bin/env python3
"""
Download linguistic typology sources for the concept tokenizer.

Downloads CLDF (Cross-Linguistic Data Format) datasets from:
- Glottolog: Language inventory and genealogy
- WALS: World Atlas of Language Structures (typological features)
- Grambank: Grammar feature database

These provide the language-level metadata needed for multilingual tokenization.

Usage:
    python download_typology.py                    # Download all sources
    python download_typology.py --source glottolog # Download specific source
    python download_typology.py --list             # Show available sources
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Script directory
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "reference"

# Data sources with download URLs
# Note: These are the CLDF releases from Zenodo/GitHub
SOURCES = {
    "glottolog": {
        "name": "Glottolog",
        "description": "Language inventory, ISO codes, and genealogical classification",
        "url": "https://github.com/glottolog/glottolog-cldf/archive/refs/heads/main.zip",
        "output_dir": "glottolog",
        "format": "cldf",
        "files": ["languages.csv", "values.csv"]
    },
    "wals": {
        "name": "WALS (World Atlas of Language Structures)",
        "description": "Typological features: word order, morphology, etc.",
        "url": "https://github.com/cldf-datasets/wals/archive/refs/heads/main.zip",
        "output_dir": "wals",
        "format": "cldf",
        "files": ["languages.csv", "parameters.csv", "values.csv"]
    },
    "grambank": {
        "name": "Grambank",
        "description": "195 binary grammatical features for 2400+ languages",
        "url": "https://github.com/grambank/grambank/archive/refs/heads/main.zip",
        "output_dir": "grambank",
        "format": "cldf",
        "files": ["languages.csv", "parameters.csv", "values.csv"]
    }
}


def download_with_retry(url: str, output_path: Path, retries: int = 4) -> bool:
    """Download a file with exponential backoff retry."""
    delays = [2, 4, 8, 16]

    for attempt in range(retries + 1):
        try:
            print(f"  Downloading from {url[:80]}...")
            req = Request(url, headers={"User-Agent": "Emergence-Tokenizer/1.0"})

            with urlopen(req, timeout=300) as response:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size else None

                output_path.parent.mkdir(parents=True, exist_ok=True)

                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks

                with open(output_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            pct = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Progress: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
                        else:
                            mb_down = downloaded / (1024 * 1024)
                            print(f"\r  Downloaded: {mb_down:.1f} MB", end="", flush=True)

                print()
                return True

        except (URLError, HTTPError) as e:
            if attempt < retries:
                delay = delays[attempt]
                print(f"  Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  Failed after {retries + 1} attempts: {e}")
                return False

    return False


def extract_cldf(zip_path: Path, output_dir: Path, expected_files: list) -> bool:
    """Extract CLDF files from a zip archive."""
    print(f"  Extracting to {output_dir}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the cldf directory within the archive
            cldf_dirs = [n for n in zf.namelist() if '/cldf/' in n or n.endswith('/cldf/')]

            if not cldf_dirs:
                # Try to find CSV files directly
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
                if csv_files:
                    print(f"  Found {len(csv_files)} CSV files")
                else:
                    print("  Warning: No CLDF directory found in archive")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract all CSV files
            extracted = []
            for name in zf.namelist():
                if name.endswith('.csv'):
                    # Extract to flat structure
                    filename = Path(name).name
                    target = output_dir / filename

                    with zf.open(name) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
                    extracted.append(filename)
                    print(f"    Extracted: {filename}")

            # Check for expected files
            for expected in expected_files:
                if expected not in extracted:
                    print(f"  Warning: Expected file '{expected}' not found")

        return True

    except zipfile.BadZipFile as e:
        print(f"  Error extracting: {e}")
        return False


def download_source(source_key: str, output_base: Path, force: bool = False) -> dict:
    """Download and extract a single source."""
    source = SOURCES[source_key]
    output_dir = output_base / source["output_dir"]
    zip_path = output_base / f"{source_key}.zip"

    result = {
        "source": source_key,
        "name": source["name"],
        "output_dir": str(output_dir),
        "success": False
    }

    # Check if already exists
    if output_dir.exists() and not force:
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            print(f"  {source['name']}: Already exists ({len(csv_files)} files)")
            result["success"] = True
            result["files"] = [f.name for f in csv_files]
            return result

    print(f"\n[{source['name']}]")
    print(f"  {source['description']}")

    # Download
    if download_with_retry(source["url"], zip_path):
        # Extract
        if extract_cldf(zip_path, output_dir, source.get("files", [])):
            result["success"] = True
            result["files"] = [f.name for f in output_dir.glob("*.csv")]

            # Clean up zip
            zip_path.unlink()
        else:
            result["error"] = "Extraction failed"
    else:
        result["error"] = "Download failed"

    return result


def list_sources():
    """Print available sources."""
    print("Available typology sources:")
    print()
    for key, source in SOURCES.items():
        print(f"  {key}")
        print(f"    Name: {source['name']}")
        print(f"    Description: {source['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download linguistic typology sources"
    )
    parser.add_argument(
        "--source", "-s",
        choices=list(SOURCES.keys()),
        help="Download specific source only"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force redownload even if files exist"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available sources"
    )

    args = parser.parse_args()

    if args.list:
        list_sources()
        return 0

    sources_to_download = [args.source] if args.source else list(SOURCES.keys())

    print("Linguistic Typology Downloader")
    print("=" * 40)
    print(f"Output directory: {args.output}")
    print(f"Sources: {', '.join(sources_to_download)}")

    results = []
    for source_key in sources_to_download:
        result = download_source(source_key, args.output, args.force)
        results.append(result)

    # Summary
    print("\n" + "=" * 40)
    print("DOWNLOAD SUMMARY")
    print("=" * 40)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        files = r.get("files", [])
        print(f"  {r['name']}: {len(files)} files")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'Unknown error')}")

    # Write manifest
    manifest_path = args.output / "typology_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": results
        }, f, indent=2)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
