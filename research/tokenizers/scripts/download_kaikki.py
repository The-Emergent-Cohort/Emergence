#!/usr/bin/env python3
"""
Download Kaikki.org Wiktionary extracts for the concept tokenizer.

Kaikki.org provides weekly Wiktionary extracts with:
- Definitions/glosses
- Etymology
- Inflected forms
- Translations
- Pronunciation

Data is in JSONL format (one JSON object per line).

Usage:
    python download_kaikki.py                    # Download English only
    python download_kaikki.py --lang de fr es    # Download specific languages
    python download_kaikki.py --all              # Download all available
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Base URL for Kaikki.org data
KAIKKI_BASE = "https://kaikki.org/dictionary"

# Common languages to download (ISO 639-1 codes used by Kaikki)
COMMON_LANGUAGES = [
    "English", "German", "French", "Spanish", "Italian",
    "Portuguese", "Dutch", "Russian", "Chinese", "Japanese"
]

# Default output directory (relative to this script)
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "reference" / "kaikki"


def get_download_url(lang_name: str) -> str:
    """Construct download URL for a language."""
    # Kaikki.org URL pattern: https://kaikki.org/dictionary/{Language}/kaikki.org-dictionary-{Language}.json
    # For compressed: add .bz2
    return f"{KAIKKI_BASE}/{lang_name}/kaikki.org-dictionary-{lang_name}.json"


def download_with_retry(url: str, output_path: Path, retries: int = 4) -> bool:
    """Download a file with exponential backoff retry."""
    delays = [2, 4, 8, 16]  # Exponential backoff

    for attempt in range(retries + 1):
        try:
            print(f"  Downloading from {url}...")
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

                print()  # Newline after progress
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


def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def count_entries(filepath: Path) -> int:
    """Count JSONL entries in a file."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def download_language(lang_name: str, output_dir: Path, force: bool = False) -> dict:
    """Download Kaikki data for a single language."""
    url = get_download_url(lang_name)
    output_file = output_dir / f"{lang_name.lower()}.jsonl"

    result = {
        "language": lang_name,
        "url": url,
        "output_file": str(output_file),
        "success": False,
        "entries": 0,
        "size_mb": 0,
        "hash": None
    }

    if output_file.exists() and not force:
        print(f"  {lang_name}: Already exists (use --force to redownload)")
        result["success"] = True
        result["size_mb"] = output_file.stat().st_size / (1024 * 1024)
        result["hash"] = get_file_hash(output_file)
        return result

    print(f"\n[{lang_name}]")

    if download_with_retry(url, output_file):
        result["success"] = True
        result["size_mb"] = output_file.stat().st_size / (1024 * 1024)
        result["hash"] = get_file_hash(output_file)
        print(f"  Downloaded: {result['size_mb']:.1f} MB")

        # Count entries
        print("  Counting entries...", end="", flush=True)
        result["entries"] = count_entries(output_file)
        print(f" {result['entries']:,} entries")
    else:
        # Try compressed version
        url_bz2 = url + ".bz2"
        output_bz2 = output_dir / f"{lang_name.lower()}.jsonl.bz2"
        print(f"  Trying compressed version...")

        if download_with_retry(url_bz2, output_bz2):
            # Decompress
            import bz2
            print("  Decompressing...", end="", flush=True)
            with bz2.open(output_bz2, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    f_out.write(f_in.read())
            output_bz2.unlink()  # Remove compressed file
            print(" done")

            result["success"] = True
            result["size_mb"] = output_file.stat().st_size / (1024 * 1024)
            result["hash"] = get_file_hash(output_file)
            result["entries"] = count_entries(output_file)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaikki.org Wiktionary extracts"
    )
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        default=["English"],
        help="Languages to download (e.g., English German French)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all common languages"
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

    args = parser.parse_args()

    languages = COMMON_LANGUAGES if args.all else args.lang
    output_dir = args.output

    print(f"Kaikki.org Wiktionary Downloader")
    print(f"================================")
    print(f"Output directory: {output_dir}")
    print(f"Languages: {', '.join(languages)}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for lang in languages:
        result = download_language(lang, output_dir, args.force)
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  {r['language']}: {r['entries']:,} entries ({r['size_mb']:.1f} MB)")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['language']}: {r['url']}")

    # Write manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "kaikki.org",
            "files": results
        }, f, indent=2)
    print(f"\nManifest written to: {manifest_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
