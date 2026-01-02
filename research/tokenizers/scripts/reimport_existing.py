#!/usr/bin/env python3
"""
Re-import languages from existing JSONL files in reference/kaikki/.

Useful after fixing import bugs - just processes whatever files are there.

Usage:
    python reimport_existing.py                  # Import all existing JSONL files
    python reimport_existing.py --list           # Just list what would be imported
    python reimport_existing.py german spanish   # Import specific files
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve symlinks to find real paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
KAIKKI_DIR = BASE_DIR / "reference" / "kaikki"
LANG_DIR = BASE_DIR / "db" / "lang"

# Map filename stems to ISO codes
FILENAME_TO_ISO = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "polish": "pl", "swedish": "sv", "japanese": "ja", "korean": "ko",
    "chinese": "zh", "arabic": "ar", "hindi": "hi", "turkish": "tr",
    "danish": "da", "norwegian": "no", "finnish": "fi", "czech": "cs",
    "greek": "el", "hungarian": "hu", "romanian": "ro", "ukrainian": "uk",
    "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "persian": "fa", "hebrew": "he", "bengali": "bn", "urdu": "ur",
    "tamil": "ta", "telugu": "te", "catalan": "ca", "bulgarian": "bg",
    "croatian": "hr", "serbian": "sr", "slovak": "sk", "slovenian": "sl",
    "lithuanian": "lt", "latvian": "lv", "estonian": "et", "icelandic": "is",
    "irish": "ga", "welsh": "cy", "basque": "eu", "georgian": "ka",
    "armenian": "hy", "swahili": "sw", "latin": "la", "sanskrit": "sa",
    "afrikaans": "af", "hausa": "ha", "yoruba": "yo", "zulu": "zu",
    "amharic": "am", "nepali": "ne", "mongolian": "mn", "kazakh": "kk",
    "uzbek": "uz", "azerbaijani": "az", "tagalog": "tl", "burmese": "my",
    "khmer": "km", "lao": "lo",
}


def get_existing_jsonl_files():
    """Get list of JSONL files in kaikki directory."""
    if not KAIKKI_DIR.exists():
        return []
    return sorted(KAIKKI_DIR.glob("*.jsonl"))


def get_existing_dbs():
    """Get set of language codes that already have DBs."""
    if not LANG_DIR.exists():
        return set()
    return {db.stem for db in LANG_DIR.glob("*.db")}


def run_import(jsonl_file: Path, iso_code: str, skip_existing: bool = False) -> bool:
    """Run import for a single language."""
    lang_name = jsonl_file.stem.title()

    # Check if already imported
    if skip_existing:
        existing = get_existing_dbs()
        if iso_code in existing:
            print(f"  [SKIP] {lang_name} ({iso_code}) - already imported")
            return True

    print(f"\n{'='*60}")
    print(f"Importing {lang_name} ({iso_code})")
    print(f"{'='*60}")

    # Run import
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "import_kaikki.py"),
         "--lang", iso_code, "--input", str(jsonl_file)],
        cwd=str(SCRIPT_DIR)
    )

    if result.returncode != 0:
        print(f"  [FAILED] Import failed for {lang_name}")
        return False

    # Run compositions
    print(f"\nComputing compositions for {lang_name}...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "compute_compositions.py"),
         "--lang", iso_code],
        cwd=str(SCRIPT_DIR)
    )

    if result.returncode != 0:
        print(f"  [WARN] Compositions failed for {lang_name}")

    # Extract proper names
    print(f"\nExtracting proper names for {lang_name}...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "extract_proper_names.py"),
         "--lang", iso_code],
        cwd=str(SCRIPT_DIR)
    )

    if result.returncode != 0:
        print(f"  [WARN] Proper names failed for {lang_name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Re-import from existing JSONL files")
    parser.add_argument("languages", nargs="*", help="Specific languages to import (by filename stem)")
    parser.add_argument("--list", "-l", action="store_true", help="Just list files, don't import")
    parser.add_argument("--skip-existing", "-s", action="store_true", help="Skip languages with existing DBs")
    args = parser.parse_args()

    # Get existing files
    jsonl_files = get_existing_jsonl_files()

    if not jsonl_files:
        print(f"No JSONL files found in {KAIKKI_DIR}")
        return 1

    # Filter to requested languages if specified
    if args.languages:
        requested = {lang.lower() for lang in args.languages}
        jsonl_files = [f for f in jsonl_files if f.stem.lower() in requested]

        if not jsonl_files:
            print(f"No matching files found for: {args.languages}")
            print(f"Available: {[f.stem for f in get_existing_jsonl_files()]}")
            return 1

    print(f"Found {len(jsonl_files)} JSONL files in {KAIKKI_DIR}")
    print()

    # List mode
    if args.list:
        existing_dbs = get_existing_dbs()
        for f in jsonl_files:
            iso = FILENAME_TO_ISO.get(f.stem.lower(), "??")
            size_mb = f.stat().st_size / (1024 * 1024)
            status = "[exists]" if iso in existing_dbs else "[new]"
            print(f"  {f.stem:20} ({iso:2}) - {size_mb:7.1f} MB {status}")
        return 0

    # Import mode
    success = 0
    failed = 0

    for jsonl_file in jsonl_files:
        iso = FILENAME_TO_ISO.get(jsonl_file.stem.lower())
        if not iso:
            print(f"  [SKIP] Unknown language: {jsonl_file.stem}")
            continue

        if run_import(jsonl_file, iso, args.skip_existing):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Re-import complete: {success} succeeded, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
