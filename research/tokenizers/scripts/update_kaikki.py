#!/usr/bin/env python3
"""
Update Kaikki data for existing language databases.

Designed for cron execution - checks for weekly updates and re-imports.

Kaikki.org updates weekly (usually Sunday/Monday).
This script:
1. Checks manifest for existing downloads
2. Compares Last-Modified headers from Kaikki
3. Downloads only if source is newer
4. Re-runs import for updated files
5. Re-computes compositions if data changed

Usage:
    python update_kaikki.py                    # Update all downloaded languages
    python update_kaikki.py --lang en de fr    # Update specific languages
    python update_kaikki.py --dry-run          # Check for updates without downloading

Cron example (weekly Monday 3am):
    0 3 * * 1 cd /path/to/tokenizers && ./venv/bin/python scripts/update_kaikki.py >> logs/update.log 2>&1
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
REF_DIR = BASE_DIR / "reference"
KAIKKI_DIR = REF_DIR / "kaikki"
LOG_DIR = BASE_DIR / "logs"

# Kaikki base URL
KAIKKI_BASE = "https://kaikki.org/dictionary"


def log(msg: str, level: str = "INFO"):
    """Log with timestamp for cron output."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def get_manifest() -> dict:
    """Load existing download manifest."""
    manifest_file = KAIKKI_DIR / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            return json.load(f)
    return {"files": []}


def save_manifest(manifest: dict):
    """Save updated manifest."""
    manifest_file = KAIKKI_DIR / "manifest.json"
    manifest["last_update_check"] = datetime.now().isoformat()
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)


def get_remote_modified(lang_name: str) -> datetime | None:
    """Get Last-Modified time from Kaikki server."""
    url = f"{KAIKKI_BASE}/{lang_name}/kaikki.org-dictionary-{lang_name}.jsonl"

    try:
        req = Request(url, method="HEAD")
        req.add_header("User-Agent", "Emergence-Tokenizer-Updater/1.0")

        with urlopen(req, timeout=30) as response:
            last_modified = response.headers.get("Last-Modified")
            if last_modified:
                return parsedate_to_datetime(last_modified)
    except (URLError, HTTPError) as e:
        log(f"Could not check {lang_name}: {e}", "WARN")

    return None


def get_local_modified(lang_name: str, manifest: dict) -> datetime | None:
    """Get local file modification time from manifest."""
    for file_info in manifest.get("files", []):
        if file_info.get("language") == lang_name:
            download_date = file_info.get("download_date")
            if download_date:
                try:
                    return datetime.fromisoformat(download_date)
                except ValueError:
                    pass

    # Fallback to file mtime
    local_file = KAIKKI_DIR / f"{lang_name.lower()}.jsonl"
    if local_file.exists():
        return datetime.fromtimestamp(local_file.stat().st_mtime)

    return None


def check_updates(languages: list[str], manifest: dict) -> list[dict]:
    """Check which languages have updates available."""
    updates = []

    for lang in languages:
        remote_time = get_remote_modified(lang)
        local_time = get_local_modified(lang, manifest)

        needs_update = False
        reason = ""

        if remote_time is None:
            reason = "remote check failed"
        elif local_time is None:
            needs_update = True
            reason = "not downloaded"
        elif remote_time > local_time:
            needs_update = True
            reason = f"remote newer ({remote_time.date()} > {local_time.date()})"
        else:
            reason = "up to date"

        updates.append({
            "language": lang,
            "needs_update": needs_update,
            "reason": reason,
            "remote_time": remote_time,
            "local_time": local_time,
        })

        status = "UPDATE" if needs_update else "OK"
        log(f"{lang}: {reason}", status)

    return updates


def download_language(lang_name: str) -> bool:
    """Download/update a single language file."""
    from download_kaikki import download_language as dl_lang

    result = dl_lang(lang_name, KAIKKI_DIR, force=True)
    return result.get("success", False)


def import_language(lang_code: str) -> bool:
    """Re-import a language after update."""
    import subprocess

    result = subprocess.run(
        [sys.executable, SCRIPT_DIR / "import_kaikki.py", "--lang", lang_code],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        log(f"Import failed for {lang_code}: {result.stderr}", "ERROR")
        return False

    return True


def recompute_compositions(lang_code: str) -> bool:
    """Re-compute compositions for a language."""
    import subprocess

    result = subprocess.run(
        [sys.executable, SCRIPT_DIR / "compute_compositions.py", "--lang", lang_code],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        log(f"Composition compute failed for {lang_code}: {result.stderr}", "ERROR")
        return False

    return True


def cleanup_source(lang_name: str) -> bool:
    """Delete source JSONL file to save space."""
    jsonl_file = KAIKKI_DIR / f"{lang_name.lower()}.jsonl"
    if jsonl_file.exists():
        size_mb = jsonl_file.stat().st_size / (1024 * 1024)
        jsonl_file.unlink()
        log(f"Cleaned up {lang_name.lower()}.jsonl ({size_mb:.1f} MB freed)")
        return True
    return False


def get_downloaded_languages(manifest: dict) -> list[str]:
    """Get list of languages we've already downloaded."""
    languages = []
    for file_info in manifest.get("files", []):
        if file_info.get("success"):
            lang = file_info.get("language")
            if lang:
                languages.append(lang)
    return languages


def main():
    parser = argparse.ArgumentParser(description="Update Kaikki data")
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        help="Specific languages to check (default: all downloaded)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Check for updates without downloading"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force update even if up to date"
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Download only, skip import step"
    )
    parser.add_argument(
        "--skip-compositions",
        action="store_true",
        help="Skip composition recomputation"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete source JSONL files after successful import (saves disk space)"
    )

    args = parser.parse_args()

    log("=" * 60)
    log("Kaikki Update Check")
    log("=" * 60)

    # Ensure directories exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    KAIKKI_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = get_manifest()

    # Determine languages to check
    if args.lang:
        languages = [l.title() for l in args.lang]
    else:
        languages = get_downloaded_languages(manifest)

    if not languages:
        log("No languages to check. Run download_kaikki.py first.", "WARN")
        return 0

    log(f"Checking {len(languages)} languages...")

    # Check for updates
    updates = check_updates(languages, manifest)

    # Filter to those needing updates
    to_update = [u for u in updates if u["needs_update"] or args.force]

    if not to_update:
        log("All languages up to date.")
        save_manifest(manifest)
        return 0

    log(f"{len(to_update)} languages need updates")

    if args.dry_run:
        log("Dry run - no downloads performed")
        return 0

    # Process updates
    updated = 0
    for update_info in to_update:
        lang = update_info["language"]
        log(f"Updating {lang}...")

        # Download
        if download_language(lang):
            log(f"Downloaded {lang}")

            # Update manifest
            for file_info in manifest.get("files", []):
                if file_info.get("language") == lang:
                    file_info["download_date"] = datetime.now().isoformat()
                    file_info["previous_date"] = update_info["local_time"].isoformat() if update_info["local_time"] else None
                    break

            # Import
            if not args.skip_import:
                # Map language name to ISO code
                lang_lower = lang.lower()
                iso_map = {"english": "en", "german": "de", "french": "fr",
                          "spanish": "es", "japanese": "ja", "chinese": "zh"}
                lang_code = iso_map.get(lang_lower, lang_lower[:2])

                if import_language(lang_code):
                    log(f"Imported {lang}")

                    # Recompute compositions
                    if not args.skip_compositions:
                        if recompute_compositions(lang_code):
                            log(f"Recomputed compositions for {lang}")

                    # Cleanup source file if requested
                    if args.cleanup:
                        cleanup_source(lang)

            updated += 1
        else:
            log(f"Failed to download {lang}", "ERROR")

    # Save updated manifest
    save_manifest(manifest)

    log("=" * 60)
    log(f"Update complete: {updated}/{len(to_update)} successful")
    log("=" * 60)

    return 0 if updated == len(to_update) else 1


if __name__ == "__main__":
    sys.exit(main())
