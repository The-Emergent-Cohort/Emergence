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
    python download_kaikki.py --lang German French Spanish
    python download_kaikki.py --from-db          # Download all languages in DB
    python download_kaikki.py --from-db --limit 20  # Download top 20 by speakers
    python download_kaikki.py --all              # Download all common languages
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Base URL for Kaikki.org data
KAIKKI_BASE = "https://kaikki.org/dictionary"

# Default paths
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "reference" / "kaikki"
DEFAULT_DB = SCRIPT_DIR / "db" / "language.db"

# ISO 639-3 to Kaikki.org language name mapping
# Kaikki uses English language names, capitalized
ISO_TO_KAIKKI = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "rus": "Russian",
    "zho": "Chinese",
    "cmn": "Chinese",  # Mandarin -> Chinese
    "jpn": "Japanese",
    "kor": "Korean",
    "ara": "Arabic",
    "hin": "Hindi",
    "ben": "Bengali",
    "pol": "Polish",
    "ukr": "Ukrainian",
    "vie": "Vietnamese",
    "tur": "Turkish",
    "tha": "Thai",
    "swe": "Swedish",
    "nor": "Norwegian",
    "dan": "Danish",
    "fin": "Finnish",
    "ell": "Greek",
    "ces": "Czech",
    "ron": "Romanian",
    "hun": "Hungarian",
    "heb": "Hebrew",
    "ind": "Indonesian",
    "msa": "Malay",
    "tgl": "Tagalog",
    "swh": "Swahili",
    "lat": "Latin",
    "san": "Sanskrit",
    "fas": "Persian",
    "urd": "Urdu",
    "cat": "Catalan",
    "eus": "Basque",
    "glg": "Galician",
    "slk": "Slovak",
    "slv": "Slovenian",
    "hrv": "Croatian",
    "srp": "Serbian",
    "bul": "Bulgarian",
    "lit": "Lithuanian",
    "lav": "Latvian",
    "est": "Estonian",
    "afr": "Afrikaans",
    "isl": "Icelandic",
    "gle": "Irish",
    "cym": "Welsh",
    "gla": "Scottish Gaelic",
    "kat": "Georgian",
    "hye": "Armenian",
    "tam": "Tamil",
    "tel": "Telugu",
    "mal": "Malayalam",
    "kan": "Kannada",
    "mar": "Marathi",
    "guj": "Gujarati",
    "pan": "Punjabi",
    "nep": "Nepali",
    "sin": "Sinhala",
    "mya": "Burmese",
    "khm": "Khmer",
    "lao": "Lao",
    "amh": "Amharic",
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo",
    "zul": "Zulu",
    "xho": "Xhosa",
    "sot": "Sotho",
    "tsn": "Tswana",
    "mlg": "Malagasy",
    "fil": "Filipino",
    "jav": "Javanese",
    "sun": "Sundanese",
}

# Common languages to download with --all
COMMON_LANGUAGES = [
    "English", "German", "French", "Spanish", "Italian",
    "Portuguese", "Dutch", "Russian", "Chinese", "Japanese",
    "Korean", "Arabic", "Hindi", "Polish", "Turkish"
]

# Reverse mapping for raw data downloads (Kaikki name -> raw extract code)
# Raw extracts use 2-letter codes or special names
KAIKKI_TO_RAW = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Turkish": "tr",
    "Greek": "el",
    "Czech": "cs",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Kurdish": "ku",
}


def get_download_url(lang_name: str) -> str:
    """Construct download URL for a language (postprocessed JSONL)."""
    # Note: These files are marked as deprecated but still available
    return f"{KAIKKI_BASE}/{lang_name}/kaikki.org-dictionary-{lang_name}.jsonl"


def get_raw_download_url(lang_code: str) -> str:
    """Construct download URL for raw extract data (preferred)."""
    # Raw data is at /downloads/[code]/[code]-extract.jsonl.gz
    return f"https://kaikki.org/downloads/{lang_code}/{lang_code}-extract.jsonl.gz"


def download_with_retry(url: str, output_path: Path, retries: int = 4) -> bool:
    """Download a file with exponential backoff retry."""
    delays = [2, 4, 8, 16]

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


def get_languages_from_db(db_path: Path, limit: int = None) -> list:
    """
    Get languages from the database that need Kaikki data.

    Returns list of (lang_code, kaikki_name, name) tuples.
    Prioritizes by speaker count if available.
    """
    if not db_path.exists():
        print(f"Warning: Database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query languages, prioritize by speaker count
    query = """
        SELECT lang_code, name, speaker_count
        FROM languages
        WHERE level = 'language' OR level IS NULL
        ORDER BY
            CASE WHEN speaker_count IS NULL THEN 1 ELSE 0 END,
            speaker_count DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)

    languages = []
    for row in cursor:
        lang_code = row['lang_code']
        name = row['name']

        # Map to Kaikki name
        kaikki_name = ISO_TO_KAIKKI.get(lang_code)
        if not kaikki_name:
            # Try using the language name directly (capitalized)
            kaikki_name = name.title() if name else None

        if kaikki_name:
            languages.append((lang_code, kaikki_name, name))

    conn.close()
    return languages


def check_existing_kaikki(output_dir: Path, kaikki_name: str) -> bool:
    """Check if Kaikki data already exists for a language."""
    jsonl_file = output_dir / f"{kaikki_name.lower()}.jsonl"
    return jsonl_file.exists() and jsonl_file.stat().st_size > 0


def download_language(lang_name: str, output_dir: Path, force: bool = False) -> dict:
    """Download Kaikki data for a single language."""
    output_file = output_dir / f"{lang_name.lower()}.jsonl"

    result = {
        "language": lang_name,
        "url": "",
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

    # Try postprocessed JSONL first (direct download)
    url = get_download_url(lang_name)
    result["url"] = url

    if download_with_retry(url, output_file):
        result["success"] = True
        result["size_mb"] = output_file.stat().st_size / (1024 * 1024)
        result["hash"] = get_file_hash(output_file)
        print(f"  Downloaded: {result['size_mb']:.1f} MB")

        print("  Counting entries...", end="", flush=True)
        result["entries"] = count_entries(output_file)
        print(f" {result['entries']:,} entries")
        return result

    # Try raw extract (gzipped) if we have a code mapping
    raw_code = KAIKKI_TO_RAW.get(lang_name)
    if raw_code:
        url_raw = get_raw_download_url(raw_code)
        output_gz = output_dir / f"{lang_name.lower()}.jsonl.gz"
        print(f"  Trying raw extract...")
        result["url"] = url_raw

        if download_with_retry(url_raw, output_gz):
            import gzip
            print("  Decompressing...", end="", flush=True)
            with gzip.open(output_gz, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    # Stream to avoid loading entire file in memory
                    while chunk := f_in.read(1024 * 1024):
                        f_out.write(chunk)
            output_gz.unlink()
            print(" done")

            result["success"] = True
            result["size_mb"] = output_file.stat().st_size / (1024 * 1024)
            result["hash"] = get_file_hash(output_file)
            print(f"  Extracted: {result['size_mb']:.1f} MB")

            print("  Counting entries...", end="", flush=True)
            result["entries"] = count_entries(output_file)
            print(f" {result['entries']:,} entries")
            return result

    # Try compressed bz2 as last resort
    url_bz2 = get_download_url(lang_name) + ".bz2"
    output_bz2 = output_dir / f"{lang_name.lower()}.jsonl.bz2"
    print(f"  Trying bz2 compressed version...")

    if download_with_retry(url_bz2, output_bz2):
        import bz2
        print("  Decompressing...", end="", flush=True)
        with bz2.open(output_bz2, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())
        output_bz2.unlink()
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
        default=None,
        help="Languages to download (e.g., English German French)"
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Download languages found in the language database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Database path for --from-db (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of languages when using --from-db"
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only download languages that don't have Kaikki data yet"
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
    output_dir = args.output

    # Determine which languages to download
    if args.from_db:
        db_languages = get_languages_from_db(args.db, args.limit)
        if not db_languages:
            print("No languages found in database. Run import_glottolog.py first.")
            sys.exit(1)

        if args.missing_only:
            # Filter to only those missing Kaikki data
            db_languages = [
                (code, kaikki, name)
                for code, kaikki, name in db_languages
                if not check_existing_kaikki(output_dir, kaikki)
            ]

        languages = [kaikki for _, kaikki, _ in db_languages]
        print(f"Languages from database: {len(languages)}")
    elif args.all:
        languages = COMMON_LANGUAGES
    elif args.lang:
        languages = args.lang
    else:
        languages = ["English"]

    print(f"Kaikki.org Wiktionary Downloader")
    print(f"================================")
    print(f"Output directory: {output_dir}")
    print(f"Languages to process: {len(languages)}")
    if len(languages) <= 10:
        print(f"  {', '.join(languages)}")
    else:
        print(f"  {', '.join(languages[:5])}... and {len(languages) - 5} more")
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
