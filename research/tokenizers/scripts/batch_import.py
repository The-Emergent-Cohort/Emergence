#!/usr/bin/env python3
"""
Batch import multiple languages from Kaikki.

Handles download, import, and composition computation for language tiers.

Language tiers based on:
- Data quality (Wiktionary coverage)
- LLM training representation
- Speaker population
- Linguistic diversity (family coverage)

Usage:
    python batch_import.py --tier 1           # Top 10 high-quality languages
    python batch_import.py --tier 2           # Next 15 languages
    python batch_import.py --tier 3           # Remaining supported languages
    python batch_import.py --all              # All tiers
    python batch_import.py --lang de fr es    # Specific languages
    python batch_import.py --family germanic  # By language family

Run from: /usr/share/databases/scripts/
Requires: Phase 1 to have been run first
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
REF_DIR = BASE_DIR / "reference"
KAIKKI_DIR = REF_DIR / "kaikki"
LOG_DIR = BASE_DIR / "logs"

# Language tiers based on data quality and coverage
# Format: (kaikki_name, iso639_3, iso639_1, family, speakers_millions)

TIER_1 = [
    # Excellent data quality, strong LLM representation
    ("English", "eng", "en", "germanic", 1500),
    ("German", "deu", "de", "germanic", 135),
    ("French", "fra", "fr", "romance", 310),
    ("Spanish", "spa", "es", "romance", 550),
    ("Italian", "ita", "it", "romance", 68),
    ("Portuguese", "por", "pt", "romance", 260),
    ("Dutch", "nld", "nl", "germanic", 25),
    ("Russian", "rus", "ru", "slavic", 255),
    ("Polish", "pol", "pl", "slavic", 45),
    ("Swedish", "swe", "sv", "germanic", 13),
]

TIER_2 = [
    # Good data quality, moderate LLM representation
    ("Japanese", "jpn", "ja", "japonic", 125),
    ("Korean", "kor", "ko", "koreanic", 80),
    ("Chinese", "cmn", "zh", "sino-tibetan", 1100),
    ("Arabic", "ara", "ar", "afro-asiatic", 380),
    ("Hindi", "hin", "hi", "indo-aryan", 600),
    ("Turkish", "tur", "tr", "turkic", 90),
    ("Danish", "dan", "da", "germanic", 6),
    ("Norwegian", "nor", "no", "germanic", 5),
    ("Finnish", "fin", "fi", "uralic", 5),
    ("Czech", "ces", "cs", "slavic", 11),
    ("Hungarian", "hun", "hu", "uralic", 13),
    ("Greek", "ell", "el", "hellenic", 13),
    ("Hebrew", "heb", "he", "afro-asiatic", 9),
    ("Romanian", "ron", "ro", "romance", 26),
    ("Ukrainian", "ukr", "uk", "slavic", 45),
]

TIER_3 = [
    # Moderate data quality, limited LLM representation
    ("Indonesian", "ind", "id", "austronesian", 200),
    ("Malay", "msa", "ms", "austronesian", 80),
    ("Thai", "tha", "th", "tai-kadai", 60),
    ("Vietnamese", "vie", "vi", "austroasiatic", 85),
    ("Urdu", "urd", "ur", "indo-aryan", 230),
    ("Persian", "fas", "fa", "iranian", 110),
    ("Bengali", "ben", "bn", "indo-aryan", 270),
    ("Tamil", "tam", "ta", "dravidian", 85),
    ("Telugu", "tel", "te", "dravidian", 83),
    ("Catalan", "cat", "ca", "romance", 10),
    ("Bulgarian", "bul", "bg", "slavic", 8),
    ("Croatian", "hrv", "hr", "slavic", 6),
    ("Serbian", "srp", "sr", "slavic", 12),
    ("Slovak", "slk", "sk", "slavic", 5),
    ("Slovenian", "slv", "sl", "slavic", 2.5),
    ("Lithuanian", "lit", "lt", "baltic", 3),
    ("Latvian", "lav", "lv", "baltic", 1.5),
    ("Estonian", "est", "et", "uralic", 1.1),
    ("Icelandic", "isl", "is", "germanic", 0.4),
    ("Irish", "gle", "ga", "celtic", 1.8),
    ("Welsh", "cym", "cy", "celtic", 0.7),
    ("Basque", "eus", "eu", "isolate", 0.75),
    ("Georgian", "kat", "ka", "kartvelian", 4),
    ("Armenian", "hye", "hy", "armenian", 7),
    ("Swahili", "swh", "sw", "niger-congo", 100),
]

TIER_4 = [
    # Limited data, important for diversity
    ("Latin", "lat", "la", "romance-classical", 0),  # Classical, no native speakers
    ("Sanskrit", "san", "sa", "indo-aryan-classical", 0),  # Classical
    ("Ancient Greek", "grc", None, "hellenic-classical", 0),
    ("Afrikaans", "afr", "af", "germanic", 7),
    ("Tagalog", "tgl", "tl", "austronesian", 28),
    ("Hausa", "hau", "ha", "afro-asiatic", 80),
    ("Yoruba", "yor", "yo", "niger-congo", 47),
    ("Zulu", "zul", "zu", "niger-congo", 12),
    ("Amharic", "amh", "am", "afro-asiatic", 32),
    ("Nepali", "nep", "ne", "indo-aryan", 17),
    ("Burmese", "mya", "my", "sino-tibetan", 33),
    ("Khmer", "khm", "km", "austroasiatic", 16),
    ("Lao", "lao", "lo", "tai-kadai", 30),
    ("Mongolian", "mon", "mn", "mongolic", 6),
    ("Kazakh", "kaz", "kk", "turkic", 13),
    ("Uzbek", "uzb", "uz", "turkic", 35),
    ("Azerbaijani", "aze", "az", "turkic", 25),
]

# Family groupings for --family option
LANGUAGE_FAMILIES = {
    "germanic": ["English", "German", "Dutch", "Swedish", "Danish", "Norwegian", "Icelandic", "Afrikaans"],
    "romance": ["French", "Spanish", "Italian", "Portuguese", "Romanian", "Catalan"],
    "slavic": ["Russian", "Polish", "Ukrainian", "Czech", "Bulgarian", "Croatian", "Serbian", "Slovak", "Slovenian"],
    "sino-tibetan": ["Chinese", "Burmese"],
    "japonic": ["Japanese"],
    "koreanic": ["Korean"],
    "afro-asiatic": ["Arabic", "Hebrew", "Hausa", "Amharic"],
    "indo-aryan": ["Hindi", "Urdu", "Bengali", "Nepali"],
    "dravidian": ["Tamil", "Telugu"],
    "uralic": ["Finnish", "Hungarian", "Estonian"],
    "turkic": ["Turkish", "Kazakh", "Uzbek", "Azerbaijani"],
    "austronesian": ["Indonesian", "Malay", "Tagalog"],
    "austroasiatic": ["Vietnamese", "Khmer"],
    "tai-kadai": ["Thai", "Lao"],
    "celtic": ["Irish", "Welsh"],
    "baltic": ["Lithuanian", "Latvian"],
    "hellenic": ["Greek"],
    "iranian": ["Persian"],
    "kartvelian": ["Georgian"],
    "armenian": ["Armenian"],
    "niger-congo": ["Swahili", "Yoruba", "Zulu"],
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run_script(script_name: str, args: list) -> tuple[bool, str]:
    """Run a Python script and return success status."""
    cmd = [sys.executable, SCRIPT_DIR / script_name] + args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR
    )

    return result.returncode == 0, result.stderr if result.returncode != 0 else ""


def download_language(kaikki_name: str) -> bool:
    """Download Kaikki data for a language."""
    success, error = run_script("download_kaikki.py", ["--lang", kaikki_name])
    if not success:
        log(f"Download failed for {kaikki_name}: {error}", "ERROR")
    return success


def import_language(iso_code: str) -> bool:
    """Import language data into database."""
    success, error = run_script("import_kaikki.py", ["--lang", iso_code])
    if not success:
        log(f"Import failed for {iso_code}: {error}", "ERROR")
    return success


def compute_compositions(iso_code: str) -> bool:
    """Compute compositions for a language."""
    success, error = run_script("compute_compositions.py", ["--lang", iso_code])
    if not success:
        log(f"Compositions failed for {iso_code}: {error}", "ERROR")
    return success


def extract_proper_names(iso_code: str) -> bool:
    """Extract proper names before cleanup."""
    success, error = run_script("extract_proper_names.py", ["--lang", iso_code])
    if not success:
        log(f"Proper name extraction failed for {iso_code}: {error}", "WARN")
        # Don't fail the whole pipeline - proper names are optional
    return success


def cleanup_source(kaikki_name: str) -> bool:
    """Delete source JSONL file after successful import."""
    jsonl_file = KAIKKI_DIR / f"{kaikki_name.lower()}.jsonl"
    if jsonl_file.exists():
        size_mb = jsonl_file.stat().st_size / (1024 * 1024)
        jsonl_file.unlink()
        log(f"  Cleaned up {kaikki_name.lower()}.jsonl ({size_mb:.1f} MB freed)")
        return True
    return False


def process_language(lang_info: tuple, skip_download: bool = False,
                    skip_import: bool = False, skip_compositions: bool = False,
                    cleanup: bool = False) -> dict:
    """Process a single language through the full pipeline."""
    kaikki_name, iso3, iso1, family, speakers = lang_info
    iso_code = iso1 or iso3[:2]

    result = {
        "language": kaikki_name,
        "iso": iso_code,
        "family": family,
        "speakers_millions": speakers,
        "downloaded": False,
        "imported": False,
        "compositions": False,
        "success": False,
    }

    log(f"Processing {kaikki_name} ({iso_code}, {family}, {speakers}M speakers)")

    # Download
    if not skip_download:
        if download_language(kaikki_name):
            result["downloaded"] = True
            log(f"  Downloaded {kaikki_name}")
        else:
            return result
    else:
        result["downloaded"] = True

    # Import
    if not skip_import:
        if import_language(iso_code):
            result["imported"] = True
            log(f"  Imported {kaikki_name}")
        else:
            return result
    else:
        result["imported"] = True

    # Compositions
    if not skip_compositions:
        if compute_compositions(iso_code):
            result["compositions"] = True
            log(f"  Compositions computed for {kaikki_name}")
        else:
            return result
    else:
        result["compositions"] = True

    result["success"] = True

    # Extract proper names BEFORE cleanup (they're in the source JSONL)
    if cleanup and result["success"]:
        if extract_proper_names(iso_code):
            log(f"  Extracted proper names for {kaikki_name}")
        # Cleanup even if proper names extraction fails (it's optional)
        cleanup_source(kaikki_name)

    return result


def get_languages_for_tier(tier: int) -> list:
    """Get language list for a tier."""
    if tier == 1:
        return TIER_1
    elif tier == 2:
        return TIER_2
    elif tier == 3:
        return TIER_3
    elif tier == 4:
        return TIER_4
    return []


def get_languages_for_family(family: str) -> list:
    """Get languages for a language family."""
    family_lower = family.lower()
    if family_lower not in LANGUAGE_FAMILIES:
        log(f"Unknown family: {family}", "ERROR")
        log(f"Available: {', '.join(LANGUAGE_FAMILIES.keys())}")
        return []

    family_langs = LANGUAGE_FAMILIES[family_lower]
    all_langs = TIER_1 + TIER_2 + TIER_3 + TIER_4

    return [l for l in all_langs if l[0] in family_langs]


def main():
    parser = argparse.ArgumentParser(description="Batch import languages")
    parser.add_argument(
        "--tier", "-t",
        type=int,
        choices=[1, 2, 3, 4],
        help="Language tier to import (1=best quality, 4=diverse)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Import all tiers"
    )
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        help="Specific languages by Kaikki name"
    )
    parser.add_argument(
        "--family", "-f",
        help="Import by language family"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (use existing files)"
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip database import"
    )
    parser.add_argument(
        "--skip-compositions",
        action="store_true",
        help="Skip composition computation"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be processed"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if a language fails"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete source JSONL files after successful import"
    )

    args = parser.parse_args()

    log("=" * 60)
    log("Batch Language Import")
    log("=" * 60)

    # Determine which languages to process
    languages = []

    if args.all:
        languages = TIER_1 + TIER_2 + TIER_3 + TIER_4
    elif args.tier:
        languages = get_languages_for_tier(args.tier)
    elif args.family:
        languages = get_languages_for_family(args.family)
    elif args.lang:
        all_langs = TIER_1 + TIER_2 + TIER_3 + TIER_4
        lang_dict = {l[0].lower(): l for l in all_langs}
        for name in args.lang:
            if name.lower() in lang_dict:
                languages.append(lang_dict[name.lower()])
            else:
                log(f"Unknown language: {name}", "WARN")
    else:
        log("Specify --tier, --all, --family, or --lang")
        return 1

    if not languages:
        log("No languages to process")
        return 1

    # Summary
    log(f"Languages to process: {len(languages)}")
    total_speakers = sum(l[4] for l in languages)
    log(f"Total speaker coverage: {total_speakers:,.0f}M")

    families = set(l[3] for l in languages)
    log(f"Language families: {len(families)} ({', '.join(sorted(families))})")

    if args.dry_run:
        log("\nDry run - languages that would be processed:")
        for lang in languages:
            log(f"  {lang[0]} ({lang[2] or lang[1][:2]}) - {lang[3]}, {lang[4]}M speakers")
        return 0

    # Ensure directories exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    KAIKKI_DIR.mkdir(parents=True, exist_ok=True)
    LANG_DIR.mkdir(parents=True, exist_ok=True)

    # Process languages
    results = []
    start_time = time.time()

    for i, lang_info in enumerate(languages, 1):
        log(f"\n[{i}/{len(languages)}] Processing {lang_info[0]}...")

        result = process_language(
            lang_info,
            skip_download=args.skip_download,
            skip_import=args.skip_import,
            skip_compositions=args.skip_compositions,
            cleanup=args.cleanup
        )
        results.append(result)

        if not result["success"] and not args.continue_on_error:
            log(f"Stopping due to error with {lang_info[0]}", "ERROR")
            break

    elapsed = time.time() - start_time

    # Summary
    log("\n" + "=" * 60)
    log("BATCH IMPORT SUMMARY")
    log("=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    log(f"Successful: {len(successful)}/{len(results)}")
    log(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    if successful:
        log("\nSuccessful imports:")
        for r in successful:
            log(f"  {r['language']} ({r['iso']}) - {r['family']}")

    if failed:
        log("\nFailed imports:", "ERROR")
        for r in failed:
            stages = []
            if not r["downloaded"]:
                stages.append("download")
            elif not r["imported"]:
                stages.append("import")
            elif not r["compositions"]:
                stages.append("compositions")
            log(f"  {r['language']} - failed at: {', '.join(stages)}", "ERROR")

    # Write results log
    log_file = LOG_DIR / f"batch_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(log_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "results": results
        }, f, indent=2)
    log(f"\nResults saved to: {log_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
