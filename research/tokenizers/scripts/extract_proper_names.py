#!/usr/bin/env python3
"""
Extract proper names from Kaikki JSONL before cleanup.

Proper names (proper nouns) need special handling:
- They're labels AND concepts
- They shouldn't contribute to general semantic weights
- They often cross languages unchanged
- They need categorization (person, place, org, etc.)

This script extracts proper names into their own tables with separate token ranges.

Usage:
    python extract_proper_names.py                     # Extract from English
    python extract_proper_names.py --lang de fr es     # Multiple languages
    python extract_proper_names.py --all               # All downloaded languages

Run BEFORE cleanup deletes the source JSONL files!
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Path configuration - resolve symlinks to find real lib path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
KAIKKI_DIR = BASE_DIR / "reference" / "kaikki"

# Add lib to path for token_encoder
sys.path.insert(0, str(BASE_DIR / "lib"))
from token_encoder import LanguageCoord

# NOTE: Capitalized common nouns (January, Monday, English, etc.) are NOT excluded.
# They stay in the main concept DB with capitalize_output=1 flag.
# Only extract actual proper names (people, places, named things) here.

# Category detection patterns
PERSON_PATTERNS = [
    r'\bperson\b', r'\bpeople\b', r'\bman\b', r'\bwoman\b',
    r'\bname\b.*\b(person|human)\b', r'\bgiven name\b', r'\bsurname\b',
    r'\bfirst name\b', r'\blast name\b', r'\bforename\b',
    r'\bbiblical\b', r'\bmythological\b', r'\bgoddess\b', r'\bgod\b',
    r'\bauthor\b', r'\bwriter\b', r'\bscientist\b', r'\bartist\b',
    r'\bphilosopher\b', r'\bmonarch\b', r'\bking\b', r'\bqueen\b',
    r'\bemperor\b', r'\bpresident\b', r'\bsaint\b', r'\bprophet\b',
]

PLACE_PATTERNS = [
    r'\bplace\b', r'\blocation\b', r'\bcountry\b', r'\bnation\b',
    r'\bcity\b', r'\btown\b', r'\bvillage\b', r'\bregion\b',
    r'\bstate\b', r'\bprovince\b', r'\bcounty\b', r'\bdistrict\b',
    r'\bmountain\b', r'\briver\b', r'\blake\b', r'\bocean\b', r'\bsea\b',
    r'\bisland\b', r'\bcontinent\b', r'\bplanet\b', r'\bstar\b',
    r'\bcapital\b.*\bof\b', r'\blocated in\b',
]

ORG_PATTERNS = [
    r'\borganization\b', r'\bcompany\b', r'\bcorporation\b',
    r'\binstitution\b', r'\buniversity\b', r'\bcollege\b',
    r'\bgovernment\b', r'\bparty\b', r'\bagency\b',
    r'\bchurch\b', r'\breligion\b', r'\bdenomination\b',
    r'\bteam\b', r'\bclub\b', r'\bleague\b',
]

WORK_PATTERNS = [
    r'\bbook\b', r'\bnovel\b', r'\bfilm\b', r'\bmovie\b',
    r'\bsong\b', r'\balbum\b', r'\bpainting\b', r'\bsculpture\b',
    r'\bplay\b', r'\bopera\b', r'\bsymphony\b',
    r'\bbrand\b', r'\bproduct\b', r'\btrademark\b',
    r'\bnewspaper\b', r'\bmagazine\b', r'\bjournal\b',
]

EVENT_PATTERNS = [
    r'\bwar\b', r'\bbattle\b', r'\brevolution\b',
    r'\btreaty\b', r'\bagreement\b', r'\bconference\b',
    r'\bdisaster\b', r'\bearthquake\b', r'\bhurricane\b',
    r'\bfestival\b', r'\bholiday\b', r'\bcelebration\b',
]

# Language coordinates mapping (ISO 639-1 -> genomic coordinates)
LANG_COORDS = {
    "en": (1, 8, 127, 0),    # Indo-European.Germanic.English
    "de": (1, 8, 200, 0),    # German
    "nl": (1, 8, 210, 0),    # Dutch
    "fr": (1, 12, 100, 0),   # Romance.French
    "es": (1, 12, 150, 0),   # Spanish
    "it": (1, 12, 170, 0),   # Italian
    "pt": (1, 12, 160, 0),   # Portuguese
    "ru": (1, 15, 100, 0),   # Slavic.Russian
    "ja": (9, 0, 100, 0),    # Japonic.Japanese
    "zh": (2, 1, 100, 0),    # Sino-Tibetan.Sinitic.Mandarin
    "ko": (10, 0, 100, 0),   # Koreanic.Korean
    "ar": (3, 1, 100, 0),    # Afro-Asiatic.Semitic.Arabic
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def init_db(db_path: Path):
    """Initialize proper names tables."""
    conn = sqlite3.connect(db_path)

    # Read and execute schema
    schema_path = DB_DIR / "SCHEMA-proper-names.sql"
    if schema_path.exists():
        with open(schema_path) as f:
            conn.executescript(f.read())
        log(f"Initialized proper names schema in {db_path}")
    else:
        log(f"Schema not found: {schema_path}", "ERROR")
        sys.exit(1)

    conn.commit()
    return conn


def detect_category(word: str, senses: list, pos: str) -> tuple[int, int, str]:
    """
    Detect proper name category from word, senses, and POS.

    Returns (category_id, domain_category, reasoning)
    Domain categories: 1=person, 2=place, 3=thing
    """
    # Combine all text for pattern matching
    all_text = word.lower() + " "
    for sense in senses:
        if isinstance(sense, dict):
            gloss = sense.get("glosses", [""])[0] if sense.get("glosses") else ""
            all_text += gloss.lower() + " "
            tags = sense.get("tags", [])
            all_text += " ".join(str(t) for t in tags).lower() + " "

    # Person (domain_category = 1)
    for pattern in PERSON_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            if re.search(r'mytholog|god|goddess|deity', all_text, re.I):
                return 13, 1, "person_mythological"
            elif re.search(r'fictional|character|novel|story', all_text, re.I):
                return 12, 1, "person_fictional"
            elif re.search(r'historical|ancient|medieval|\d{3,4}', all_text, re.I):
                return 11, 1, "person_historical"
            return 1, 1, "person"

    # Place (domain_category = 2)
    for pattern in PLACE_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            if re.search(r'country|nation|republic|kingdom', all_text, re.I):
                return 21, 2, "place_country"
            elif re.search(r'city|town|village|capital', all_text, re.I):
                return 22, 2, "place_city"
            elif re.search(r'state|province|region|county', all_text, re.I):
                return 23, 2, "place_region"
            elif re.search(r'planet|star|moon|asteroid|galaxy', all_text, re.I):
                return 25, 2, "place_celestial"
            elif re.search(r'mountain|river|lake|ocean|sea|island', all_text, re.I):
                return 24, 2, "place_geographic"
            return 2, 2, "place"

    # Thing (domain_category = 3) - orgs, works, events
    for pattern in ORG_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            return 31, 3, "thing_organization"

    for pattern in WORK_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            return 32, 3, "thing_work"

    for pattern in EVENT_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            return 33, 3, "thing_event"

    # Default to thing (catch-all for named entities)
    return 3, 3, "thing"


def is_proper_noun(entry: dict) -> bool:
    """Check if entry is a proper noun (named entity)."""
    word = entry.get("word", "")
    pos = entry.get("pos", "").lower()

    # Direct POS check
    if "proper" in pos or pos == "name":
        return True

    # Check tags
    for sense in entry.get("senses", []):
        if isinstance(sense, dict):
            tags = sense.get("tags", [])
            if any("proper" in str(t).lower() for t in tags):
                return True

    # Capitalized word with person/place/org signals in gloss
    if word and word[0].isupper():
        senses = entry.get("senses", [])
        for sense in senses:
            if isinstance(sense, dict):
                gloss = sense.get("glosses", [""])[0] if sense.get("glosses") else ""
                if any(re.search(p, gloss, re.I) for p in PERSON_PATTERNS + PLACE_PATTERNS + ORG_PATTERNS):
                    return True

    return False


def extract_etymology(entry: dict) -> str | None:
    """Extract etymology if present."""
    etymology = entry.get("etymology_text", "")
    if etymology:
        return etymology[:1000]  # Truncate very long etymologies

    # Try etymology_templates
    etym_templates = entry.get("etymology_templates", [])
    if etym_templates:
        parts = []
        for tmpl in etym_templates:
            if isinstance(tmpl, dict):
                expansion = tmpl.get("expansion", "")
                if expansion:
                    parts.append(expansion)
        if parts:
            return "; ".join(parts)[:1000]

    return None


def extract_translations(entry: dict) -> list[tuple[str, str, str]]:
    """
    Extract translations from entry.

    Returns list of (lang, translation, variant_type)
    """
    translations = []

    # Direct translations field
    for trans in entry.get("translations", []):
        if isinstance(trans, dict):
            lang = trans.get("lang", trans.get("code", ""))
            word = trans.get("word", "")
            if lang and word:
                translations.append((lang, word, "translation"))

    return translations


def build_genomic_token(domain_category: int, lang_coords: tuple, fingerprint: int, collision: int) -> str:
    """Build genomic token for proper name: 99.10.{category}.{lang}.{fp}.{col}"""
    fam, sub, lng, dial = lang_coords
    return f"99.10.{domain_category}.{fam}.{sub}.{lng}.{dial}.{fingerprint}.{collision}"


def compute_name_fingerprint(name: str, gloss: str = "") -> int:
    """
    Compute fingerprint for proper name.

    Proper fingerprint = sum of descriptor concept fingerprints
    (PERSON + SCIENTIST + GERMAN + ... for "Einstein")

    TODO: When concept DB is available, look up descriptor fingerprints.
    For now, hash gloss text as approximation of descriptors.
    """
    # Use gloss as proxy for descriptors until concept DB lookup available
    text = f"{name} {gloss}".lower()
    h = 0
    for c in text:
        h = (h * 31 + ord(c)) % 1_000_000
    return h


def process_kaikki_file(filepath: Path, conn: sqlite3.Connection, lang: str) -> dict:
    """Process a single Kaikki JSONL file and extract proper names."""
    stats = {
        "total_entries": 0,
        "proper_nouns": 0,
        "inserted": 0,
        "duplicates": 0,
        "by_category": {},
    }

    cursor = conn.cursor()

    # Get language coordinates
    lang_coords = LANG_COORDS.get(lang, (0, 0, 0, 0))

    # Track collisions per genomic prefix
    collision_tracker = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_entries"] += 1

            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "")
            if not word:
                continue

            if not is_proper_noun(entry):
                continue

            stats["proper_nouns"] += 1

            senses = entry.get("senses", [])
            pos = entry.get("pos", "")

            # Detect category (returns category_id, domain_category, reason)
            category_id, domain_category, category_reason = detect_category(word, senses, pos)

            # Track category stats
            cat_key = f"{category_id}:{category_reason}"
            stats["by_category"][cat_key] = stats["by_category"].get(cat_key, 0) + 1

            # Get gloss for fingerprint computation
            gloss = ""
            if senses:
                for sense in senses:
                    if isinstance(sense, dict) and sense.get("glosses"):
                        gloss = sense["glosses"][0]
                        break

            # Compute fingerprint (from gloss descriptors) and collision
            fingerprint = compute_name_fingerprint(word, gloss)
            prefix = f"99.10.{domain_category}.{'.'.join(map(str, lang_coords))}.{fingerprint}"
            collision = collision_tracker.get(prefix, 0)
            collision_tracker[prefix] = collision + 1

            # Build genomic token
            token_genomic = build_genomic_token(domain_category, lang_coords, fingerprint, collision)

            # Check for duplicate by canonical name and category
            cursor.execute(
                "SELECT name_id FROM proper_names WHERE canonical_name = ? AND category_id = ?",
                (word, category_id)
            )
            existing = cursor.fetchone()
            if existing:
                stats["duplicates"] += 1
                name_id = existing[0]
            else:
                # Insert new proper name with genomic notation
                etymology = extract_etymology(entry)
                fam, sub, lng, dial = lang_coords

                cursor.execute("""
                    INSERT INTO proper_names (
                        token_genomic, canonical_name, category_id,
                        lang_family, lang_subfamily, lang_code, lang_dialect,
                        fingerprint, collision, etymology, source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (token_genomic, word, category_id, fam, sub, lng, dial,
                      fingerprint, collision, etymology, "kaikki"))

                name_id = cursor.lastrowid
                stats["inserted"] += 1

                # Add canonical variant with language coordinates
                cursor.execute("""
                    INSERT INTO proper_name_variants (
                        name_id, variant, lang_family, lang_subfamily, lang_code, lang_dialect,
                        variant_type, is_primary
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'canonical', 1)
                """, (name_id, word, fam, sub, lng, dial))

            # Add translations as variants
            for trans_lang, trans_word, var_type in extract_translations(entry):
                # Get language coordinates for translation (default to universal if unknown)
                trans_coords = LANG_COORDS.get(trans_lang.lower()[:2], (0, 0, 0, 0))
                tfam, tsub, tlng, tdial = trans_coords

                # Check if variant exists
                cursor.execute("""
                    SELECT id FROM proper_name_variants
                    WHERE name_id = ? AND variant = ? AND lang_family = ? AND lang_code = ?
                """, (name_id, trans_word, tfam, tlng))

                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO proper_name_variants (
                            name_id, variant, lang_family, lang_subfamily, lang_code, lang_dialect,
                            variant_type
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (name_id, trans_word, tfam, tsub, tlng, tdial, var_type))

            # Commit periodically
            if stats["proper_nouns"] % 1000 == 0:
                conn.commit()
                log(f"  Processed {stats['proper_nouns']} proper nouns...")

    conn.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract proper names from Kaikki")
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        default=["en"],
        help="Languages to process (default: en)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all downloaded languages"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_DIR / "language_registry.db",
        help="Database path"
    )

    args = parser.parse_args()

    log("=" * 60)
    log("Proper Names Extraction")
    log("=" * 60)

    # Determine which languages to process
    if args.all:
        # Find all downloaded Kaikki files
        languages = []
        for jsonl in KAIKKI_DIR.glob("*.jsonl"):
            lang = jsonl.stem  # e.g., "english" -> use first 2 chars
            # Map Kaikki names to ISO codes
            lang_map = {
                "english": "en", "german": "de", "french": "fr",
                "spanish": "es", "italian": "it", "portuguese": "pt",
                "dutch": "nl", "russian": "ru", "japanese": "ja",
                "chinese": "zh", "korean": "ko", "arabic": "ar",
            }
            iso = lang_map.get(lang.lower(), lang[:2])
            languages.append((iso, jsonl))
    else:
        languages = []
        for lang in args.lang:
            # Try to find corresponding Kaikki file
            lang_lower = lang.lower()
            # Map ISO to Kaikki names
            iso_map = {
                "en": "english", "de": "german", "fr": "french",
                "es": "spanish", "it": "italian", "pt": "portuguese",
                "nl": "dutch", "ru": "russian", "ja": "japanese",
                "zh": "chinese", "ko": "korean", "ar": "arabic",
            }
            kaikki_name = iso_map.get(lang_lower, lang_lower)
            jsonl_path = KAIKKI_DIR / f"{kaikki_name}.jsonl"

            if jsonl_path.exists():
                languages.append((lang_lower, jsonl_path))
            else:
                log(f"Kaikki file not found for {lang}: {jsonl_path}", "WARN")

    if not languages:
        log("No languages to process", "ERROR")
        return 1

    log(f"Processing {len(languages)} languages")

    # Initialize database
    conn = init_db(args.db)

    # Process each language
    all_stats = {}
    for lang, jsonl_path in languages:
        log(f"\nProcessing {lang} from {jsonl_path.name}...")
        stats = process_kaikki_file(jsonl_path, conn, lang)
        all_stats[lang] = stats
        log(f"  Entries: {stats['total_entries']:,}")
        log(f"  Proper nouns found: {stats['proper_nouns']:,}")
        log(f"  New insertions: {stats['inserted']:,}")
        log(f"  Duplicates skipped: {stats['duplicates']:,}")

    # Summary
    log("\n" + "=" * 60)
    log("EXTRACTION SUMMARY")
    log("=" * 60)

    total_inserted = sum(s["inserted"] for s in all_stats.values())
    total_proper = sum(s["proper_nouns"] for s in all_stats.values())

    log(f"Total proper nouns found: {total_proper:,}")
    log(f"Total new entries: {total_inserted:,}")

    # Category breakdown
    all_categories = {}
    for stats in all_stats.values():
        for cat, count in stats["by_category"].items():
            all_categories[cat] = all_categories.get(cat, 0) + count

    log("\nBy category:")
    for cat, count in sorted(all_categories.items(), key=lambda x: -x[1]):
        log(f"  {cat}: {count:,}")

    # Final counts
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM proper_names")
    total_names = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM proper_name_variants")
    total_variants = cursor.fetchone()[0]

    log(f"\nDatabase totals:")
    log(f"  Proper names: {total_names:,}")
    log(f"  Variants: {total_variants:,}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
