#!/usr/bin/env python3
"""
Build spelling correction database for a language.

Combines:
1. Valid forms from Kaikki language DB
2. Frequency data from pyspellchecker
3. Phonetic codes for homonym detection
4. Common misspellings

Usage:
    python build_spelling_db.py --lang en      # Build English spelling DB
    python build_spelling_db.py --lang de      # Build German spelling DB
    python build_spelling_db.py --all          # Build for all imported languages

Requires:
    pip install pyspellchecker metaphone
"""

import argparse
import sqlite3
from pathlib import Path

# Try to import optional dependencies
try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False
    print("Warning: pyspellchecker not installed. Run: pip install pyspellchecker")

try:
    from metaphone import doublemetaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False
    print("Warning: metaphone not installed. Run: pip install metaphone")

# Hardcoded paths
BASE_DIR = Path("/usr/share/databases")
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
SPELLING_DIR = DB_DIR / "spelling"
SCHEMA_FILE = DB_DIR / "SCHEMA-spelling.sql"

# Language code to pyspellchecker language mapping
LANG_TO_SPELLCHECKER = {
    "en": "en", "eng": "en",
    "de": "de", "deu": "de",
    "es": "es", "spa": "es",
    "fr": "fr", "fra": "fr",
    "pt": "pt", "por": "pt",
    "it": "it", "ita": "it",
    "nl": "nl", "nld": "nl",
    "ru": "ru", "rus": "ru",
    "ar": "ar", "ara": "ar",
    "lv": "lv", "lat": "lv",
}

# Common English homonym groups
ENGLISH_HOMONYMS = [
    {
        "phonetic": "THER",
        "members": [
            ("there", "location - 'over there'", "place, location"),
            ("their", "possession - 'their house'", "belongs to them"),
            ("they're", "contraction - 'they are'", "they are"),
        ]
    },
    {
        "phonetic": "YR",
        "members": [
            ("your", "possession - 'your book'", "belongs to you"),
            ("you're", "contraction - 'you are'", "you are"),
        ]
    },
    {
        "phonetic": "TS",
        "members": [
            ("its", "possession - 'its color'", "belongs to it"),
            ("it's", "contraction - 'it is'", "it is"),
        ]
    },
    {
        "phonetic": "T",
        "members": [
            ("to", "direction/infinitive - 'go to', 'to run'", "toward, infinitive"),
            ("too", "also/excessive - 'me too', 'too much'", "also, excessive"),
            ("two", "number - '2'", "the number 2"),
        ]
    },
    {
        "phonetic": "HR",
        "members": [
            ("here", "location - 'right here'", "this place"),
            ("hear", "perception - 'I hear you'", "auditory sense"),
        ]
    },
    {
        "phonetic": "RITE",
        "members": [
            ("right", "correct/direction - 'turn right'", "correct, opposite of left"),
            ("write", "compose - 'write a letter'", "put words on paper"),
            ("rite", "ceremony - 'a sacred rite'", "ritual, ceremony"),
        ]
    },
    {
        "phonetic": "NO",
        "members": [
            ("no", "negative - 'no way'", "negation"),
            ("know", "awareness - 'I know'", "have knowledge"),
        ]
    },
    {
        "phonetic": "WETHER",
        "members": [
            ("weather", "climate - 'nice weather'", "atmospheric conditions"),
            ("whether", "conditional - 'whether or not'", "if, in case"),
        ]
    },
    {
        "phonetic": "THAN",
        "members": [
            ("then", "time/sequence - 'back then', 'then we go'", "at that time, next"),
            ("than", "comparison - 'bigger than'", "comparative"),
        ]
    },
    {
        "phonetic": "AFEKT",
        "members": [
            ("affect", "verb - 'it will affect you'", "to influence"),
            ("effect", "noun - 'the effect was'", "result, outcome"),
        ]
    },
]


def get_phonetic_code(word: str) -> str:
    """Generate phonetic code for a word."""
    if HAS_METAPHONE:
        codes = doublemetaphone(word)
        return codes[0] if codes[0] else codes[1] if codes[1] else ""
    # Fallback: simple soundex-like
    return word[:4].upper() if word else ""


def init_spelling_db(db_path: Path) -> sqlite3.Connection:
    """Initialize spelling database with schema."""
    SPELLING_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Read and execute schema
    if SCHEMA_FILE.exists():
        schema = SCHEMA_FILE.read_text()
        conn.executescript(schema)
    else:
        print(f"Warning: Schema file not found: {SCHEMA_FILE}")

    return conn


def load_valid_forms_from_lang_db(lang_db: Path, spelling_conn: sqlite3.Connection):
    """Load valid word forms from language database."""
    if not lang_db.exists():
        print(f"  Language DB not found: {lang_db}")
        return 0

    lang_conn = sqlite3.connect(lang_db)
    cursor = lang_conn.cursor()

    # Get all lemmas and surface forms
    cursor.execute("""
        SELECT DISTINCT lemma, concept_id FROM concepts
        UNION
        SELECT DISTINCT surface_form, concept_id FROM surface_forms
    """)

    spell_cursor = spelling_conn.cursor()
    count = 0

    for form, concept_id in cursor.fetchall():
        if not form or len(form) < 2:
            continue

        phonetic = get_phonetic_code(form)

        try:
            spell_cursor.execute("""
                INSERT OR IGNORE INTO valid_forms
                (form, form_lower, concept_id, phonetic_code)
                VALUES (?, ?, ?, ?)
            """, (form, form.lower(), concept_id, phonetic))
            count += 1
        except Exception as e:
            pass  # Skip duplicates

    spelling_conn.commit()
    lang_conn.close()

    return count


def load_from_pyspellchecker(lang_code: str, spelling_conn: sqlite3.Connection):
    """Load frequency data and known misspellings from pyspellchecker."""
    if not HAS_SPELLCHECKER:
        return 0, 0

    spellcheck_lang = LANG_TO_SPELLCHECKER.get(lang_code)
    if not spellcheck_lang:
        print(f"  No pyspellchecker support for {lang_code}")
        return 0, 0

    try:
        spell = SpellChecker(language=spellcheck_lang)
    except Exception as e:
        print(f"  Could not load pyspellchecker for {spellcheck_lang}: {e}")
        return 0, 0

    cursor = spelling_conn.cursor()

    # Update frequencies from pyspellchecker's word frequency list
    freq_count = 0
    for word, freq in spell.word_frequency.items():
        cursor.execute("""
            UPDATE valid_forms SET frequency = ?
            WHERE form_lower = ?
        """, (freq, word.lower()))
        if cursor.rowcount > 0:
            freq_count += 1

    # Add any words from pyspellchecker not in our valid_forms
    for word, freq in spell.word_frequency.items():
        phonetic = get_phonetic_code(word)
        cursor.execute("""
            INSERT OR IGNORE INTO valid_forms
            (form, form_lower, frequency, phonetic_code, concept_id)
            VALUES (?, ?, ?, ?, NULL)
        """, (word, word.lower(), freq, phonetic))

    spelling_conn.commit()

    return freq_count, len(spell.word_frequency)


def add_homonym_groups(spelling_conn: sqlite3.Connection, homonyms: list):
    """Add homonym groups for disambiguation."""
    cursor = spelling_conn.cursor()

    for group in homonyms:
        # Create group
        cursor.execute("""
            INSERT OR IGNORE INTO homonym_groups (phonetic_code, group_name)
            VALUES (?, ?)
        """, (group["phonetic"], f"{group['members'][0][0]}-group"))

        cursor.execute("SELECT id FROM homonym_groups WHERE phonetic_code = ?",
                      (group["phonetic"],))
        row = cursor.fetchone()
        if not row:
            continue
        group_id = row[0]

        # Add members
        for form, meaning_hint, usage_context in group["members"]:
            cursor.execute("""
                INSERT OR IGNORE INTO homonym_members
                (group_id, form, meaning_hint, usage_context)
                VALUES (?, ?, ?, ?)
            """, (group_id, form, meaning_hint, usage_context))

    spelling_conn.commit()
    return len(homonyms)


def generate_keyboard_misspellings(spelling_conn: sqlite3.Connection, limit: int = 10000):
    """Generate potential keyboard-based misspellings for common words."""
    cursor = spelling_conn.cursor()

    # Get adjacency map
    cursor.execute("""
        SELECT key_char, adjacent_chars FROM key_adjacency WHERE layout_id = 1
    """)
    adjacency = {row[0]: row[1].split(',') for row in cursor.fetchall()}

    # Get most frequent valid words
    cursor.execute("""
        SELECT form_lower FROM valid_forms
        WHERE frequency > 0
        ORDER BY frequency DESC
        LIMIT ?
    """, (limit,))

    misspell_count = 0
    for (word,) in cursor.fetchall():
        if len(word) < 3:
            continue

        # Generate adjacent-key substitutions
        for i, char in enumerate(word):
            if char in adjacency:
                for adj_char in adjacency[char]:
                    misspelled = word[:i] + adj_char + word[i+1:]

                    # Only add if it's not already a valid word
                    cursor.execute("""
                        SELECT 1 FROM valid_forms WHERE form_lower = ?
                    """, (misspelled,))

                    if not cursor.fetchone():
                        cursor.execute("""
                            INSERT OR IGNORE INTO misspellings
                            (misspelling, misspelling_lower, correction, error_type, confidence, source)
                            VALUES (?, ?, ?, 'keyboard', 0.7, 'generated')
                        """, (misspelled, misspelled, word))
                        misspell_count += 1

    spelling_conn.commit()
    return misspell_count


def build_spelling_db(lang_code: str) -> bool:
    """Build spelling database for a language."""
    # Normalize language code
    iso3_map = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa",
                "ja": "jpn", "zh": "cmn", "it": "ita", "pt": "por"}
    iso3 = iso3_map.get(lang_code, lang_code)

    lang_db = LANG_DIR / f"{iso3}.db"
    if not lang_db.exists():
        # Try 2-letter code
        lang_db = LANG_DIR / f"{lang_code}.db"
        if not lang_db.exists():
            print(f"Language DB not found for {lang_code}")
            return False

    spelling_db = SPELLING_DIR / f"{lang_code}.db"

    print(f"\nBuilding spelling DB for {lang_code}")
    print(f"  Source: {lang_db}")
    print(f"  Output: {spelling_db}")

    # Remove existing and recreate
    if spelling_db.exists():
        spelling_db.unlink()

    conn = init_spelling_db(spelling_db)

    # Load valid forms from language DB
    valid_count = load_valid_forms_from_lang_db(lang_db, conn)
    print(f"  Valid forms from Kaikki: {valid_count:,}")

    # Load frequency data from pyspellchecker
    freq_count, spell_total = load_from_pyspellchecker(lang_code, conn)
    print(f"  Frequency updates from pyspellchecker: {freq_count:,}")
    print(f"  Total pyspellchecker vocabulary: {spell_total:,}")

    # Add homonym groups (English only for now)
    if lang_code in ("en", "eng"):
        homonym_count = add_homonym_groups(conn, ENGLISH_HOMONYMS)
        print(f"  Homonym groups added: {homonym_count}")

    # Generate keyboard misspellings
    keyboard_count = generate_keyboard_misspellings(conn)
    print(f"  Keyboard misspellings generated: {keyboard_count:,}")

    conn.close()

    # Report size
    size_mb = spelling_db.stat().st_size / (1024 * 1024)
    print(f"  Output size: {size_mb:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build spelling correction database")
    parser.add_argument("--lang", "-l", help="Language code (e.g., en, de, fr)")
    parser.add_argument("--all", "-a", action="store_true", help="Build for all imported languages")
    args = parser.parse_args()

    print("=" * 60)
    print("Building Spelling Correction Databases")
    print("=" * 60)

    if args.all:
        # Build for all languages with DBs
        for lang_db in sorted(LANG_DIR.glob("*.db")):
            lang_code = lang_db.stem
            build_spelling_db(lang_code)
    elif args.lang:
        build_spelling_db(args.lang)
    else:
        # Default to English
        build_spelling_db("en")

    print("\n" + "=" * 60)
    print("Spelling DB build complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
