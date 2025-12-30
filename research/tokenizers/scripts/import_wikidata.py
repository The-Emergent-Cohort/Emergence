#!/usr/bin/env python3
"""Import Wikidata Lexemes into tokenizer database morphemes table"""

import csv
import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "wikidata"

SOURCE_NAME = "wikidata"
SOURCE_CONFIDENCE = 0.80

# Language code to ISO 639-3 mapping
WIKIDATA_LANG_MAP = {
    'DE': ('deu', 'German'),
    'RU': ('rus', 'Russian'),
    'EN': ('eng', 'English'),
    'IT': ('ita', 'Italian'),
    'ES': ('spa', 'Spanish'),
    'HE': ('heb', 'Hebrew'),
    'CS': ('ces', 'Czech'),
    'FR': ('fra', 'French'),
    'SUX': ('sux', 'Sumerian'),
    'AKK': ('akk', 'Akkadian'),
    'TR': ('tur', 'Turkish'),
    'AR': ('ara', 'Arabic'),
    'HIT': ('hit', 'Hittite'),
}

# Lexical category to morpheme_type and pos_tendency
CATEGORY_MAP = {
    'verb': ('root', 'verb'),
    'noun': ('root', 'noun'),
    'adjective': ('root', 'adjective'),
    'adverb': ('root', 'adverb'),
    'pronoun': ('root', 'pronoun'),
    'preposition': ('word', 'preposition'),
    'conjunction': ('word', 'conjunction'),
    'interjection': ('word', 'interjection'),
    'article': ('word', 'article'),
    'numeral': ('root', 'numeral'),
    'particle': ('word', 'particle'),
    'prefix': ('prefix', None),
    'suffix': ('suffix', None),
    'infix': ('infix', None),
    'affix': ('affix', None),
    'proper noun': ('root', 'noun'),
}


def get_or_create_source(cur) -> int:
    """Register this source in data_sources table"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence)
        VALUES (?, ?, ?)
    """, (SOURCE_NAME, 'lexicon', SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def get_or_create_language(cur, iso_code: str, lang_name: str) -> int:
    """Get or create language entry"""
    cur.execute("SELECT id FROM language_families WHERE iso_639_3 = ?", (iso_code,))
    result = cur.fetchone()
    if result:
        return result[0]

    cur.execute("""
        INSERT INTO language_families (name, level, iso_639_3, source)
        VALUES (?, 'language', ?, 'wikidata')
    """, (lang_name, iso_code))
    return cur.lastrowid


def parse_wikidata_csv(csv_path: Path):
    """Parse a Wikidata Lexemes CSV file"""
    with open(csv_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lexeme_id = row.get('lexeme_id', '')
            lemma = row.get('lexemeLabel', '')
            category = row.get('lexical_categoryLabel', '').lower()

            if not lemma:
                continue

            # Normalize lemma - lowercase for consistency
            lemma = lemma.strip()

            yield {
                'lexeme_id': lexeme_id,
                'lemma': lemma,
                'category': category,
            }


def import_wikidata_file(conn, csv_path: Path, lang_code: str, source_id: int):
    """Import a single Wikidata CSV file"""
    cur = conn.cursor()

    iso_code, lang_name = WIKIDATA_LANG_MAP.get(lang_code, (lang_code.lower(), lang_code))
    lang_id = get_or_create_language(cur, iso_code, lang_name)

    morpheme_count = 0
    surface_count = 0
    batch_morphemes = []
    batch_surfaces = []
    batch_size = 1000

    for rec in parse_wikidata_csv(csv_path):
        # Determine morpheme_type and pos from category
        morpheme_type, pos = CATEGORY_MAP.get(rec['category'], ('word', None))

        # Add to morphemes table (the concept)
        batch_morphemes.append((
            rec['lemma'],
            rec['category'],
            morpheme_type,
            None,  # origin - not in Wikidata
            pos,
        ))

        if len(batch_morphemes) >= batch_size:
            cur.executemany("""
                INSERT OR IGNORE INTO morphemes
                (morpheme, meaning, morpheme_type, origin, pos_tendency)
                VALUES (?, ?, ?, ?, ?)
            """, batch_morphemes)
            morpheme_count += cur.rowcount
            conn.commit()
            batch_morphemes = []

    # Final morpheme batch
    if batch_morphemes:
        cur.executemany("""
            INSERT OR IGNORE INTO morphemes
            (morpheme, meaning, morpheme_type, origin, pos_tendency)
            VALUES (?, ?, ?, ?, ?)
        """, batch_morphemes)
        morpheme_count += cur.rowcount
        conn.commit()

    # Now create surface_forms linking to morphemes
    # Re-parse to link with morpheme IDs
    cur.execute("SELECT id, morpheme FROM morphemes")
    morpheme_lookup = {row[1]: row[0] for row in cur.fetchall()}

    for rec in parse_wikidata_csv(csv_path):
        morpheme_id = morpheme_lookup.get(rec['lemma'])
        if not morpheme_id:
            continue

        morpheme_type, pos = CATEGORY_MAP.get(rec['category'], ('word', None))

        batch_surfaces.append((
            morpheme_id,
            lang_id,
            rec['lemma'],
            morpheme_type,
            pos,
            SOURCE_NAME,
        ))

        if len(batch_surfaces) >= batch_size:
            cur.executemany("""
                INSERT OR IGNORE INTO surface_forms
                (morpheme_id, language_id, surface_form, morpheme_type, pos_tag, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch_surfaces)
            surface_count += cur.rowcount
            conn.commit()
            batch_surfaces = []

    # Final surface batch
    if batch_surfaces:
        cur.executemany("""
            INSERT OR IGNORE INTO surface_forms
            (morpheme_id, language_id, surface_form, morpheme_type, pos_tag, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch_surfaces)
        surface_count += cur.rowcount
        conn.commit()

    return morpheme_count, surface_count


def main():
    print(f"Importing Wikidata Lexemes from {REF_PATH}")

    if not REF_PATH.exists():
        print(f"ERROR: Reference directory not found: {REF_PATH}")
        print("Run: python download_sources.py --base-dir ../reference --sources wikidata")
        return 1

    csv_files = list(REF_PATH.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {REF_PATH}")
        return 1

    print(f"  Found {len(csv_files)} Wikidata CSV files")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        cur = conn.cursor()
        source_id = get_or_create_source(cur)
        conn.commit()

        total_morphemes = 0
        total_surfaces = 0

        for csv_path in sorted(csv_files):
            # Extract language code from filename (e.g., "DE_Q188_213917_03_2024.csv" -> "DE")
            lang_code = csv_path.name.split('_')[0]
            print(f"  Importing {lang_code} from {csv_path.name}...", end=" ")

            morphemes, surfaces = import_wikidata_file(conn, csv_path, lang_code, source_id)
            total_morphemes += morphemes
            total_surfaces += surfaces
            print(f"{morphemes} morphemes, {surfaces} surface forms")

        print(f"\n  Total morphemes: {total_morphemes}")
        print(f"  Total surface forms: {total_surfaces}")

        # Validation
        cur.execute("SELECT COUNT(*) FROM morphemes")
        total = cur.fetchone()[0]
        print(f"\n  Morphemes table now has: {total} entries")

        cur.execute("""
            SELECT morpheme_type, COUNT(*) as count
            FROM morphemes
            WHERE morpheme_type IS NOT NULL
            GROUP BY morpheme_type
            ORDER BY count DESC
        """)
        print("\n  Morphemes by type:")
        for mtype, count in cur.fetchall():
            print(f"    {mtype}: {count}")

    finally:
        conn.close()

    print("\nWikidata import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
