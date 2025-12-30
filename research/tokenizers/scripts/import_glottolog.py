#!/usr/bin/env python3
"""Import Glottolog language data into tokenizer database"""

import csv
import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "glottolog"

SOURCE_NAME = "glottolog"
SOURCE_CONFIDENCE = 0.95  # High confidence - authoritative source


def find_cldf_dir(ref_path: Path) -> Path:
    """Find the cldf directory within the extracted data"""
    # Look for cldf subdirectory
    for pattern in ['cldf', 'cldf-datasets-*', '**/cldf']:
        matches = list(ref_path.glob(pattern))
        if matches:
            return matches[0]
    return ref_path


def get_or_create_source(cur) -> int:
    """Register this source in data_sources table"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence)
        VALUES (?, ?, ?)
    """, (SOURCE_NAME, 'reference', SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def parse_languages(cldf_dir: Path):
    """Parse CLDF languages.csv file"""
    lang_file = cldf_dir / "languages.csv"

    if not lang_file.exists():
        # Try alternative names
        for alt in ["Language.csv", "languoids.csv"]:
            alt_file = cldf_dir / alt
            if alt_file.exists():
                lang_file = alt_file
                break

    if not lang_file.exists():
        raise FileNotFoundError(f"No language file found in {cldf_dir}")

    print(f"  Reading {lang_file.name}...")

    with open(lang_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                'glottolog_code': row.get('ID') or row.get('id') or row.get('Glottocode'),
                'name': row.get('Name') or row.get('name'),
                'iso_639_3': row.get('ISO639P3code') or row.get('iso639P3code') or row.get('ISO'),
                'parent_code': row.get('Family_ID') or row.get('parent_id') or row.get('Parent_ID'),
                'level': row.get('level') or row.get('Level') or 'language',
                'latitude': row.get('Latitude') or row.get('latitude'),
                'longitude': row.get('Longitude') or row.get('longitude'),
                'macroarea': row.get('Macroarea') or row.get('macroarea'),
            }


def import_languages(conn, cldf_dir: Path, source_id: int):
    """Import languages in two passes (for parent references)"""
    cur = conn.cursor()

    # First pass: insert all languages without parent_id
    print("  Pass 1: Inserting languages...")
    records = list(parse_languages(cldf_dir))
    print(f"  Found {len(records)} language records")

    batch = []
    for rec in records:
        # Skip if no glottolog code
        if not rec['glottolog_code']:
            continue

        batch.append((
            rec['name'],
            rec['level'],
            rec['iso_639_3'],
            None,  # iso_639_1 not in glottolog
            rec['glottolog_code'],
            None,  # wals_code added later
            None,  # default_word_order
            None,  # default_morphology_type
            SOURCE_NAME,
        ))

    cur.executemany("""
        INSERT OR IGNORE INTO language_families
        (name, level, iso_639_3, iso_639_1, glottolog_code, wals_code,
         default_word_order, default_morphology_type, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)

    conn.commit()
    print(f"  Inserted {cur.rowcount} languages")

    # Second pass: update parent_id references
    print("  Pass 2: Linking parent references...")

    # Build lookup table
    cur.execute("SELECT id, glottolog_code FROM language_families WHERE glottolog_code IS NOT NULL")
    code_to_id = {row[1]: row[0] for row in cur.fetchall()}

    update_count = 0
    for rec in records:
        if rec['parent_code'] and rec['glottolog_code']:
            parent_id = code_to_id.get(rec['parent_code'])
            if parent_id:
                cur.execute("""
                    UPDATE language_families
                    SET parent_id = ?
                    WHERE glottolog_code = ?
                """, (parent_id, rec['glottolog_code']))
                update_count += cur.rowcount

    conn.commit()
    print(f"  Updated {update_count} parent references")

    # Track provenance
    cur.execute("""
        INSERT INTO provenance (table_name, row_id, source_id, confidence)
        SELECT 'language_families', id, ?, ?
        FROM language_families
        WHERE source = ?
    """, (source_id, SOURCE_CONFIDENCE, SOURCE_NAME))
    conn.commit()


def main():
    print(f"Importing Glottolog from {REF_PATH}")

    if not REF_PATH.exists():
        print(f"ERROR: Reference directory not found: {REF_PATH}")
        print("Run download_sources.py first")
        return 1

    cldf_dir = find_cldf_dir(REF_PATH)
    print(f"  CLDF directory: {cldf_dir}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        cur = conn.cursor()
        source_id = get_or_create_source(cur)
        conn.commit()

        import_languages(conn, cldf_dir, source_id)

        # Validation
        cur.execute("SELECT level, COUNT(*) FROM language_families GROUP BY level")
        print("\n  Language counts by level:")
        for level, count in cur.fetchall():
            print(f"    {level}: {count}")

    finally:
        conn.close()

    print("\nGlottolog import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
