#!/usr/bin/env python3
"""
Import Glottolog CLDF data into the language database.

Populates:
- languages: Language inventory with ISO codes, glottocodes
- language_families: Genealogical hierarchy

Usage:
    python import_glottolog.py                           # Default paths
    python import_glottolog.py --input reference/glottolog
    python import_glottolog.py --db db/language.db
"""

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "reference" / "glottolog"
DEFAULT_DB = SCRIPT_DIR / "db" / "language.db"


def find_csv_file(directory: Path, patterns: list) -> Path:
    """Find a CSV file matching one of the patterns."""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def import_glottolog(input_dir: Path, db_path: Path):
    """Import Glottolog data into SQLite."""

    print(f"Importing Glottolog data")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {db_path}")
    print()

    # Find the languages file
    lang_file = find_csv_file(input_dir, ["languages.csv", "*languages*.csv"])
    if not lang_file:
        print(f"Error: No languages.csv found in {input_dir}")
        sys.exit(1)

    print(f"Found: {lang_file.name}")

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    # Read and parse CSV
    families = {}  # glottocode -> family info
    languages = []

    print("Reading languages...", end="", flush=True)
    with open(lang_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Detect column names (Glottolog CLDF may vary)
        fieldnames = reader.fieldnames
        print(f" columns: {fieldnames[:5]}...")

        # Map columns
        id_col = next((c for c in fieldnames if c.lower() in ['id', 'glottocode']), None)
        name_col = next((c for c in fieldnames if c.lower() == 'name'), None)
        iso_col = next((c for c in fieldnames if 'iso' in c.lower()), None)
        family_col = next((c for c in fieldnames if 'family' in c.lower() or 'parent' in c.lower()), None)
        level_col = next((c for c in fieldnames if 'level' in c.lower()), None)
        macroarea_col = next((c for c in fieldnames if 'macroarea' in c.lower() or 'area' in c.lower()), None)
        lat_col = next((c for c in fieldnames if 'lat' in c.lower()), None)
        lon_col = next((c for c in fieldnames if 'lon' in c.lower() or 'lng' in c.lower()), None)

        for row in reader:
            glottocode = row.get(id_col, "") if id_col else ""
            name = row.get(name_col, "") if name_col else ""
            iso = row.get(iso_col, "") if iso_col else ""
            family_id = row.get(family_col, "") if family_col else ""
            level = row.get(level_col, "") if level_col else ""
            macroarea = row.get(macroarea_col, "") if macroarea_col else ""
            lat = row.get(lat_col, "") if lat_col else ""
            lon = row.get(lon_col, "") if lon_col else ""

            if not glottocode:
                continue

            # Determine if this is a family or a language
            if level == "family" or (not iso and level != "dialect"):
                families[glottocode] = {
                    "glottocode": glottocode,
                    "name": name,
                    "parent_id": family_id if family_id != glottocode else None,
                    "level": 0
                }
            else:
                # It's a language or dialect
                lang_code = iso if iso else glottocode[:3]
                languages.append({
                    "lang_code": lang_code,
                    "glottocode": glottocode,
                    "name": name,
                    "family_id": family_id,
                    "level": level,
                    "macroarea": macroarea,
                    "latitude": float(lat) if lat else None,
                    "longitude": float(lon) if lon else None,
                    "status": "living",  # Default, could parse from other fields
                    "speaker_count": None
                })

    print(f" Found {len(families)} families, {len(languages)} languages")

    # Insert families
    print("Inserting language families...", end="", flush=True)
    conn.executemany(
        """INSERT OR REPLACE INTO language_families
           (glottocode, name, parent_id, level)
           VALUES (:glottocode, :name, :parent_id, :level)""",
        list(families.values())
    )
    conn.commit()
    print(" done")

    # Insert languages
    print("Inserting languages...", end="", flush=True)
    conn.executemany(
        """INSERT OR REPLACE INTO languages
           (lang_code, glottocode, name, family_id, level, macroarea, latitude, longitude, status, speaker_count)
           VALUES (:lang_code, :glottocode, :name, :family_id, :level, :macroarea, :latitude, :longitude, :status, :speaker_count)""",
        languages
    )
    conn.commit()
    print(" done")

    # Record import
    conn.execute(
        """INSERT INTO import_metadata (source, record_count, notes)
           VALUES (?, ?, ?)""",
        ("glottolog", len(languages), f"families={len(families)}")
    )
    conn.commit()

    # Summary
    print(f"\n{'=' * 40}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 40}")
    print(f"  Families:  {len(families):,}")
    print(f"  Languages: {len(languages):,}")

    # Show some stats
    cursor = conn.execute("SELECT macroarea, COUNT(*) FROM languages GROUP BY macroarea ORDER BY COUNT(*) DESC LIMIT 5")
    print(f"\n  Top macroareas:")
    for row in cursor:
        print(f"    {row[0] or 'Unknown'}: {row[1]}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import Glottolog CLDF data"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input directory (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        default=DEFAULT_DB,
        help=f"Output database (default: {DEFAULT_DB})"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        print(f"\nRun download_typology.py first.")
        sys.exit(1)

    import_glottolog(args.input, args.db)


if __name__ == "__main__":
    main()
