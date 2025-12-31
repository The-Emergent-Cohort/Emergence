#!/usr/bin/env python3
"""
Import Grambank CLDF data.

Grambank provides 195 binary grammatical features for 2400+ languages.
Features are coded as 0/1/? for presence/absence/unknown.

Populates:
- feature_definitions: What each feature means
- language_features: Feature values per language

Usage:
    python import_grambank.py                           # Default paths
    python import_grambank.py --input reference/grambank
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "reference" / "grambank"
DEFAULT_DB = SCRIPT_DIR / "db" / "language.db"


def find_csv_file(directory: Path, patterns: list) -> Path:
    """Find a CSV file matching one of the patterns, searching recursively."""
    for pattern in patterns:
        # First try direct match
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
        # Then try recursive search
        matches = list(directory.glob(f"**/{pattern}"))
        if matches:
            return matches[0]
    return None


def import_grambank(input_dir: Path, db_path: Path):
    """Import Grambank data into SQLite."""

    print(f"Importing Grambank data")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {db_path}")
    print()

    # Find files
    params_file = find_csv_file(input_dir, ["parameters.csv", "*parameter*.csv"])
    values_file = find_csv_file(input_dir, ["values.csv", "*value*.csv"])

    if not values_file:
        print(f"Error: No values.csv found in {input_dir}")
        sys.exit(1)

    print(f"Found: {values_file.name}")
    if params_file:
        print(f"Found: {params_file.name}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    # Load feature definitions
    features = {}
    if params_file:
        print("\nReading feature definitions...", end="", flush=True)
        with open(params_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_id = row.get("ID", row.get("id", ""))
                name = row.get("Name", row.get("name", ""))
                desc = row.get("Description", row.get("description", ""))

                if feature_id:
                    # Grambank features are GB### format
                    # Infer domain from feature description
                    domain = "grammar"
                    name_lower = name.lower() if name else ""
                    desc_lower = desc.lower() if desc else ""
                    combined = name_lower + " " + desc_lower

                    if any(x in combined for x in ["order", "position", "preverb", "postverb"]):
                        domain = "word_order"
                    elif any(x in combined for x in ["case", "ergative", "accusative", "nominative"]):
                        domain = "case"
                    elif any(x in combined for x in ["gender", "noun class", "classifier"]):
                        domain = "nominal"
                    elif any(x in combined for x in ["tense", "aspect", "mood", "verb"]):
                        domain = "verbal"
                    elif any(x in combined for x in ["negat"]):
                        domain = "negation"
                    elif any(x in combined for x in ["affix", "prefix", "suffix", "inflect"]):
                        domain = "morphology"
                    elif any(x in combined for x in ["plural", "number", "dual"]):
                        domain = "number"
                    elif any(x in combined for x in ["definite", "article", "demonstrat"]):
                        domain = "definiteness"

                    features[feature_id] = {
                        "feature_id": feature_id,
                        "name": name,
                        "description": desc,
                        "domain": domain,
                        "source": "grambank",
                        "possible_values": json.dumps(["0", "1", "?"])  # Binary + unknown
                    }
        print(f" {len(features)} features")

    # Insert feature definitions
    if features:
        print("Inserting feature definitions...", end="", flush=True)
        conn.executemany(
            """INSERT OR REPLACE INTO feature_definitions
               (feature_id, name, description, domain, source, possible_values)
               VALUES (:feature_id, :name, :description, :domain, :source, :possible_values)""",
            list(features.values())
        )
        conn.commit()
        print(" done")

    # Load language feature values
    print("Reading language feature values...", end="", flush=True)
    values = []
    skipped_unknown = 0

    with open(values_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        lang_col = next((c for c in fieldnames if 'language' in c.lower() and 'id' in c.lower()), None)
        param_col = next((c for c in fieldnames if 'parameter' in c.lower() and 'id' in c.lower()), None)
        value_col = next((c for c in fieldnames if c.lower() == 'value'), None)

        for row in reader:
            lang_code = row.get(lang_col, "") if lang_col else ""
            feature_id = row.get(param_col, "") if param_col else ""
            value = row.get(value_col, "") if value_col else ""

            if not lang_code or not feature_id:
                continue

            # Skip unknown values (?) to save space - they don't add info
            if value == "?":
                skipped_unknown += 1
                continue

            # Map binary to human-readable
            value_name = "Present" if value == "1" else "Absent" if value == "0" else value

            values.append({
                "lang_code": lang_code,
                "feature_id": feature_id,
                "value": value,
                "value_name": value_name,
                "source": "grambank",
                "confidence": 1.0
            })

    print(f" {len(values)} values (skipped {skipped_unknown} unknown)")

    print("Inserting language features...", end="", flush=True)
    batch_size = 10000
    for i in range(0, len(values), batch_size):
        batch = values[i:i + batch_size]
        conn.executemany(
            """INSERT OR REPLACE INTO language_features
               (lang_code, feature_id, value, value_name, source, confidence)
               VALUES (:lang_code, :feature_id, :value, :value_name, :source, :confidence)""",
            batch
        )
        conn.commit()
        print(".", end="", flush=True)
    print(" done")

    # Record import
    conn.execute(
        """INSERT INTO import_metadata (source, record_count, notes)
           VALUES (?, ?, ?)""",
        ("grambank", len(values), f"features={len(features)}, skipped_unknown={skipped_unknown}")
    )
    conn.commit()

    # Summary
    print(f"\n{'=' * 40}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 40}")
    print(f"  Features:       {len(features):,}")
    print(f"  Feature values: {len(values):,}")
    print(f"  Skipped (?):    {skipped_unknown:,}")

    # Show domain breakdown
    cursor = conn.execute(
        "SELECT domain, COUNT(*) FROM feature_definitions WHERE source='grambank' GROUP BY domain ORDER BY COUNT(*) DESC"
    )
    print(f"\n  Features by domain:")
    for row in cursor:
        print(f"    {row[0]}: {row[1]}")

    # Show coverage
    cursor = conn.execute(
        "SELECT COUNT(DISTINCT lang_code) FROM language_features WHERE source='grambank'"
    )
    lang_count = cursor.fetchone()[0]
    print(f"\n  Languages with Grambank data: {lang_count}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import Grambank CLDF data"
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

    import_grambank(args.input, args.db)


if __name__ == "__main__":
    main()
