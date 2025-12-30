#!/usr/bin/env python3
"""
Import WALS (World Atlas of Language Structures) CLDF data.

Populates:
- feature_definitions: What each feature means
- language_features: Feature values per language

Usage:
    python import_wals.py                           # Default paths
    python import_wals.py --input reference/wals
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "reference" / "wals"
DEFAULT_DB = SCRIPT_DIR / "db" / "language.db"


def find_csv_file(directory: Path, patterns: list) -> Path:
    """Find a CSV file matching one of the patterns."""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def import_wals(input_dir: Path, db_path: Path):
    """Import WALS data into SQLite."""

    print(f"Importing WALS data")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {db_path}")
    print()

    # Find files
    params_file = find_csv_file(input_dir, ["parameters.csv", "*parameter*.csv"])
    values_file = find_csv_file(input_dir, ["values.csv", "*value*.csv"])
    codes_file = find_csv_file(input_dir, ["codes.csv", "*code*.csv"])

    if not values_file:
        print(f"Error: No values.csv found in {input_dir}")
        sys.exit(1)

    print(f"Found: {values_file.name}")
    if params_file:
        print(f"Found: {params_file.name}")
    if codes_file:
        print(f"Found: {codes_file.name}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    # Load feature definitions from parameters.csv
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
                    # Infer domain from feature ID or name
                    domain = "other"
                    name_lower = name.lower()
                    if "order" in name_lower:
                        domain = "word_order"
                    elif "case" in name_lower:
                        domain = "case"
                    elif "gender" in name_lower or "noun" in name_lower or "plural" in name_lower:
                        domain = "nominal"
                    elif "verb" in name_lower or "tense" in name_lower or "aspect" in name_lower:
                        domain = "verbal"
                    elif "negat" in name_lower:
                        domain = "negation"
                    elif any(x in name_lower for x in ["morpho", "affix", "prefix", "suffix"]):
                        domain = "morphology"

                    features[feature_id] = {
                        "feature_id": feature_id,
                        "name": name,
                        "description": desc,
                        "domain": domain,
                        "source": "wals",
                        "possible_values": None
                    }
        print(f" {len(features)} features")

    # Load code values if available (maps value IDs to names)
    value_names = {}
    if codes_file:
        print("Reading value codes...", end="", flush=True)
        with open(codes_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code_id = row.get("ID", row.get("id", ""))
                name = row.get("Name", row.get("name", ""))
                param_id = row.get("Parameter_ID", row.get("parameter_id", ""))
                if code_id and name:
                    value_names[code_id] = name
                    # Also track possible values per feature
                    if param_id and param_id in features:
                        if not features[param_id]["possible_values"]:
                            features[param_id]["possible_values"] = []
                        features[param_id]["possible_values"].append(name)
        print(f" {len(value_names)} codes")

    # Convert possible_values to JSON
    for f in features.values():
        if f["possible_values"]:
            f["possible_values"] = json.dumps(f["possible_values"])

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

    # Load and insert language feature values
    print("Reading language feature values...", end="", flush=True)
    values = []
    with open(values_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        lang_col = next((c for c in fieldnames if 'language' in c.lower() and 'id' in c.lower()), None)
        param_col = next((c for c in fieldnames if 'parameter' in c.lower() and 'id' in c.lower()), None)
        value_col = next((c for c in fieldnames if c.lower() == 'value'), None)
        code_col = next((c for c in fieldnames if 'code' in c.lower() and 'id' in c.lower()), None)

        for row in reader:
            lang_code = row.get(lang_col, "") if lang_col else ""
            feature_id = row.get(param_col, "") if param_col else ""
            value = row.get(value_col, "") if value_col else ""
            code_id = row.get(code_col, "") if code_col else ""

            if not lang_code or not feature_id:
                continue

            # Get human-readable value name
            value_name = value_names.get(code_id, value)

            values.append({
                "lang_code": lang_code,
                "feature_id": feature_id,
                "value": value,
                "value_name": value_name,
                "source": "wals",
                "confidence": 1.0
            })

    print(f" {len(values)} values")

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
        ("wals", len(values), f"features={len(features)}")
    )
    conn.commit()

    # Summary
    print(f"\n{'=' * 40}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 40}")
    print(f"  Features:      {len(features):,}")
    print(f"  Feature values: {len(values):,}")

    # Show domain breakdown
    cursor = conn.execute(
        "SELECT domain, COUNT(*) FROM feature_definitions WHERE source='wals' GROUP BY domain ORDER BY COUNT(*) DESC"
    )
    print(f"\n  Features by domain:")
    for row in cursor:
        print(f"    {row[0]}: {row[1]}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import WALS CLDF data"
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

    import_wals(args.input, args.db)


if __name__ == "__main__":
    main()
