#!/usr/bin/env python3
"""
Import Grambank CLDF data.

Populates language_features in language_registry.db with grammatical features.

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py and import_glottolog.py to have been run first
"""

import csv
import sqlite3
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
LANGUAGE_REGISTRY_DB = DB_DIR / "language_registry.db"


def find_csv_file(directory: Path, patterns: list) -> Path:
    """Find a CSV file matching one of the patterns."""
    for pattern in patterns:
        matches = list(directory.rglob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def main():
    print("=" * 60)
    print("Importing Grambank Grammatical Features")
    print("=" * 60)

    if not LANGUAGE_REGISTRY_DB.exists():
        print(f"ERROR: {LANGUAGE_REGISTRY_DB} not found.")
        print("Run init_schemas.py and import_glottolog.py first.")
        return 1

    # Find Grambank data
    grambank_dir = REF_DIR / "grambank"
    if not grambank_dir.exists():
        print(f"ERROR: {grambank_dir} not found.")
        return 1

    values_file = find_csv_file(grambank_dir, ["values.csv", "*value*.csv"])
    codes_file = find_csv_file(grambank_dir, ["codes.csv", "*code*.csv"])
    params_file = find_csv_file(grambank_dir, ["parameters.csv", "*parameter*.csv"])

    if not values_file:
        print(f"ERROR: No values.csv found in {grambank_dir}")
        return 1

    print(f"\nFound: {values_file}")

    conn = sqlite3.connect(LANGUAGE_REGISTRY_DB)
    cursor = conn.cursor()

    # Build glottocode -> lang_id mapping
    cursor.execute("SELECT id, glottocode FROM language_codes WHERE glottocode IS NOT NULL")
    glotto_to_id = {row[1]: row[0] for row in cursor.fetchall()}
    print(f"Loaded {len(glotto_to_id)} language mappings")

    # Load parameter names if available
    param_names = {}
    if params_file:
        print("Reading parameter definitions...")
        with open(params_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_id = row.get("ID", row.get("id", ""))
                name = row.get("Name", row.get("name", ""))
                if param_id and name:
                    param_names[param_id] = name
        print(f"  Loaded {len(param_names)} parameters")

    # Load code values if available
    value_names = {}
    if codes_file:
        print("Reading value codes...")
        with open(codes_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code_id = row.get("ID", row.get("id", ""))
                name = row.get("Name", row.get("name", ""))
                if code_id and name:
                    value_names[code_id] = name
        print(f"  Loaded {len(value_names)} codes")

    # Clear existing Grambank features
    cursor.execute("DELETE FROM language_features WHERE source = 'grambank'")

    # Load and insert values
    print("Reading Grambank values...")
    inserted = 0
    skipped = 0
    skipped_unknown = 0

    with open(values_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        lang_col = next((c for c in fieldnames if 'language' in c.lower() and 'id' in c.lower()), None)
        param_col = next((c for c in fieldnames if 'parameter' in c.lower() and 'id' in c.lower()), None)
        value_col = next((c for c in fieldnames if c.lower() == 'value'), None)
        code_col = next((c for c in fieldnames if 'code' in c.lower() and 'id' in c.lower()), None)

        batch = []
        for row in reader:
            lang_glotto = row.get(lang_col, "") if lang_col else ""
            feature_id = row.get(param_col, "") if param_col else ""
            value = row.get(value_col, "") if value_col else ""
            code_id = row.get(code_col, "") if code_col else ""

            if not lang_glotto or not feature_id:
                continue

            # Skip question marks and empty values
            if value in ["?", ""]:
                skipped_unknown += 1
                continue

            # Map glottocode to lang_id
            lang_id = glotto_to_id.get(lang_glotto)
            if not lang_id:
                skipped += 1
                continue

            # Get value name - for Grambank binary features
            if code_id and code_id in value_names:
                value_name = value_names[code_id]
            elif value == "1":
                value_name = "Present"
            elif value == "0":
                value_name = "Absent"
            else:
                value_name = value

            batch.append({
                "lang_id": lang_id,
                "feature_id": f"GB_{feature_id}",
                "value": value,
                "value_name": value_name,
                "source": "grambank",
            })

            if len(batch) >= 10000:
                cursor.executemany("""
                    INSERT INTO language_features (lang_id, feature_id, value, value_name, source)
                    VALUES (:lang_id, :feature_id, :value, :value_name, :source)
                """, batch)
                inserted += len(batch)
                print(f"  Inserted {inserted}...")
                batch = []

        if batch:
            cursor.executemany("""
                INSERT INTO language_features (lang_id, feature_id, value, value_name, source)
                VALUES (:lang_id, :feature_id, :value, :value_name, :source)
            """, batch)
            inserted += len(batch)

    conn.commit()

    print("\n" + "=" * 60)
    print("Grambank import complete!")
    print("=" * 60)
    print(f"\nFeatures inserted: {inserted}")
    print(f"Skipped (no language match): {skipped}")
    print(f"Skipped (unknown values): {skipped_unknown}")
    print(f"\nDatabase: {LANGUAGE_REGISTRY_DB}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
