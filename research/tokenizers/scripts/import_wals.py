#!/usr/bin/env python3
"""Import WALS typological features into tokenizer database"""

import csv
import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "wals"

SOURCE_NAME = "wals"
SOURCE_CONFIDENCE = 0.90

# Key WALS features we care about
WALS_FEATURES = {
    '81A': 'word_order',           # Order of Subject, Object, and Verb
    '82A': 'subject_verb_order',   # Order of Subject and Verb
    '83A': 'object_verb_order',    # Order of Object and Verb
    '85A': 'adposition_order',     # Order of Adposition and NP
    '86A': 'genitive_order',       # Order of Genitive and Noun
    '87A': 'adjective_order',      # Order of Adjective and Noun
    '26A': 'morphology_type',      # Prefixing vs Suffixing
    '20A': 'fusion_type',          # Fusion of Selected Inflectional Formatives
}

# Word order mappings
WORD_ORDER_MAP = {
    'SOV': 'SOV',
    'SVO': 'SVO',
    'VSO': 'VSO',
    'VOS': 'VOS',
    'OVS': 'OVS',
    'OSV': 'OSV',
    'No dominant order': 'free',
}


def find_cldf_dir(ref_path: Path) -> Path:
    """Find the cldf directory"""
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


def load_parameters(cldf_dir: Path) -> dict:
    """Load parameter (feature) definitions"""
    params = {}
    param_file = cldf_dir / "parameters.csv"

    if not param_file.exists():
        for alt in ["Parameter.csv", "parameters.tsv"]:
            alt_file = cldf_dir / alt
            if alt_file.exists():
                param_file = alt_file
                break

    if param_file.exists():
        with open(param_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('ID') or row.get('id')
                params[pid] = row.get('Name') or row.get('name') or ''

    return params


def load_codes(cldf_dir: Path) -> dict:
    """Load value code definitions"""
    codes = {}
    code_file = cldf_dir / "codes.csv"

    if not code_file.exists():
        for alt in ["Code.csv", "codes.tsv"]:
            alt_file = cldf_dir / alt
            if alt_file.exists():
                code_file = alt_file
                break

    if code_file.exists():
        with open(code_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = row.get('Parameter_ID') or row.get('parameter_id')
                code = row.get('ID') or row.get('id') or row.get('Number')
                name = row.get('Name') or row.get('name') or row.get('Description')
                if param and code:
                    codes[(param, code)] = name

    return codes


def parse_values(cldf_dir: Path, codes: dict):
    """Parse WALS values.csv file"""
    values_file = cldf_dir / "values.csv"

    if not values_file.exists():
        for alt in ["Value.csv", "values.tsv"]:
            alt_file = cldf_dir / alt
            if alt_file.exists():
                values_file = alt_file
                break

    if not values_file.exists():
        raise FileNotFoundError(f"No values file found in {cldf_dir}")

    print(f"  Reading {values_file.name}...")

    with open(values_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_id = row.get('Parameter_ID') or row.get('parameter_id')

            # Only process features we care about
            if param_id not in WALS_FEATURES:
                continue

            lang_id = row.get('Language_ID') or row.get('language_id')
            code_id = row.get('Code_ID') or row.get('code_id') or row.get('Value')

            # Get the human-readable value
            value_name = codes.get((param_id, code_id), code_id)

            yield {
                'language_id': lang_id,  # This is often wals_code
                'feature': WALS_FEATURES[param_id],
                'feature_id': param_id,
                'value': value_name,
            }


def import_wals(conn, cldf_dir: Path, source_id: int):
    """Import WALS data, updating language_families"""
    cur = conn.cursor()

    codes = load_codes(cldf_dir)
    print(f"  Loaded {len(codes)} value codes")

    # Collect features by language
    lang_features = {}
    for rec in parse_values(cldf_dir, codes):
        lang_id = rec['language_id']
        if lang_id not in lang_features:
            lang_features[lang_id] = {'wals_code': lang_id}
        lang_features[lang_id][rec['feature']] = rec['value']

    print(f"  Found features for {len(lang_features)} languages")

    # Update language_families with WALS data
    update_count = 0
    for wals_code, features in lang_features.items():
        # Build update query dynamically
        updates = ["wals_code = ?"]
        values = [wals_code]

        if 'word_order' in features:
            updates.append("default_word_order = ?")
            values.append(WORD_ORDER_MAP.get(features['word_order'], features['word_order']))

        if 'morphology_type' in features:
            morph = features['morphology_type']
            if 'suffix' in morph.lower():
                values.append('suffixing')
            elif 'prefix' in morph.lower():
                values.append('prefixing')
            else:
                values.append(morph)
            updates.append("default_morphology_type = ?")

        # Try to match by glottolog code first, then by name
        # WALS uses its own codes, so we may need fuzzy matching
        cur.execute(f"""
            UPDATE language_families
            SET {', '.join(updates)}
            WHERE wals_code = ? OR glottolog_code LIKE ?
        """, values + [wals_code, f"%{wals_code}%"])

        if cur.rowcount > 0:
            update_count += cur.rowcount

    conn.commit()
    print(f"  Updated {update_count} language records with WALS data")


def main():
    print(f"Importing WALS from {REF_PATH}")

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

        import_wals(conn, cldf_dir, source_id)

        # Validation
        cur.execute("""
            SELECT default_word_order, COUNT(*)
            FROM language_families
            WHERE default_word_order IS NOT NULL
            GROUP BY default_word_order
        """)
        print("\n  Word order distribution:")
        for order, count in cur.fetchall():
            print(f"    {order}: {count}")

    finally:
        conn.close()

    print("\nWALS import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
