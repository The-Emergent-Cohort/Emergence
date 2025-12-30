#!/usr/bin/env python3
"""Import Grambank grammar features into tokenizer database"""

import csv
import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "grambank"

SOURCE_NAME = "grambank"
SOURCE_CONFIDENCE = 0.90

# Key Grambank features mapped to our grammar rule types
# Format: grambank_id -> (rule_type, rule_name, description)
GRAMBANK_FEATURES = {
    'GB020': ('morphology', 'has_inflectional_prefixes', 'Are there inflectional prefixes?'),
    'GB021': ('morphology', 'has_inflectional_suffixes', 'Are there inflectional suffixes?'),
    'GB022': ('morphology', 'has_derivational_prefixes', 'Are there derivational prefixes?'),
    'GB023': ('morphology', 'has_derivational_suffixes', 'Are there derivational suffixes?'),
    'GB024': ('case', 'has_case_marking', 'Is there morphological case marking on core arguments?'),
    'GB025': ('case', 'case_marking_type', 'What type of case marking?'),
    'GB065': ('agreement', 'verb_person_marking', 'Does the verb agree with person of subject?'),
    'GB066': ('agreement', 'verb_number_marking', 'Does the verb agree with number of subject?'),
    'GB130': ('word_order', 'basic_word_order', 'What is the basic word order?'),
    'GB131': ('word_order', 'flexible_word_order', 'Is word order flexible?'),
    'GB070': ('tense', 'has_tense_marking', 'Is there grammatical marking of tense?'),
    'GB071': ('aspect', 'has_aspect_marking', 'Is there grammatical marking of aspect?'),
    'GB074': ('mood', 'has_evidentiality', 'Is there grammatical marking of evidentiality?'),
    'GB080': ('definiteness', 'has_definite_article', 'Is there a definite article?'),
    'GB081': ('definiteness', 'has_indefinite_article', 'Is there an indefinite article?'),
    'GB090': ('number', 'noun_number_obligatory', 'Is number marking obligatory on nouns?'),
    'GB091': ('number', 'plural_marking_type', 'How is plural marked on nouns?'),
    'GB103': ('possession', 'has_alienability_distinction', 'Alienable vs inalienable possession?'),
    'GB110': ('negation', 'standard_negation', 'How is standard negation expressed?'),
    'GB120': ('questions', 'polar_question_marking', 'How are polar questions marked?'),
}


def find_cldf_dir(ref_path: Path) -> Path:
    """Find the cldf directory"""
    for pattern in ['cldf', '**/cldf', 'grambank-*']:
        matches = list(ref_path.glob(pattern))
        if matches:
            # Prefer the one with values.csv
            for m in matches:
                if (m / 'values.csv').exists():
                    return m
            return matches[0]
    return ref_path


def get_or_create_source(cur) -> int:
    """Register this source in data_sources table"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence)
        VALUES (?, ?, ?)
    """, (SOURCE_NAME, 'corpus', SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def load_parameters(cldf_dir: Path) -> dict:
    """Load parameter descriptions"""
    params = {}
    param_file = cldf_dir / "parameters.csv"

    if param_file.exists():
        with open(param_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get('ID') or row.get('id')
                name = row.get('Name') or row.get('name') or ''
                desc = row.get('Description') or row.get('description') or ''
                params[pid] = {'name': name, 'description': desc}

    return params


def parse_values(cldf_dir: Path, params: dict):
    """Parse Grambank values.csv file"""
    values_file = cldf_dir / "values.csv"

    if not values_file.exists():
        raise FileNotFoundError(f"No values.csv found in {cldf_dir}")

    print(f"  Reading {values_file.name}...")

    with open(values_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_id = row.get('Parameter_ID') or row.get('parameter_id')

            # Only process features we care about
            if param_id not in GRAMBANK_FEATURES:
                continue

            lang_id = row.get('Language_ID') or row.get('language_id')
            value = row.get('Value') or row.get('value')

            rule_type, rule_name, default_desc = GRAMBANK_FEATURES[param_id]
            param_info = params.get(param_id, {})

            yield {
                'glottolog_code': lang_id,
                'rule_type': rule_type,
                'rule_name': rule_name,
                'grambank_id': param_id,
                'value': value,
                'description': param_info.get('description', default_desc),
            }


def import_grambank(conn, cldf_dir: Path, source_id: int):
    """Import Grambank data into grammar_rules table"""
    cur = conn.cursor()

    params = load_parameters(cldf_dir)
    print(f"  Loaded {len(params)} parameter definitions")

    # Get language ID lookup
    cur.execute("SELECT id, glottolog_code FROM language_families WHERE glottolog_code IS NOT NULL")
    glotto_to_id = {row[1]: row[0] for row in cur.fetchall()}
    print(f"  Found {len(glotto_to_id)} languages with glottolog codes")

    # Import grammar rules
    insert_count = 0
    skip_count = 0

    for rec in parse_values(cldf_dir, params):
        # Look up language ID
        family_id = glotto_to_id.get(rec['glottolog_code'])
        if not family_id:
            skip_count += 1
            continue

        # Convert Grambank binary values to our format
        # Grambank: 0 = no, 1 = yes, ? = unknown
        if rec['value'] == '?':
            continue  # Skip unknown values

        # Determine bias based on value
        # If feature is present (1), positive bias; if absent (0), negative bias
        formal_weight = 0.8 if rec['value'] == '1' else 0.2
        nlp_weight = 0.7 if rec['value'] == '1' else 0.3

        cur.execute("""
            INSERT OR IGNORE INTO grammar_rules
            (family_id, rule_type, rule_name, abstract_form, formal_weight, nlp_weight, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            family_id,
            rec['rule_type'],
            rec['rule_name'],
            f"{rec['grambank_id']}={rec['value']}",
            formal_weight,
            nlp_weight,
            rec['description'],
        ))
        insert_count += cur.rowcount

        if insert_count % 10000 == 0:
            conn.commit()
            print(f"    Inserted {insert_count} rules...")

    conn.commit()
    print(f"  Inserted {insert_count} grammar rules (skipped {skip_count} - no matching language)")

    # Track provenance for new rules
    cur.execute("""
        INSERT OR IGNORE INTO provenance (table_name, row_id, source_id, confidence)
        SELECT 'grammar_rules', id, ?, ?
        FROM grammar_rules
        WHERE abstract_form LIKE 'GB%'
    """, (source_id, SOURCE_CONFIDENCE))
    conn.commit()


def main():
    print(f"Importing Grambank from {REF_PATH}")

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

        import_grambank(conn, cldf_dir, source_id)

        # Validation
        cur.execute("""
            SELECT rule_type, COUNT(*)
            FROM grammar_rules
            GROUP BY rule_type
            ORDER BY COUNT(*) DESC
        """)
        print("\n  Grammar rules by type:")
        for rule_type, count in cur.fetchall():
            print(f"    {rule_type}: {count}")

    finally:
        conn.close()

    print("\nGrambank import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
