#!/usr/bin/env python3
"""Import UniMorph inflection data into tokenizer database"""

import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "unimorph"

SOURCE_NAME = "unimorph"
SOURCE_CONFIDENCE = 0.85

# Language code to ISO 639-3 mapping
LANG_CODE_MAP = {
    'eng': 'eng',
    'deu': 'deu',
    'fra': 'fra',
    'spa': 'spa',
    'ita': 'ita',
    'por': 'por',
    'rus': 'rus',
    'ara': 'ara',
    'heb': 'heb',
    'jpn': 'jpn',
    'zho': 'zho',
    'kor': 'kor',
    'hin': 'hin',
    'tur': 'tur',
    'pol': 'pol',
    'nld': 'nld',
    'swe': 'swe',
    'fin': 'fin',
    'hun': 'hun',
    'ces': 'ces',
}

# UniMorph feature to readable mapping
FEATURE_MAP = {
    # POS
    'V': 'verb',
    'N': 'noun',
    'ADJ': 'adjective',
    'ADV': 'adverb',
    # Tense
    'PRS': 'present',
    'PST': 'past',
    'FUT': 'future',
    # Aspect
    'IPFV': 'imperfective',
    'PFV': 'perfective',
    'PRF': 'perfect',
    'PROG': 'progressive',
    # Mood
    'IND': 'indicative',
    'SBJV': 'subjunctive',
    'IMP': 'imperative',
    'COND': 'conditional',
    # Person
    '1': 'first_person',
    '2': 'second_person',
    '3': 'third_person',
    # Number
    'SG': 'singular',
    'PL': 'plural',
    'DU': 'dual',
    # Gender
    'MASC': 'masculine',
    'FEM': 'feminine',
    'NEUT': 'neuter',
    # Case
    'NOM': 'nominative',
    'ACC': 'accusative',
    'GEN': 'genitive',
    'DAT': 'dative',
    'INS': 'instrumental',
    'LOC': 'locative',
    'ABL': 'ablative',
    'VOC': 'vocative',
    # Voice
    'ACT': 'active',
    'PASS': 'passive',
    # Finiteness
    'NFIN': 'non_finite',
    'FIN': 'finite',
    # Participles
    'V.PTCP': 'participle',
    'V.MSDR': 'masdar',
    'V.CVB': 'converb',
}


def get_or_create_source(cur) -> int:
    """Register this source in data_sources table"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence)
        VALUES (?, ?, ?)
    """, (SOURCE_NAME, 'corpus', SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def get_or_create_language(cur, lang_code: str) -> int:
    """Get or create language entry"""
    iso_code = LANG_CODE_MAP.get(lang_code, lang_code)

    cur.execute("SELECT id FROM language_families WHERE iso_639_3 = ?", (iso_code,))
    result = cur.fetchone()
    if result:
        return result[0]

    # Create language entry if not exists
    cur.execute("""
        INSERT INTO language_families (name, level, iso_639_3, source)
        VALUES (?, 'language', ?, 'unimorph')
    """, (lang_code.upper(), iso_code))
    return cur.lastrowid


def parse_unimorph_file(file_path: Path):
    """Parse a UniMorph TSV file"""
    with open(file_path, encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            lemma = parts[0]
            surface = parts[1]
            features = parts[2]

            yield {
                'lemma': lemma,
                'surface_form': surface,
                'features': features,
                'line_num': line_num,
            }


def parse_features(feature_str: str) -> dict:
    """Parse UniMorph feature bundle"""
    result = {
        'raw': feature_str,
        'pos': None,
        'tense': None,
        'aspect': None,
        'mood': None,
        'person': None,
        'number': None,
        'gender': None,
        'case': None,
        'voice': None,
    }

    parts = feature_str.split(';')
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check against feature map
        readable = FEATURE_MAP.get(part)
        if readable:
            # Categorize the feature
            if part in ['V', 'N', 'ADJ', 'ADV']:
                result['pos'] = readable
            elif part in ['PRS', 'PST', 'FUT']:
                result['tense'] = readable
            elif part in ['IPFV', 'PFV', 'PRF', 'PROG']:
                result['aspect'] = readable
            elif part in ['IND', 'SBJV', 'IMP', 'COND']:
                result['mood'] = readable
            elif part in ['1', '2', '3']:
                result['person'] = readable
            elif part in ['SG', 'PL', 'DU']:
                result['number'] = readable
            elif part in ['MASC', 'FEM', 'NEUT']:
                result['gender'] = readable
            elif part in ['NOM', 'ACC', 'GEN', 'DAT', 'INS', 'LOC', 'ABL', 'VOC']:
                result['case'] = readable
            elif part in ['ACT', 'PASS']:
                result['voice'] = readable

    return result


def import_language(conn, lang_dir: Path, lang_code: str, source_id: int):
    """Import UniMorph data for a single language"""
    cur = conn.cursor()

    # Get language ID
    lang_id = get_or_create_language(cur, lang_code)

    # Find data files
    data_files = list(lang_dir.glob("*.txt")) + list(lang_dir.glob(lang_code))
    if not data_files:
        print(f"    No data files found for {lang_code}")
        return 0

    insert_count = 0
    batch = []
    batch_size = 1000

    for data_file in data_files:
        for rec in parse_unimorph_file(data_file):
            features = parse_features(rec['features'])

            batch.append((
                None,  # morpheme_id - linked separately
                lang_id,
                rec['surface_form'],
                0,  # frequency - UniMorph doesn't have this
                features['pos'],
                rec['features'],  # Store full feature bundle
                SOURCE_NAME,
            ))

            if len(batch) >= batch_size:
                cur.executemany("""
                    INSERT OR IGNORE INTO surface_forms
                    (morpheme_id, language_id, surface_form, frequency, pos_tag, pos_features, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, batch)
                insert_count += cur.rowcount
                conn.commit()
                batch = []

    # Final batch
    if batch:
        cur.executemany("""
            INSERT OR IGNORE INTO surface_forms
            (morpheme_id, language_id, surface_form, frequency, pos_tag, pos_features, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, batch)
        insert_count += cur.rowcount
        conn.commit()

    return insert_count


def main():
    print(f"Importing UniMorph from {REF_PATH}")

    if not REF_PATH.exists():
        print(f"ERROR: Reference directory not found: {REF_PATH}")
        print("Run download_sources.py first")
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        cur = conn.cursor()
        source_id = get_or_create_source(cur)
        conn.commit()

        # Process each language directory
        total_count = 0
        lang_dirs = [d for d in REF_PATH.iterdir() if d.is_dir()]

        if not lang_dirs:
            print("  No language directories found")
            return 1

        print(f"  Found {len(lang_dirs)} language directories")

        for lang_dir in sorted(lang_dirs):
            lang_code = lang_dir.name
            print(f"  Importing {lang_code}...", end=" ")

            count = import_language(conn, lang_dir, lang_code, source_id)
            total_count += count
            print(f"{count} forms")

        print(f"\n  Total surface forms imported: {total_count}")

        # Validation
        cur.execute("""
            SELECT lf.iso_639_3, COUNT(sf.id) as forms
            FROM language_families lf
            JOIN surface_forms sf ON sf.language_id = lf.id
            WHERE sf.source = ?
            GROUP BY lf.id
            ORDER BY forms DESC
        """, (SOURCE_NAME,))

        print("\n  Forms per language:")
        for iso, count in cur.fetchall():
            print(f"    {iso}: {count}")

    finally:
        conn.close()

    print("\nUniMorph import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
