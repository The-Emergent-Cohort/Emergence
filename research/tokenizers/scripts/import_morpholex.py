#!/usr/bin/env python3
"""Import MorphoLex-en morpheme data into tokenizer database"""

import sqlite3
from pathlib import Path

try:
    import openpyxl
except ImportError:
    print("Missing openpyxl. Run: pip install openpyxl")
    exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
REF_PATH = BASE_DIR / "reference" / "morpholex"

SOURCE_NAME = "morpholex-en"
SOURCE_CONFIDENCE = 0.85

# Morpheme mappings - surface form to concept
# These map common English affixes to our core morpheme concepts
PREFIX_TO_CONCEPT = {
    'un': 'NEGATION',
    'in': 'NEGATION',
    'im': 'NEGATION',
    'il': 'NEGATION',
    'ir': 'NEGATION',
    'dis': 'NEGATION',
    'non': 'NEGATION',
    'de': 'NEGATION',
    'anti': 'AGAINST',
    're': 'AGAIN',
    'pre': 'BEFORE',
    'post': 'AFTER',
    'sub': 'UNDER',
    'super': 'ABOVE',
    'over': 'ABOVE',
    'under': 'UNDER',
    'inter': 'BETWEEN',
    'trans': 'ACROSS',
    'ex': 'OUT',
    'out': 'OUT',
    'in': 'INTO',
    'en': 'CAUSATIVE',
    'em': 'CAUSATIVE',
    'co': 'TOGETHER',
    'com': 'TOGETHER',
    'con': 'TOGETHER',
    'multi': 'MANY',
    'poly': 'MANY',
    'mono': 'ONE',
    'uni': 'ONE',
    'bi': 'TWO',
    'di': 'TWO',
    'tri': 'THREE',
    'semi': 'HALF',
    'auto': 'SELF',
    'self': 'SELF',
    'mis': 'WRONG',
    'mal': 'BAD',
    'bene': 'GOOD',
}

SUFFIX_TO_CONCEPT = {
    'er': 'AGENT_OF',
    'or': 'AGENT_OF',
    'ist': 'AGENT_OF',
    'ian': 'AGENT_OF',
    'ant': 'AGENT_OF',
    'ent': 'AGENT_OF',
    'ee': 'PATIENT_OF',
    'ness': 'STATE_OF',
    'ment': 'STATE_OF',
    'tion': 'STATE_OF',
    'sion': 'STATE_OF',
    'ity': 'STATE_OF',
    'ty': 'STATE_OF',
    'ance': 'STATE_OF',
    'ence': 'STATE_OF',
    'dom': 'STATE_OF',
    'hood': 'STATE_OF',
    'ship': 'STATE_OF',
    'ful': 'FULL_OF',
    'ous': 'QUALITY_OF',
    'ious': 'QUALITY_OF',
    'eous': 'QUALITY_OF',
    'al': 'QUALITY_OF',
    'ial': 'QUALITY_OF',
    'ic': 'QUALITY_OF',
    'ical': 'QUALITY_OF',
    'ive': 'QUALITY_OF',
    'less': 'WITHOUT',
    'able': 'CAPABLE_OF',
    'ible': 'CAPABLE_OF',
    'ly': 'MANNER',
    'ize': 'CAUSATIVE',
    'ise': 'CAUSATIVE',
    'ify': 'CAUSATIVE',
    'en': 'CAUSATIVE',
    'ate': 'CAUSATIVE',
    'ward': 'DIRECTION',
    'wards': 'DIRECTION',
    'wise': 'MANNER',
    'like': 'SIMILAR',
    'ish': 'SOMEWHAT',
    'most': 'SUPERLATIVE',
    'est': 'SUPERLATIVE',
}


def get_or_create_source(cur) -> int:
    """Register this source in data_sources table"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence)
        VALUES (?, ?, ?)
    """, (SOURCE_NAME, 'corpus', SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def get_english_language_id(cur) -> int:
    """Get or create English language entry"""
    cur.execute("SELECT id FROM language_families WHERE iso_639_3 = 'eng'")
    result = cur.fetchone()
    if result:
        return result[0]

    # Create English entry if not exists
    cur.execute("""
        INSERT INTO language_families (name, level, iso_639_3, source)
        VALUES ('English', 'language', 'eng', 'morpholex')
    """)
    return cur.lastrowid


def get_morpheme_id(cur, concept: str) -> int:
    """Get morpheme ID for a concept, or None if not found"""
    cur.execute("SELECT id FROM morphemes WHERE morpheme = ?", (concept,))
    result = cur.fetchone()
    return result[0] if result else None


def parse_morpholex(ref_path: Path):
    """Parse MorphoLex Excel file"""
    xlsx_file = ref_path / "MorphoLex_en.xlsx"

    if not xlsx_file.exists():
        # Try to find any xlsx file
        xlsx_files = list(ref_path.glob("*.xlsx"))
        if xlsx_files:
            xlsx_file = xlsx_files[0]
        else:
            raise FileNotFoundError(f"No Excel file found in {ref_path}")

    print(f"  Reading {xlsx_file.name}...")

    wb = openpyxl.load_workbook(xlsx_file, read_only=True)
    ws = wb.active

    # Get headers from first row
    headers = []
    for cell in next(ws.iter_rows(max_row=1)):
        headers.append(cell.value)

    # Find column indices
    word_col = None
    segm_col = None
    freq_col = None
    pos_col = None

    for i, h in enumerate(headers):
        if h and 'Word' in str(h):
            word_col = i
        elif h and 'MorphoLexSegm' in str(h):
            segm_col = i
        elif h and 'SUBTLEX' in str(h) and 'Freq' in str(h):
            freq_col = i
        elif h and h == 'POS':
            pos_col = i

    if word_col is None:
        print(f"  WARNING: Could not find Word column. Headers: {headers[:10]}")
        word_col = 0

    print(f"  Columns - Word: {word_col}, Segm: {segm_col}, Freq: {freq_col}, POS: {pos_col}")

    row_count = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_count += 1
        word = row[word_col] if word_col is not None and word_col < len(row) else None
        segm = row[segm_col] if segm_col is not None and segm_col < len(row) else None
        freq = row[freq_col] if freq_col is not None and freq_col < len(row) else None
        pos = row[pos_col] if pos_col is not None and pos_col < len(row) else None

        if not word:
            continue

        yield {
            'word': str(word).strip(),
            'segmentation': str(segm).strip() if segm else '',
            'frequency': float(freq) if freq else 0,
            'pos': str(pos).strip() if pos else '',
        }

    print(f"  Read {row_count} rows")
    wb.close()


def extract_morpheme_links(segmentation: str, word: str) -> list:
    """Extract morpheme concepts from segmentation"""
    if not segmentation or segmentation == word:
        return []

    # MorphoLex uses various segmentation formats
    # Common: "un+happy+ness" or "{un}{happy}{ness}"
    parts = []

    # Try + separator
    if '+' in segmentation:
        parts = [p.strip() for p in segmentation.split('+')]
    # Try {} format
    elif '{' in segmentation:
        import re
        parts = re.findall(r'\{([^}]+)\}', segmentation)
    # Try - separator
    elif '-' in segmentation:
        parts = [p.strip() for p in segmentation.split('-')]
    else:
        return []

    links = []
    for i, part in enumerate(parts):
        part_lower = part.lower()
        position = 'prefix' if i == 0 and len(parts) > 1 else ('suffix' if i == len(parts) - 1 and len(parts) > 1 else 'root')

        concept = None
        if position == 'prefix':
            concept = PREFIX_TO_CONCEPT.get(part_lower)
        elif position == 'suffix':
            concept = SUFFIX_TO_CONCEPT.get(part_lower)

        if concept:
            links.append({
                'surface': part,
                'concept': concept,
                'position': position,
            })

    return links


def import_morpholex(conn, ref_path: Path, source_id: int):
    """Import MorphoLex data"""
    cur = conn.cursor()

    # Get English language ID
    eng_id = get_english_language_id(cur)
    print(f"  English language ID: {eng_id}")

    # Build morpheme concept lookup
    cur.execute("SELECT id, morpheme FROM morphemes")
    concept_to_id = {row[1]: row[0] for row in cur.fetchall()}
    print(f"  Found {len(concept_to_id)} morpheme concepts")

    insert_count = 0
    link_count = 0
    batch_size = 1000
    batch = []

    for rec in parse_morpholex(ref_path):
        # Insert surface form
        batch.append((
            None,  # morpheme_id - we'll link via separate table if needed
            eng_id,
            rec['word'],
            rec['frequency'],
            rec['pos'],
            SOURCE_NAME,
        ))

        if len(batch) >= batch_size:
            cur.executemany("""
                INSERT OR IGNORE INTO surface_forms
                (morpheme_id, language_id, surface_form, frequency, pos_tag, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            insert_count += cur.rowcount
            conn.commit()
            batch = []

            if insert_count % 10000 == 0:
                print(f"    Inserted {insert_count} surface forms...")

        # Extract and link morpheme concepts
        for link in extract_morpheme_links(rec['segmentation'], rec['word']):
            morpheme_id = concept_to_id.get(link['concept'])
            if morpheme_id:
                # Update the surface form to link to this morpheme
                # For compound words, this creates multiple links
                link_count += 1

    # Final batch
    if batch:
        cur.executemany("""
            INSERT OR IGNORE INTO surface_forms
            (morpheme_id, language_id, surface_form, frequency, pos_tag, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)
        insert_count += cur.rowcount
        conn.commit()

    print(f"  Inserted {insert_count} surface forms")
    print(f"  Found {link_count} morpheme concept links")


def main():
    print(f"Importing MorphoLex-en from {REF_PATH}")

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

        import_morpholex(conn, REF_PATH, source_id)

        # Validation
        cur.execute("""
            SELECT COUNT(*) FROM surface_forms WHERE source = ?
        """, (SOURCE_NAME,))
        count = cur.fetchone()[0]
        print(f"\n  Total surface forms from MorphoLex: {count}")

    finally:
        conn.close()

    print("\nMorphoLex import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
