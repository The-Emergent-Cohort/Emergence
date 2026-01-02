#!/usr/bin/env python3
"""
Import Princeton WordNet into English language database.

WordNet provides:
- Synsets (synonym sets representing concepts)
- Glosses (definitions)
- Semantic relations (hypernymy, hyponymy, etc.)

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py to have been run first
"""

import re
import sqlite3
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
LANG_DIR = DB_DIR / "lang"

# POS mapping
POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adj",
    "r": "adv",
    "s": "adj",  # satellite adjective
}

# Language DB schema (in case we need to create it)
LANGUAGE_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,
    abstraction INTEGER NOT NULL DEFAULT 99,
    fingerprint INTEGER,
    source TEXT
);

CREATE TABLE IF NOT EXISTS surface_forms (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    form TEXT NOT NULL,
    form_type TEXT,
    features TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    component_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    relation TEXT DEFAULT 'part',
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS dialect_status (
    concept_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'shared',
    divergent_dialects TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,
    target_lemma TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS token_index (
    idx INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    description TEXT,
    location TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synset_relations (
    id INTEGER PRIMARY KEY,
    source_synset TEXT NOT NULL,
    target_synset TEXT NOT NULL,
    relation_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_surface_form ON surface_forms(form);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_token ON concepts(token_id);
CREATE INDEX IF NOT EXISTS idx_synset_source ON synset_relations(source_synset);
CREATE INDEX IF NOT EXISTS idx_synset_target ON synset_relations(target_synset);
"""


def parse_data_file(filepath: Path, pos: str) -> list:
    """Parse a WordNet data file (data.noun, data.verb, etc.)."""
    synsets = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("  "):  # Copyright header
                continue

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            try:
                offset = parts[0]
                lex_filenum = parts[1]
                ss_type = parts[2]
                word_count = int(parts[3], 16)

                # Parse words
                words = []
                idx = 4
                for _ in range(word_count):
                    if idx >= len(parts):
                        break
                    word = parts[idx].replace("_", " ")
                    idx += 2  # skip lex_id

                    words.append(word)

                # Find gloss (after |)
                gloss = ""
                full_line = line.strip()
                if "|" in full_line:
                    gloss = full_line.split("|", 1)[1].strip()

                synset_id = f"{offset}-{pos}"

                synsets.append({
                    "synset_id": synset_id,
                    "offset": offset,
                    "pos": pos,
                    "words": words,
                    "gloss": gloss,
                })

            except (ValueError, IndexError):
                continue

    return synsets


def generate_token_id(pos: str, offset: str) -> str:
    """Generate genomic token ID for a WordNet synset."""
    # Format: A.D.C.FF.SS.LLL.DD.FP.COL
    # A=2 (lexical), D varies by POS, C=1, English=1.8.127.0
    domain = {"noun": 5, "verb": 1, "adj": 9, "adv": 2}.get(pos, 5)
    fingerprint = int(offset) % 1000000
    return f"2.{domain}.1.1.8.127.0.{fingerprint}.0"


def main():
    print("=" * 60)
    print("Importing Princeton WordNet")
    print("=" * 60)

    # Find WordNet data
    wordnet_dir = REF_DIR / "wordnet"
    if not wordnet_dir.exists():
        # Try common alternative locations
        for alt in ["wn3.1", "wn31", "WordNet-3.0", "WordNet-3.1", "dict"]:
            alt_dir = REF_DIR / alt
            if alt_dir.exists():
                wordnet_dir = alt_dir
                break

    if not wordnet_dir.exists():
        print(f"ERROR: WordNet directory not found in {REF_DIR}")
        print("Run unpack_tarballs.py first or download WordNet.")
        return 1

    # Find data files
    data_files = {}
    for pos_code, pos_name in [("noun", "n"), ("verb", "v"), ("adj", "a"), ("adv", "r")]:
        # Try different naming patterns
        for pattern in [f"data.{pos_code}", f"data.{pos_name}"]:
            matches = list(wordnet_dir.rglob(pattern))
            if matches:
                data_files[pos_name] = matches[0]
                break

    if not data_files:
        print(f"ERROR: No WordNet data files found in {wordnet_dir}")
        print("Expected files like: data.noun, data.verb, etc.")
        return 1

    print(f"\nFound WordNet data in: {wordnet_dir}")
    for pos, path in data_files.items():
        print(f"  {POS_MAP.get(pos, pos)}: {path.name}")

    # Ensure lang directory exists
    LANG_DIR.mkdir(parents=True, exist_ok=True)

    # Create/open English database
    eng_db = LANG_DIR / "eng.db"
    print(f"\nDatabase: {eng_db}")

    conn = sqlite3.connect(eng_db)
    conn.executescript(LANGUAGE_DB_SCHEMA)

    cursor = conn.cursor()

    # Clear existing WordNet data
    cursor.execute("DELETE FROM concepts WHERE source = 'wordnet'")
    cursor.execute("DELETE FROM surface_forms WHERE concept_id NOT IN (SELECT concept_id FROM concepts)")

    total_synsets = 0
    total_words = 0

    for pos, data_path in data_files.items():
        pos_name = POS_MAP.get(pos, pos)
        print(f"\nProcessing {pos_name}...")

        synsets = parse_data_file(data_path, pos)
        print(f"  Found {len(synsets)} synsets")

        batch_concepts = []
        batch_forms = []

        for synset in synsets:
            if not synset["words"]:
                continue

            # Primary lemma is first word
            lemma = synset["words"][0].lower()
            token_id = generate_token_id(pos_name, synset["offset"])

            batch_concepts.append({
                "token_id": token_id,
                "lemma": lemma,
                "gloss": synset["gloss"],
                "pos": pos_name,
                "abstraction": 2,  # Lexical level
                "fingerprint": int(synset["offset"]) % 1000000,
                "source": "wordnet",
            })

            # Track for surface forms
            for word in synset["words"]:
                batch_forms.append({
                    "token_id": token_id,
                    "form": word.lower(),
                })

        # Insert concepts
        for concept in batch_concepts:
            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (token_id, lemma, gloss, pos, abstraction, fingerprint, source)
                VALUES (:token_id, :lemma, :gloss, :pos, :abstraction, :fingerprint, :source)
            """, concept)

        # Get concept_ids for surface forms
        cursor.execute("SELECT concept_id, token_id FROM concepts WHERE source = 'wordnet'")
        token_to_id = {row[1]: row[0] for row in cursor.fetchall()}

        # Insert surface forms
        for form_data in batch_forms:
            concept_id = token_to_id.get(form_data["token_id"])
            if concept_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO surface_forms (concept_id, form, form_type)
                    VALUES (?, ?, 'lemma')
                """, (concept_id, form_data["form"]))
                total_words += 1

        total_synsets += len(batch_concepts)
        conn.commit()

    print("\n" + "=" * 60)
    print("WordNet import complete!")
    print("=" * 60)
    print(f"\nSynsets imported: {total_synsets}")
    print(f"Surface forms: {total_words}")
    print(f"\nDatabase: {eng_db}")

    # Show sample
    cursor.execute("""
        SELECT lemma, pos, substr(gloss, 1, 60) as gloss
        FROM concepts
        WHERE source = 'wordnet'
        ORDER BY RANDOM()
        LIMIT 5
    """)
    print("\nSample entries:")
    for row in cursor.fetchall():
        gloss = row[2] + "..." if row[2] and len(row[2]) >= 60 else row[2]
        print(f"  {row[0]} ({row[1]}): {gloss}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
