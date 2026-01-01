#!/usr/bin/env python3
"""
Import Kaikki.org Wiktionary data into the language database.

Streaming pipeline:
1. Read JSONL entries one at a time (memory efficient)
2. Extract senses → group into synsets by meaning clusters
3. Map inflected forms → surface_forms table
4. Extract translations → translations table
5. Assign token IDs based on synset structure

Usage:
    python import_kaikki.py                           # Import English data
    python import_kaikki.py --db custom.db            # Specify database
    python import_kaikki.py --input reference/kaikki/german.jsonl
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "reference" / "kaikki" / "english.jsonl"
DEFAULT_DB = SCRIPT_DIR / "db" / "language.db"
SCHEMA_FILE = SCRIPT_DIR / "db" / "SCHEMA-language-db.sql"

# Token ID constants
CONCEPT_BASE = 3_000_000  # Concepts start here
SYNSET_SLOTS = 128        # Slots per synset for concepts


class SynsetBuilder:
    """Builds synsets from Kaikki sense data."""

    def __init__(self):
        self.synset_counter = 0
        self.synset_map = {}  # gloss_hash -> synset_id
        self.synsets = []     # List of synset dicts
        self.concepts = []    # List of concept dicts
        self.concept_offset = defaultdict(int)  # synset_id -> next offset

    def normalize_gloss(self, gloss: str) -> str:
        """Normalize gloss for synset matching."""
        # Remove parenthetical notes, lowercase, strip
        gloss = re.sub(r'\([^)]*\)', '', gloss)
        gloss = gloss.lower().strip()
        gloss = re.sub(r'\s+', ' ', gloss)
        return gloss

    def get_or_create_synset(self, gloss: str, pos: str = None, domain: str = None, lemma: str = None) -> int:
        """Get existing synset or create new one."""
        norm_gloss = self.normalize_gloss(gloss)

        # For proper nouns (names), include lemma in hash - "Russell" and "Scott"
        # shouldn't share a synset even if glosses are similar
        # Detect by POS or by gloss patterns (given name, surname, place name, etc.)
        is_proper_noun = (
            pos in ('name', 'proper noun', 'prop', 'noun') and lemma and (
                lemma[0].isupper() and (
                    'given name' in norm_gloss or
                    'surname' in norm_gloss or
                    'family name' in norm_gloss or
                    'male name' in norm_gloss or
                    'female name' in norm_gloss or
                    'place name' in norm_gloss or
                    'city in' in norm_gloss or
                    'country in' in norm_gloss or
                    'region in' in norm_gloss
                )
            )
        )

        # Chemical compounds also need their own synsets - thousands share "a chemical compound"
        is_chemistry = (
            'chemical compound' in norm_gloss or
            'organic compound' in norm_gloss or
            'inorganic compound' in norm_gloss or
            'chemical element' in norm_gloss or
            'pharmaceutical' in norm_gloss or
            'drug used' in norm_gloss
        )

        if is_proper_noun or is_chemistry:
            hash_input = f"{lemma}:{norm_gloss}"
        else:
            hash_input = norm_gloss

        gloss_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

        if gloss_hash in self.synset_map:
            return self.synset_map[gloss_hash]

        synset_id = self.synset_counter
        self.synset_counter += 1
        self.synset_map[gloss_hash] = synset_id

        self.synsets.append({
            "synset_id": synset_id,
            "gloss": gloss,
            "pos": pos,
            "domain": domain,
            "source": "kaikki",
            "source_id": gloss_hash
        })

        return synset_id

    def add_concept(self, synset_id: int, lemma: str, gloss: str = None, lang: str = "en") -> int:
        """Add a concept to a synset."""
        offset = self.concept_offset[synset_id]
        if offset >= SYNSET_SLOTS:
            # Synset full, this shouldn't happen often
            print(f"  Warning: synset {synset_id} full, skipping concept '{lemma}'")
            return -1

        concept_id = len(self.concepts)
        self.concepts.append({
            "concept_id": concept_id,
            "synset_id": synset_id,
            "concept_offset": offset,
            "lemma": lemma,
            "gloss": gloss,
            "lang": lang,
            "frequency": 0
        })

        self.concept_offset[synset_id] = offset + 1
        return concept_id


def parse_kaikki_entry(entry: dict) -> dict:
    """Parse a single Kaikki.org JSONL entry."""
    result = {
        "word": entry.get("word", ""),
        "pos": entry.get("pos", ""),
        "lang": entry.get("lang", ""),
        "lang_code": entry.get("lang_code", "en"),
        "senses": [],
        "forms": [],
        "translations": [],
        "etymology": entry.get("etymology_text", "")
    }

    # Extract senses
    for sense in entry.get("senses", []):
        glosses = sense.get("glosses", [])
        if glosses:
            result["senses"].append({
                "gloss": glosses[0],  # Primary gloss
                "tags": sense.get("tags", []),
                "categories": sense.get("categories", [])
            })

    # Extract forms (inflections)
    for form in entry.get("forms", []):
        form_text = form.get("form", "")
        if form_text and form_text != result["word"]:
            result["forms"].append({
                "form": form_text,
                "tags": form.get("tags", [])
            })

    # Extract translations
    for trans in entry.get("translations", []):
        trans_text = trans.get("word", "")
        trans_lang = trans.get("lang", "")
        if trans_text and trans_lang:
            result["translations"].append({
                "word": trans_text,
                "lang": trans_lang,
                "lang_code": trans.get("code", "")
            })

    return result


def stream_jsonl(filepath: Path) -> Iterator[dict]:
    """Stream JSONL file, yielding parsed entries."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error line {line_num}: {e}")
                continue


def tags_to_features(tags: list) -> str:
    """Convert Kaikki tags to UniMorph-style feature string."""
    # Map common Kaikki tags to feature codes
    tag_map = {
        "singular": "SG", "plural": "PL",
        "first-person": "1", "second-person": "2", "third-person": "3",
        "present": "PRS", "past": "PST", "future": "FUT",
        "indicative": "IND", "subjunctive": "SBJV", "imperative": "IMP",
        "active": "ACT", "passive": "PASS",
        "masculine": "MASC", "feminine": "FEM", "neuter": "NEUT",
        "nominative": "NOM", "accusative": "ACC", "dative": "DAT", "genitive": "GEN",
        "infinitive": "INF", "participle": "PTCP", "gerund": "GER",
        "comparative": "CMPR", "superlative": "SPRL"
    }

    features = []
    for tag in tags:
        if tag in tag_map:
            features.append(tag_map[tag])

    return ";".join(features) if features else None


def import_to_db(
    input_file: Path,
    db_path: Path,
    schema_file: Path,
    limit: Optional[int] = None,
    batch_size: int = 10000
):
    """Import Kaikki data into SQLite database."""

    print(f"Importing Kaikki.org data")
    print(f"  Input:  {input_file}")
    print(f"  Output: {db_path}")
    print()

    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # Faster bulk import
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Apply schema
    print("Applying schema...")
    with open(schema_file, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()

    # Build synsets and concepts
    builder = SynsetBuilder()
    surface_forms = []
    translations = []

    print("Processing entries...")
    start_time = time.time()
    entry_count = 0
    sense_count = 0
    form_count = 0
    proper_noun_count = 0  # Skipped - belong in Identity DB

    for raw_entry in stream_jsonl(input_file):
        if limit and entry_count >= limit:
            break

        entry = parse_kaikki_entry(raw_entry)
        word = entry["word"]
        pos = entry["pos"]
        lang_code = entry["lang_code"]

        if not word or not entry["senses"]:
            continue

        # Skip proper nouns - they belong in Identity DB, not Language DB
        # Proper nouns are entity references, not composable concepts
        if pos in ('name', 'proper noun', 'prop'):
            proper_noun_count += 1
            continue

        entry_count += 1

        # Process each sense as potential concept
        for sense in entry["senses"]:
            gloss = sense["gloss"]
            if not gloss:
                continue

            # Get or create synset from gloss (pass lemma for proper nouns)
            synset_id = builder.get_or_create_synset(gloss, pos=pos, lemma=word)

            # Add concept
            concept_id = builder.add_concept(synset_id, word, gloss, lang_code)
            if concept_id < 0:
                continue

            sense_count += 1

            # Add lemma as surface form
            surface_forms.append({
                "surface_form": word,
                "concept_id": concept_id,
                "lang": lang_code,
                "form_type": "lemma",
                "pos_features": None,
                "frequency": 0
            })

            # Add inflected forms
            for form_data in entry["forms"]:
                form_text = form_data["form"]
                features = tags_to_features(form_data["tags"])

                surface_forms.append({
                    "surface_form": form_text,
                    "concept_id": concept_id,
                    "lang": lang_code,
                    "form_type": "inflected",
                    "pos_features": features,
                    "frequency": 0
                })
                form_count += 1

            # Add translations
            for trans in entry["translations"]:
                translations.append({
                    "concept_id": concept_id,
                    "target_lang": trans["lang_code"] or trans["lang"][:3].lower(),
                    "translation": trans["word"],
                    "confidence": 1.0,
                    "source": "kaikki"
                })

        # Progress update
        if entry_count % 10000 == 0:
            elapsed = time.time() - start_time
            rate = entry_count / elapsed
            print(f"  Processed {entry_count:,} entries ({rate:.0f}/sec)...")

    print(f"\nBuilding database tables...")
    print(f"  Synsets:       {len(builder.synsets):,}")
    print(f"  Concepts:      {len(builder.concepts):,}")
    print(f"  Surface forms: {len(surface_forms):,}")
    print(f"  Translations:  {len(translations):,}")

    # Insert synsets
    print("\nInserting synsets...", end="", flush=True)
    conn.executemany(
        """INSERT OR IGNORE INTO synsets
           (synset_id, gloss, pos, domain, source, source_id)
           VALUES (:synset_id, :gloss, :pos, :domain, :source, :source_id)""",
        builder.synsets
    )
    conn.commit()
    print(" done")

    # Insert concepts
    print("Inserting concepts...", end="", flush=True)
    conn.executemany(
        """INSERT OR IGNORE INTO concepts
           (concept_id, synset_id, concept_offset, lemma, gloss, lang, frequency)
           VALUES (:concept_id, :synset_id, :concept_offset, :lemma, :gloss, :lang, :frequency)""",
        builder.concepts
    )
    conn.commit()
    print(" done")

    # Insert surface forms in batches
    print("Inserting surface forms...", end="", flush=True)
    for i in range(0, len(surface_forms), batch_size):
        batch = surface_forms[i:i + batch_size]
        conn.executemany(
            """INSERT OR IGNORE INTO surface_forms
               (surface_form, concept_id, lang, form_type, pos_features, frequency)
               VALUES (:surface_form, :concept_id, :lang, :form_type, :pos_features, :frequency)""",
            batch
        )
        conn.commit()
        print(".", end="", flush=True)
    print(" done")

    # Insert translations in batches
    print("Inserting translations...", end="", flush=True)
    for i in range(0, len(translations), batch_size):
        batch = translations[i:i + batch_size]
        conn.executemany(
            """INSERT OR IGNORE INTO translations
               (concept_id, target_lang, translation, confidence, source)
               VALUES (:concept_id, :target_lang, :translation, :confidence, :source)""",
            batch
        )
        conn.commit()
        print(".", end="", flush=True)
    print(" done")

    # Record import metadata
    file_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()[:16]
    conn.execute(
        """INSERT INTO import_metadata (source, file_hash, record_count, notes)
           VALUES (?, ?, ?, ?)""",
        ("kaikki", file_hash, entry_count, f"synsets={len(builder.synsets)}, concepts={len(builder.concepts)}")
    )
    conn.commit()

    # Re-enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Time:          {elapsed:.1f} seconds")
    print(f"  Entries:       {entry_count:,}")
    print(f"  Proper nouns:  {proper_noun_count:,} (skipped → Identity DB)")
    print(f"  Synsets:       {len(builder.synsets):,}")
    print(f"  Concepts:      {len(builder.concepts):,}")
    print(f"  Surface forms: {len(surface_forms):,}")
    print(f"  Translations:  {len(translations):,}")
    print(f"  Database:      {db_path}")
    print(f"  Size:          {db_path.stat().st_size / (1024*1024):.1f} MB")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Import Kaikki.org Wiktionary data into language database"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        default=DEFAULT_DB,
        help=f"Output database (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--schema", "-s",
        type=Path,
        default=SCHEMA_FILE,
        help=f"Schema SQL file (default: {SCHEMA_FILE})"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of entries to import (for testing)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="Batch size for database inserts (default: 10000)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print(f"\nRun download_kaikki.py first to download the data.")
        sys.exit(1)

    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)

    import_to_db(
        args.input,
        args.db,
        args.schema,
        limit=args.limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
