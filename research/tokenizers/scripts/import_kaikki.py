#!/usr/bin/env python3
"""
Import Kaikki.org Wiktionary data into language databases.

Kaikki provides comprehensive dictionary data with:
- Word entries with definitions (glosses)
- Inflected forms
- Translations
- Etymology information

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py to have been run first
"""

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterator

# Hardcoded paths - data lives at /usr/share/databases
BASE_DIR = Path("/usr/share/databases")
SCRIPT_DIR = BASE_DIR / "scripts"
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
LANG_DIR = DB_DIR / "lang"
LANG_REGISTRY_DB = DB_DIR / "language_registry.db"

# Language DB schema
LANGUAGE_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,
    abstraction INTEGER NOT NULL DEFAULT 99,
    fingerprint INTEGER,
    capitalize_output INTEGER DEFAULT 0,
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

CREATE INDEX IF NOT EXISTS idx_surface_form ON surface_forms(form);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_token ON concepts(token_id);
CREATE INDEX IF NOT EXISTS idx_concept_fp ON concepts(fingerprint);
CREATE INDEX IF NOT EXISTS idx_trans_target ON translations(target_lang, target_lemma);
"""

# POS to domain mapping
POS_DOMAIN = {
    "noun": 5,      # abstract/category
    "verb": 1,      # physical/motion
    "adj": 9,       # evaluative
    "adv": 2,       # temporal
    "prep": 1,      # physical/location
    "conj": 5,      # abstract/logic
    "det": 5,       # abstract
    "pron": 4,      # social
    "intj": 3,      # mental/emotion
    "num": 5,       # abstract/quantity
    "name": 4,      # social (proper nouns)
}

# Domain jargon genomic coordinates (non-human symbol systems)
# Maps Kaikki category tags to full genomic coordinates: (family, subfamily, lang, dialect)
# Family 51 = Math, 52 = Chemistry, 53 = Physics, 54 = Computing, 55 = Music
#
# Structure: category -> (family, subfamily, lang, dialect)
# Subfamily = major branch, Lang = specialization, Dialect = sub-specialization
JARGON_GENOMIC = {
    # Math (family 51)
    "mathematics":    (51, 0, 0, 0),   # General mathematics
    "arithmetic":     (51, 1, 0, 0),   # Basic number operations
    "algebra":        (51, 2, 0, 0),   # Algebra branch
    "linear algebra": (51, 2, 1, 0),   # ... linear specialization
    "calculus":       (51, 3, 0, 0),   # Calculus branch
    "geometry":       (51, 4, 0, 0),   # Geometry branch
    "trigonometry":   (51, 4, 1, 0),   # ... trig specialization
    "topology":       (51, 4, 2, 0),   # ... topology specialization
    "number theory":  (51, 5, 0, 0),   # Number theory branch
    "set theory":     (51, 6, 0, 0),   # Set theory branch
    "logic":          (51, 6, 1, 0),   # ... logic specialization
    "combinatorics":  (51, 7, 0, 0),   # Combinatorics branch
    "statistics":     (51, 8, 0, 0),   # Statistics branch
    "probability":    (51, 8, 1, 0),   # ... probability specialization

    # Chemistry (family 52)
    "chemistry":          (52, 0, 0, 0),   # General chemistry
    "organic chemistry":  (52, 1, 0, 0),   # Organic branch
    "inorganic chemistry": (52, 2, 0, 0),  # Inorganic branch
    "biochemistry":       (52, 3, 0, 0),   # Biochemistry branch
    "analytical chemistry": (52, 4, 0, 0), # Analytical branch
    "electrochemistry":   (52, 5, 0, 0),   # Electrochemistry branch
    "chemical element":   (52, 0, 1, 0),   # Elements (general subcategory)
    "chemical compound":  (52, 0, 2, 0),   # Compounds (general subcategory)
    "pharmacology":       (52, 3, 1, 0),   # Pharmacology (biochem adjacent)

    # Physics (family 53)
    "physics":            (53, 0, 0, 0),   # General physics
    "mechanics":          (53, 1, 0, 0),   # Classical mechanics
    "thermodynamics":     (53, 2, 0, 0),   # Thermodynamics
    "electromagnetism":   (53, 3, 0, 0),   # E&M
    "optics":             (53, 3, 1, 0),   # ... optics specialization
    "acoustics":          (53, 4, 0, 0),   # Acoustics
    "quantum mechanics":  (53, 5, 0, 0),   # Quantum branch
    "quantum physics":    (53, 5, 0, 0),   # Alias
    "nuclear physics":    (53, 5, 1, 0),   # ... nuclear specialization
    "particle physics":   (53, 5, 2, 0),   # ... particle specialization
    "relativity":         (53, 6, 0, 0),   # Relativity
    "astrophysics":       (53, 7, 0, 0),   # Astrophysics

    # Computing (family 54)
    "computing":          (54, 0, 0, 0),   # General computing
    "computer science":   (54, 0, 0, 0),   # Alias
    "programming":        (54, 1, 0, 0),   # Programming branch
    "software":           (54, 1, 1, 0),   # ... software specialization
    "computer hardware":  (54, 2, 0, 0),   # Hardware branch
    "networking":         (54, 3, 0, 0),   # Networking branch
    "internet":           (54, 3, 1, 0),   # ... internet specialization
    "databases":          (54, 4, 0, 0),   # Databases branch
    "cryptography":       (54, 5, 0, 0),   # Cryptography branch

    # Music (family 55)
    "music":              (55, 0, 0, 0),   # General music
    "music theory":       (55, 1, 0, 0),   # Theory branch
    "musical instruments": (55, 2, 0, 0),  # Instruments branch
}

# Capitalized common nouns - stay in main DB with capitalize_output=1
CAPITALIZE_PATTERNS = [
    # Days and months
    r'^(January|February|March|April|May|June|July|August|September|October|November|December)$',
    r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',
    # Languages (when used as noun, not adjective)
    r'^(English|French|German|Spanish|Italian|Portuguese|Chinese|Japanese|Korean|Arabic|Russian|Hindi)$',
]

import re

def detect_jargon_genomic(entry: dict) -> tuple | None:
    """
    Detect if entry is domain jargon using Kaikki's category tags.

    Returns (family, subfamily, lang, dialect) genomic coordinates or None.
    """
    # Check categories in senses
    for sense in entry.get("senses", []):
        if isinstance(sense, dict):
            # Check raw_tags, tags, and topics for category info
            for tag_field in ["raw_tags", "tags", "topics"]:
                tags = sense.get(tag_field, [])
                for tag in tags:
                    tag_lower = str(tag).lower()
                    if tag_lower in JARGON_GENOMIC:
                        return JARGON_GENOMIC[tag_lower]

            # Check categories field directly
            categories = sense.get("categories", [])
            for cat in categories:
                if isinstance(cat, dict):
                    cat_name = cat.get("name", "").lower()
                else:
                    cat_name = str(cat).lower()
                if cat_name in JARGON_GENOMIC:
                    return JARGON_GENOMIC[cat_name]

    return None


def is_capitalized_common_noun(word: str) -> bool:
    """Check if word is a capitalized common noun (January, Monday, etc.)."""
    for pattern in CAPITALIZE_PATTERNS:
        if re.match(pattern, word):
            return True
    return False


def get_lang_genomic(iso_code: str) -> str:
    """Get genomic code for a language from registry."""
    if not LANG_REGISTRY_DB.exists():
        # Fallback for common languages
        fallbacks = {
            "en": "1.8.127.0", "eng": "1.8.127.0",
            "de": "1.8.200.0", "deu": "1.8.200.0",
            "fr": "1.12.100.0", "fra": "1.12.100.0",
            "es": "1.12.200.0", "spa": "1.12.200.0",
            "ja": "9.0.1.0", "jpn": "9.0.1.0",
            "zh": "2.0.1.0", "cmn": "2.0.1.0",
        }
        return fallbacks.get(iso_code, "0.0.0.0")

    conn = sqlite3.connect(LANG_REGISTRY_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT genomic_code FROM language_codes WHERE iso639_3 = ? OR iso639_1 = ?",
        (iso_code, iso_code)
    )
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else "0.0.0.0"


def generate_fingerprint(lemma: str, gloss: str) -> int:
    """Generate fingerprint from lemma and gloss."""
    combined = f"{lemma.lower()}:{gloss[:100] if gloss else ''}"
    hash_bytes = hashlib.md5(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], 'big') % 1000000


def generate_token_id(lang_genomic: str, pos: str, fingerprint: int) -> str:
    """Generate genomic token ID."""
    domain = POS_DOMAIN.get(pos, 5)
    return f"2.{domain}.1.{lang_genomic}.{fingerprint}.0"


def stream_jsonl(filepath: Path) -> Iterator[dict]:
    """Stream JSONL file, yielding parsed entries."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tags_to_features(tags: list) -> str:
    """Convert Kaikki tags to feature string."""
    tag_map = {
        "singular": "SG", "plural": "PL",
        "first-person": "1", "second-person": "2", "third-person": "3",
        "present": "PRS", "past": "PST", "future": "FUT",
        "indicative": "IND", "subjunctive": "SBJV", "imperative": "IMP",
        "masculine": "MASC", "feminine": "FEM", "neuter": "NEUT",
        "nominative": "NOM", "accusative": "ACC", "dative": "DAT", "genitive": "GEN",
        "infinitive": "INF", "participle": "PTCP",
        "comparative": "CMPR", "superlative": "SPRL"
    }

    features = [tag_map[t] for t in tags if t in tag_map]
    return ";".join(features) if features else None


def get_or_create_jargon_db(family: int) -> sqlite3.Connection:
    """Get or create a jargon domain database."""
    # Map family to name
    family_names = {51: "math", 52: "chemistry", 53: "physics", 54: "computing", 55: "music"}
    name = family_names.get(family, f"domain_{family}")

    jargon_dir = DB_DIR / "jargon"
    jargon_dir.mkdir(parents=True, exist_ok=True)
    jargon_db = jargon_dir / f"{name}.db"

    conn = sqlite3.connect(jargon_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(LANGUAGE_DB_SCHEMA)

    return conn


def import_kaikki(input_file: Path, lang_code: str, limit: int = None):
    """Import Kaikki data for a single language."""
    print(f"\nImporting {lang_code} from {input_file.name}...")

    # Get genomic code
    lang_genomic = get_lang_genomic(lang_code)
    print(f"  Language genomic: {lang_genomic}")

    # Create/open language database
    LANG_DIR.mkdir(parents=True, exist_ok=True)
    iso3 = lang_code if len(lang_code) == 3 else {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "ja": "jpn", "zh": "cmn"}.get(lang_code, lang_code)
    lang_db = LANG_DIR / f"{iso3}.db"

    conn = sqlite3.connect(lang_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(LANGUAGE_DB_SCHEMA)

    cursor = conn.cursor()

    # Track existing fingerprints to avoid duplicates
    cursor.execute("SELECT fingerprint FROM concepts WHERE source = 'kaikki'")
    existing_fps = {row[0] for row in cursor.fetchall()}

    # Track jargon DB connections (lazy open)
    jargon_conns = {}  # family -> (conn, cursor, existing_fps)

    start_time = time.time()
    entry_count = 0
    concept_count = 0
    form_count = 0
    trans_count = 0
    skipped_proper = 0
    jargon_count = 0

    batch_concepts = []
    batch_forms = []
    batch_trans = []

    # Jargon batches: family -> {"concepts": [], "forms": [], "trans": []}
    jargon_batches = {}

    for entry in stream_jsonl(input_file):
        if limit and entry_count >= limit:
            break

        word = entry.get("word", "")
        pos = entry.get("pos", "")
        senses = entry.get("senses", [])

        if not word or not senses:
            continue

        # Skip proper nouns (belong in identity DB)
        if pos in ("name", "proper noun", "prop"):
            skipped_proper += 1
            continue

        entry_count += 1

        # Check for domain jargon - route to separate DB
        jargon_genomic = detect_jargon_genomic(entry)

        # Check for capitalized common nouns
        capitalize = 1 if is_capitalized_common_noun(word) else 0

        # Process each sense
        for sense in senses:
            glosses = sense.get("glosses", [])
            if not glosses:
                continue

            gloss = glosses[0]
            fingerprint = generate_fingerprint(word, gloss)

            if jargon_genomic:
                # Route to jargon DB
                family = jargon_genomic[0]
                if family not in jargon_conns:
                    jc = get_or_create_jargon_db(family)
                    jcursor = jc.cursor()
                    jcursor.execute("SELECT fingerprint FROM concepts WHERE source = 'kaikki'")
                    jfps = {row[0] for row in jcursor.fetchall()}
                    jargon_conns[family] = (jc, jcursor, jfps)
                    jargon_batches[family] = {"concepts": [], "forms": [], "trans": []}

                _, _, jfps = jargon_conns[family]
                if fingerprint in jfps:
                    continue

                # Build genomic token ID using jargon coordinates
                fam, sub, lng, dial = jargon_genomic
                domain = POS_DOMAIN.get(pos, 5)
                token_id = f"2.{domain}.1.{fam}.{sub}.{lng}.{dial}.{fingerprint}.0"

                jargon_batches[family]["concepts"].append({
                    "token_id": token_id,
                    "lemma": word,
                    "gloss": gloss[:500] if gloss else None,
                    "pos": pos,
                    "abstraction": 2,
                    "fingerprint": fingerprint,
                    "capitalize_output": 0,
                    "source": "kaikki",
                })
                jfps.add(fingerprint)
                jargon_count += 1
                continue

            # Regular language DB
            if fingerprint in existing_fps:
                continue

            token_id = generate_token_id(lang_genomic, pos, fingerprint)

            batch_concepts.append({
                "token_id": token_id,
                "lemma": word,
                "gloss": gloss[:500] if gloss else None,  # Truncate long glosses
                "pos": pos,
                "abstraction": 2,
                "fingerprint": fingerprint,
                "capitalize_output": capitalize,
                "source": "kaikki",
            })
            existing_fps.add(fingerprint)
            concept_count += 1

            # We'll link forms and translations after inserting concepts
            # Store temporarily with fingerprint as key
            for form_data in entry.get("forms", []):
                form_text = form_data.get("form", "")
                if form_text and form_text != word:
                    batch_forms.append({
                        "fingerprint": fingerprint,
                        "form": form_text,
                        "features": tags_to_features(form_data.get("tags", [])),
                    })
                    form_count += 1

            for trans in entry.get("translations", []):
                trans_word = trans.get("word", "")
                trans_lang = trans.get("code", "") or trans.get("lang", "")[:3].lower()
                if trans_word and trans_lang:
                    batch_trans.append({
                        "fingerprint": fingerprint,
                        "target_lang": trans_lang,
                        "target_lemma": trans_word,
                    })
                    trans_count += 1

        # Batch insert periodically
        if len(batch_concepts) >= 5000:
            _flush_batch(cursor, batch_concepts, batch_forms, batch_trans)
            batch_concepts = []
            batch_forms = []
            batch_trans = []
            conn.commit()

            # Flush jargon batches too
            for family, batches in jargon_batches.items():
                if len(batches["concepts"]) >= 1000:
                    jc, jcursor, _ = jargon_conns[family]
                    _flush_batch(jcursor, batches["concepts"], batches["forms"], batches["trans"])
                    jc.commit()
                    batches["concepts"] = []
                    batches["forms"] = []
                    batches["trans"] = []

            elapsed = time.time() - start_time
            rate = entry_count / elapsed if elapsed > 0 else 0
            print(f"  Processed {entry_count:,} entries, {concept_count:,} concepts, {jargon_count:,} jargon ({rate:.0f}/sec)...")

    # Final flush - main DB
    if batch_concepts:
        _flush_batch(cursor, batch_concepts, batch_forms, batch_trans)
        conn.commit()

    # Final flush - jargon DBs
    for family, batches in jargon_batches.items():
        if batches["concepts"]:
            jc, jcursor, _ = jargon_conns[family]
            _flush_batch(jcursor, batches["concepts"], batches["forms"], batches["trans"])
            jc.commit()

    elapsed = time.time() - start_time

    print(f"\n  Completed {lang_code}:")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Entries: {entry_count:,}")
    print(f"    Concepts: {concept_count:,}")
    print(f"    Jargon terms: {jargon_count:,}")
    print(f"    Forms: {form_count:,}")
    print(f"    Translations: {trans_count:,}")
    print(f"    Skipped (proper nouns): {skipped_proper:,}")
    print(f"    Database: {lang_db}")

    # Report jargon DBs
    for family, (jc, _, _) in jargon_conns.items():
        family_names = {51: "math", 52: "chemistry", 53: "physics", 54: "computing", 55: "music"}
        name = family_names.get(family, f"domain_{family}")
        print(f"    Jargon DB ({name}): db/jargon/{name}.db")
        jc.close()

    conn.close()
    return concept_count


def _flush_batch(cursor, concepts, forms, trans):
    """Insert batched data."""
    # Insert concepts
    for c in concepts:
        cursor.execute("""
            INSERT OR IGNORE INTO concepts
            (token_id, lemma, gloss, pos, abstraction, fingerprint, capitalize_output, source)
            VALUES (:token_id, :lemma, :gloss, :pos, :abstraction, :fingerprint, :capitalize_output, :source)
        """, c)

    # Build fingerprint -> concept_id map
    fps = [c["fingerprint"] for c in concepts]
    if fps:
        placeholders = ",".join("?" * len(fps))
        cursor.execute(f"SELECT concept_id, fingerprint FROM concepts WHERE fingerprint IN ({placeholders})", fps)
        fp_to_id = {row[1]: row[0] for row in cursor.fetchall()}

        # Insert forms
        for f in forms:
            concept_id = fp_to_id.get(f["fingerprint"])
            if concept_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO surface_forms (concept_id, form, form_type, features)
                    VALUES (?, ?, 'inflected', ?)
                """, (concept_id, f["form"], f["features"]))

        # Insert translations
        for t in trans:
            concept_id = fp_to_id.get(t["fingerprint"])
            if concept_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO translations (concept_id, target_lang, target_lemma)
                    VALUES (?, ?, ?)
                """, (concept_id, t["target_lang"], t["target_lemma"]))


def main():
    print("=" * 60)
    print("Importing Kaikki.org Wiktionary Data")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Import Kaikki.org data")
    parser.add_argument("--lang", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--name", "-n", help="Language name for file lookup (e.g., German)")
    parser.add_argument("--input", "-i", type=Path, help="Input JSONL file")
    parser.add_argument("--limit", type=int, help="Limit entries (for testing)")
    args = parser.parse_args()

    # Find input file
    kaikki_dir = REF_DIR / "kaikki"
    if args.input:
        input_file = args.input
    else:
        # Try to find file for language
        patterns = [
            f"kaikki.org-dictionary-{args.lang}*.jsonl",
            f"{args.lang}*.jsonl",
        ]
        # Add language name patterns if provided
        if args.name:
            patterns.insert(0, f"{args.name.lower()}.jsonl")
            patterns.append(f"*{args.name.lower()}*.jsonl")
        input_file = None
        for pattern in patterns:
            matches = list(kaikki_dir.glob(pattern)) if kaikki_dir.exists() else []
            if matches:
                input_file = max(matches, key=lambda p: p.stat().st_size)
                break

    if not input_file or not input_file.exists():
        print(f"ERROR: No Kaikki data found for '{args.lang}'")
        print(f"Searched in: {kaikki_dir}")
        print("\nDownload from https://kaikki.org/dictionary/ or run download_kaikki.py")
        return 1

    print(f"\nInput: {input_file}")
    print(f"Size: {input_file.stat().st_size / (1024*1024):.1f} MB")

    total = import_kaikki(input_file, args.lang, args.limit)

    print("\n" + "=" * 60)
    print("Import complete!")
    print("=" * 60)
    print(f"Total concepts: {total:,}")

    return 0


if __name__ == "__main__":
    exit(main())
