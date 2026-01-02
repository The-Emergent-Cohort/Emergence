#!/usr/bin/env python3
"""
Initialize all database schemas for the tokenizer system.

Creates:
- db/primitives.db
- db/language_registry.db
- db/lang/ directory (for per-language DBs)

Run from: /usr/share/databases/scripts/
"""

import sqlite3
from pathlib import Path
import sys

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"

# Schema definitions

PRIMITIVES_SCHEMA = """
-- Semantic primitives: NSM primes, image schemas, verb classes
CREATE TABLE IF NOT EXISTS primitives (
    primitive_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    source TEXT NOT NULL,
    domain INTEGER NOT NULL,
    category INTEGER NOT NULL,
    description TEXT,
    examples TEXT,
    UNIQUE(canonical_name, source)
);

-- Cross-linguistic surface forms for primitives
CREATE TABLE IF NOT EXISTS primitive_forms (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    lang_genomic TEXT NOT NULL,
    surface_form TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id)
);

-- Relations between primitives
CREATE TABLE IF NOT EXISTS primitive_relations (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    related_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    FOREIGN KEY (related_id) REFERENCES primitives(primitive_id)
);

-- Semantic domains
CREATE TABLE IF NOT EXISTS domains (
    domain_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT
);

-- Categories within domains
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (domain_id) REFERENCES domains(domain_id),
    UNIQUE(domain_id, name)
);

-- Master token index for primitives
CREATE TABLE IF NOT EXISTS token_index (
    idx INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    description TEXT,
    location TEXT NOT NULL DEFAULT 'primitives'
);

CREATE INDEX IF NOT EXISTS idx_prim_name ON primitives(canonical_name);
CREATE INDEX IF NOT EXISTS idx_prim_source ON primitives(source);
CREATE INDEX IF NOT EXISTS idx_prim_forms_lang ON primitive_forms(lang_genomic);
CREATE INDEX IF NOT EXISTS idx_prim_forms_surface ON primitive_forms(surface_form);
CREATE INDEX IF NOT EXISTS idx_token_location ON token_index(location);
"""

LANGUAGE_REGISTRY_SCHEMA = """
-- Language family codes
CREATE TABLE IF NOT EXISTS language_families (
    family_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    glottocode TEXT,
    description TEXT
);

-- Language codes with genomic notation
CREATE TABLE IF NOT EXISTS language_codes (
    id INTEGER PRIMARY KEY,
    family INTEGER NOT NULL,
    subfamily INTEGER NOT NULL,
    language INTEGER NOT NULL,
    dialect INTEGER DEFAULT 0,
    genomic_code TEXT GENERATED ALWAYS AS (
        family || '.' || subfamily || '.' || language || '.' || dialect
    ) STORED,
    iso639_3 TEXT,
    iso639_1 TEXT,
    glottocode TEXT,
    name TEXT NOT NULL,
    native_name TEXT,
    speaker_count INTEGER,
    status TEXT DEFAULT 'living',
    parent_glottocode TEXT,
    level TEXT,
    UNIQUE(family, subfamily, language, dialect)
);

-- Typological and grammatical features
CREATE TABLE IF NOT EXISTS language_features (
    id INTEGER PRIMARY KEY,
    lang_id INTEGER NOT NULL,
    feature_id TEXT NOT NULL,
    value TEXT NOT NULL,
    value_name TEXT,
    source TEXT NOT NULL,
    FOREIGN KEY (lang_id) REFERENCES language_codes(id),
    UNIQUE(lang_id, feature_id, source)
);

CREATE INDEX IF NOT EXISTS idx_lang_genomic ON language_codes(genomic_code);
CREATE INDEX IF NOT EXISTS idx_lang_iso ON language_codes(iso639_3);
CREATE INDEX IF NOT EXISTS idx_lang_glotto ON language_codes(glottocode);
CREATE INDEX IF NOT EXISTS idx_lang_family ON language_codes(family);
CREATE INDEX IF NOT EXISTS idx_features_lang ON language_features(lang_id);
CREATE INDEX IF NOT EXISTS idx_features_id ON language_features(feature_id);
"""

LANGUAGE_DB_SCHEMA = """
-- Concepts in this language
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,
    abstraction INTEGER NOT NULL DEFAULT 99,
    fingerprint INTEGER,
    source TEXT,
    UNIQUE(lemma, pos)
);

-- Surface forms (inflections, variants)
CREATE TABLE IF NOT EXISTS surface_forms (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    form TEXT NOT NULL,
    form_type TEXT,
    features TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Primitive compositions for each concept
CREATE TABLE IF NOT EXISTS compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    primitive_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    relation TEXT DEFAULT 'part',
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Dialect divergence tracking
CREATE TABLE IF NOT EXISTS dialect_status (
    concept_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'shared',
    divergent_dialects TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Translations to other languages
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,
    target_lemma TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

-- Local token index
CREATE TABLE IF NOT EXISTS token_index (
    idx INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    description TEXT,
    location TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_surface_form ON surface_forms(form);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_abstraction ON concepts(abstraction);
CREATE INDEX IF NOT EXISTS idx_concept_token ON concepts(token_id);
CREATE INDEX IF NOT EXISTS idx_trans_target ON translations(target_lang, target_lemma);
"""


def init_db(db_path: Path, schema: str, name: str):
    """Initialize a database with the given schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing if present
    if db_path.exists():
        print(f"  Removing existing {name}...")
        db_path.unlink()

    print(f"  Creating {name}...")
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print(f"  ✓ {name} initialized")


def init_primitives():
    """Initialize primitives.db with domains and categories."""
    db_path = DB_DIR / "primitives.db"
    init_db(db_path, PRIMITIVES_SCHEMA, "primitives.db")

    # Seed domains
    domains = [
        (1, "physical", "Matter, space, motion"),
        (2, "temporal", "Time, sequence, duration"),
        (3, "mental", "Cognition, perception, emotion"),
        (4, "social", "Relations, communication, groups"),
        (5, "abstract", "Logic, mathematics, categories"),
        (6, "biological", "Life, organisms, health"),
        (7, "artifact", "Made things, tools, technology"),
        (8, "natural", "Nature, environment, elements"),
        (9, "evaluative", "Good/bad, values, judgments"),
    ]

    # Seed categories
    categories = [
        # Physical (1)
        (1, 1, "motion", "Movement through space"),
        (2, 1, "location", "Position and place"),
        (3, 1, "contact", "Physical touching"),
        (4, 1, "change", "State transitions"),
        (5, 1, "state", "Physical conditions"),
        (6, 1, "size", "Magnitude and extent"),
        (7, 1, "shape", "Form and structure"),
        (8, 1, "substance", "Material composition"),
        # Temporal (2)
        (9, 2, "sequence", "Order of events"),
        (10, 2, "duration", "Length of time"),
        (11, 2, "frequency", "Repetition rate"),
        (12, 2, "tense", "Time reference"),
        # Mental (3)
        (13, 3, "perception", "Sensory experience"),
        (14, 3, "cognition", "Thinking processes"),
        (15, 3, "emotion", "Feelings and affect"),
        (16, 3, "volition", "Will and intention"),
        (17, 3, "memory", "Recall and storage"),
        (18, 3, "attention", "Focus and awareness"),
        (19, 3, "understanding", "Comprehension"),
        (20, 3, "belief", "Mental states about truth"),
        # Social (4)
        (21, 4, "communication", "Information exchange"),
        (22, 4, "relation", "Interpersonal connections"),
        (23, 4, "group", "Collective entities"),
        (24, 4, "possession", "Ownership and control"),
        (25, 4, "exchange", "Transfer between parties"),
        (26, 4, "conflict", "Opposition and competition"),
        (27, 4, "cooperation", "Joint action"),
        (28, 4, "authority", "Power and hierarchy"),
        # Abstract (5)
        (29, 5, "quantity", "Amount and number"),
        (30, 5, "degree", "Intensity and extent"),
        (31, 5, "similarity", "Likeness and difference"),
        (32, 5, "logic", "Reasoning and inference"),
        (33, 5, "category", "Classification"),
        (34, 5, "part_whole", "Composition and inclusion"),
    ]

    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO domains (domain_id, name, description) VALUES (?, ?, ?)",
        domains
    )
    conn.executemany(
        "INSERT INTO categories (category_id, domain_id, name, description) VALUES (?, ?, ?, ?)",
        categories
    )
    conn.commit()
    conn.close()
    print(f"  ✓ Seeded {len(domains)} domains and {len(categories)} categories")


def init_language_registry():
    """Initialize language_registry.db."""
    db_path = DB_DIR / "language_registry.db"
    init_db(db_path, LANGUAGE_REGISTRY_SCHEMA, "language_registry.db")

    # Seed major language families
    families = [
        (0, "universal", None, "Cross-linguistic primitives"),
        (1, "indo-european", "indo1319", "Indo-European languages"),
        (2, "sino-tibetan", "sino1245", "Sino-Tibetan languages"),
        (3, "afro-asiatic", "afro1255", "Afro-Asiatic languages"),
        (4, "niger-congo", "atla1278", "Atlantic-Congo languages"),
        (5, "austronesian", "aust1307", "Austronesian languages"),
        (6, "dravidian", "drav1251", "Dravidian languages"),
        (7, "uralic", "ural1272", "Uralic languages"),
        (8, "turkic", "turk1311", "Turkic languages"),
        (9, "japonic", "japo1237", "Japonic languages"),
        (10, "koreanic", "kore1284", "Koreanic languages"),
        (11, "austroasiatic", "aust1305", "Austroasiatic languages"),
        (12, "tai-kadai", "taik1256", "Tai-Kadai languages"),
        (13, "nilo-saharan", "nilo1247", "Nilo-Saharan languages"),
    ]

    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO language_families (family_id, name, glottocode, description) VALUES (?, ?, ?, ?)",
        families
    )
    conn.commit()
    conn.close()
    print(f"  ✓ Seeded {len(families)} language families")


def init_lang_directory():
    """Create the lang/ directory for per-language databases."""
    lang_dir = DB_DIR / "lang"
    lang_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {lang_dir}")


def create_language_db(iso_code: str):
    """Create a new per-language database."""
    db_path = DB_DIR / "lang" / f"{iso_code}.db"
    init_db(db_path, LANGUAGE_DB_SCHEMA, f"lang/{iso_code}.db")
    return db_path


def main():
    print("=" * 60)
    print("Initializing Tokenizer Database System")
    print("=" * 60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Database directory: {DB_DIR}")
    print()

    # Create directories
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize databases
    print("1. Initializing primitives.db...")
    init_primitives()
    print()

    print("2. Initializing language_registry.db...")
    init_language_registry()
    print()

    print("3. Creating lang/ directory...")
    init_lang_directory()
    print()

    print("=" * 60)
    print("Initialization complete!")
    print("=" * 60)
    print("\nCreated:")
    print(f"  - {DB_DIR / 'primitives.db'}")
    print(f"  - {DB_DIR / 'language_registry.db'}")
    print(f"  - {DB_DIR / 'lang/'}")
    print("\nNext: Run import_nsm_primes.py to populate primitives")


if __name__ == "__main__":
    main()
