#!/usr/bin/env python3
"""
Etymology-based tokenizer database builder

Extracts English morphological data from etymology-db and builds
a SQLite database for tokenizer remapping.

Numbering scheme:
- Big numbers with gaps (tokenizer provides condensed index at runtime)
- Semantic ranges for different morpheme types:
  - 1_000_000_000+ : Roots
  - 2_000_000_000+ : Prefixes
  - 3_000_000_000+ : Suffixes
  - 4_000_000_000+ : Complete words
  - 5_000_000_000+ : Compounds
"""

import csv
import sqlite3
import hashlib
from pathlib import Path

# ID ranges for semantic grouping
ID_ROOTS = 1_000_000_000
ID_PREFIXES = 2_000_000_000
ID_SUFFIXES = 3_000_000_000
ID_WORDS = 4_000_000_000
ID_COMPOUNDS = 5_000_000_000

def stable_id(text, range_base):
    """Generate stable numeric ID from text within a range"""
    h = hashlib.sha256(text.encode()).hexdigest()[:15]
    offset = int(h, 16) % 900_000_000  # Leave room in range
    return range_base + offset

def create_schema(conn):
    """Create database schema"""
    c = conn.cursor()

    # Core morphemes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS morphemes (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            type TEXT NOT NULL,  -- 'root', 'prefix', 'suffix', 'word', 'compound'
            lang TEXT DEFAULT 'en',        -- ISO 639-1 language code
            modal TEXT DEFAULT 'text',     -- Modality: 'text', 'code', 'speech', 'thought', 'physics'
            etymology_id TEXT,             -- Original ID from etymology-db
            frequency INTEGER DEFAULT 0,   -- Usage count in vocabulary
            concept_id INTEGER,            -- Link to language-independent concept

            -- DI learning/annotation columns
            usage_notes TEXT,              -- DI notes about usage
            last_context TEXT,             -- Last context where used
            learned_nuance TEXT,           -- Personal nuance observations
            preference_weight REAL DEFAULT 0.0,  -- Learned preference (-1 to 1)
            register TEXT,                 -- 'formal', 'casual', 'technical', etc.
            intensity REAL,                -- Emotional/semantic intensity (0 to 1)

            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,

            UNIQUE(text, type, lang)
        )
    ''')

    # Modal wrappers for output routing
    c.execute('''
        CREATE TABLE IF NOT EXISTS modals (
            id INTEGER PRIMARY KEY,
            wrapper TEXT NOT NULL UNIQUE,  -- e.g., 'text_en', 'code_python', 'thought'
            modal_type TEXT NOT NULL,      -- 'text', 'code', 'speech', 'thought', 'physics'
            lang TEXT,                     -- Language code if applicable
            format TEXT,                   -- Additional format info (e.g., 'python', 'json')
            description TEXT
        )
    ''')

    # Concept table: language-independent meaning
    c.execute('''
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            canonical TEXT,                -- Canonical representation
            domain TEXT,                   -- Semantic domain: 'physical', 'abstract', 'emotional', etc.
            subdomain TEXT,                -- More specific category
            description TEXT,
            composable INTEGER DEFAULT 1,  -- Can this concept compose with others?
            core_level INTEGER DEFAULT 0,  -- 0=derived, 1-10=core primitives

            -- For physics/body concepts (0-1M range)
            physics_type TEXT,             -- 'position', 'velocity', 'force', 'contact', etc.
            body_region TEXT,              -- 'hand', 'arm', 'head', etc. if applicable
            sensory_modality TEXT,         -- 'visual', 'tactile', 'auditory', etc.

            -- Learning annotations
            usage_notes TEXT,
            learned_associations TEXT      -- JSON: related concepts learned through experience
        )
    ''')

    # Nuance/synonym groups for generation queries
    c.execute('''
        CREATE TABLE IF NOT EXISTS nuance_groups (
            id INTEGER PRIMARY KEY,
            concept_id INTEGER,            -- Base concept
            morpheme_id INTEGER,           -- A word expressing this concept
            nuance TEXT,                   -- 'intense', 'mild', 'formal', 'casual', etc.
            context_hint TEXT,             -- When to prefer this form
            preference_weight REAL DEFAULT 0.0,
            FOREIGN KEY(concept_id) REFERENCES concepts(id),
            FOREIGN KEY(morpheme_id) REFERENCES morphemes(id)
        )
    ''')

    # Word decomposition: how words break into morphemes
    c.execute('''
        CREATE TABLE IF NOT EXISTS decompositions (
            word_id INTEGER,
            morpheme_id INTEGER,
            position INTEGER,      -- Order in the word (0, 1, 2...)
            morph_type TEXT,       -- 'prefix', 'root', 'suffix'
            FOREIGN KEY(word_id) REFERENCES morphemes(id),
            FOREIGN KEY(morpheme_id) REFERENCES morphemes(id),
            PRIMARY KEY(word_id, position)
        )
    ''')

    # Etymology relationships (for reference, not tokenization)
    c.execute('''
        CREATE TABLE IF NOT EXISTS etymology (
            term_id INTEGER,
            related_id INTEGER,
            relationship TEXT,     -- 'borrowed_from', 'derived_from', etc.
            FOREIGN KEY(term_id) REFERENCES morphemes(id),
            FOREIGN KEY(related_id) REFERENCES morphemes(id)
        )
    ''')

    # Token mappings: old BPE tokens -> new morpheme tokens
    c.execute('''
        CREATE TABLE IF NOT EXISTS token_mappings (
            spelling TEXT PRIMARY KEY,  -- The actual character sequence
            old_tokens TEXT,            -- JSON: how BPE tokenizes it
            new_tokens TEXT,            -- JSON: morpheme-based tokenization
            morpheme_ids TEXT           -- JSON: list of morpheme IDs
        )
    ''')

    # Indexes for efficient lookup
    c.execute('CREATE INDEX IF NOT EXISTS idx_morpheme_text ON morphemes(text)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_morpheme_type ON morphemes(type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_morpheme_lang ON morphemes(lang)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_morpheme_modal ON morphemes(modal)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_morpheme_concept ON morphemes(concept_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_decomp_word ON decompositions(word_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_modal_wrapper ON modals(wrapper)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_domain ON concepts(domain)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_physics ON concepts(physics_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_nuance_concept ON nuance_groups(concept_id)')

    # Insert default modal wrappers
    default_modals = [
        ('text_en', 'text', 'en', None, 'English text output'),
        ('text_es', 'text', 'es', None, 'Spanish text output'),
        ('text_fr', 'text', 'fr', None, 'French text output'),
        ('text_de', 'text', 'de', None, 'German text output'),
        ('text_zh', 'text', 'zh', None, 'Chinese text output'),
        ('code_python', 'code', None, 'python', 'Python code'),
        ('code_js', 'code', None, 'javascript', 'JavaScript code'),
        ('code_sql', 'code', None, 'sql', 'SQL queries'),
        ('thought', 'thought', None, None, 'Internal reasoning (scratchpad)'),
        ('physics_body', 'physics', None, 'body', 'Body sensation/proprioception'),
        ('physics_env', 'physics', None, 'environment', 'Environment state'),
        ('physics_contact', 'physics', None, 'contact', 'Contact/collision events'),
        ('speech_en', 'speech', 'en', None, 'English speech output'),
        ('tool_call', 'tool', None, None, 'Tool invocation'),
        ('tool_result', 'tool', None, None, 'Tool response'),
    ]
    c.executemany('''
        INSERT OR IGNORE INTO modals (wrapper, modal_type, lang, format, description)
        VALUES (?, ?, ?, ?, ?)
    ''', default_modals)

    conn.commit()

def classify_morpheme(text, reltype):
    """Classify morpheme type based on text and relationship"""
    text = text.strip()

    # Prefixes typically start with text and end with -
    if text.endswith('-') or reltype == 'has_prefix':
        return 'prefix', ID_PREFIXES

    # Suffixes typically start with -
    if text.startswith('-') or reltype == 'has_suffix':
        return 'suffix', ID_SUFFIXES

    # Roots from PIE or explicit root relationships
    if reltype == 'has_root' or 'Proto-Indo-European' in str(reltype):
        return 'root', ID_ROOTS

    # Compounds
    if reltype == 'compound_of':
        return 'compound', ID_COMPOUNDS

    # Default to word
    return 'word', ID_WORDS

def extract_english_morphology(csv_path, db_path):
    """Extract English morphological data from etymology CSV"""

    conn = sqlite3.connect(db_path)
    create_schema(conn)
    c = conn.cursor()

    # Track what we've inserted
    seen_morphemes = {}  # (text, type) -> id
    word_decompositions = {}  # word_text -> [(morpheme_text, position, type)]

    print("Reading etymology data...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        row_count = 0
        english_count = 0

        for row in reader:
            row_count += 1
            if row_count % 100000 == 0:
                print(f"  Processed {row_count:,} rows...")

            # Filter for English
            if row['lang'] != 'English':
                continue

            lang_code = 'en'  # ISO 639-1

            english_count += 1

            term = row['term']
            reltype = row['reltype']
            related = row['related_term']
            position = int(row['position']) if row['position'] else 0
            etym_id = row['term_id']

            # Insert the main term as a word
            word_key = (term, 'word', lang_code)
            if word_key not in seen_morphemes:
                word_id = stable_id(term, ID_WORDS)
                try:
                    c.execute('''
                        INSERT OR IGNORE INTO morphemes (id, text, type, lang, etymology_id)
                        VALUES (?, ?, 'word', ?, ?)
                    ''', (word_id, term, lang_code, etym_id))
                    seen_morphemes[word_key] = word_id
                except sqlite3.IntegrityError:
                    pass

            # Handle affix relationships (morphological decomposition)
            if reltype in ('has_affix', 'has_prefix', 'has_suffix', 'has_prefix_with_root'):
                if related:
                    morph_type, id_range = classify_morpheme(related, reltype)
                    morph_key = (related, morph_type, lang_code)

                    if morph_key not in seen_morphemes:
                        morph_id = stable_id(related + morph_type, id_range)
                        try:
                            c.execute('''
                                INSERT OR IGNORE INTO morphemes (id, text, type, lang)
                                VALUES (?, ?, ?, ?)
                            ''', (morph_id, related, morph_type, lang_code))
                            seen_morphemes[morph_key] = morph_id
                        except sqlite3.IntegrityError:
                            pass

                    # Track decomposition
                    if term not in word_decompositions:
                        word_decompositions[term] = []
                    word_decompositions[term].append((related, position, morph_type, lang_code))

    print(f"Processed {row_count:,} total rows, {english_count:,} English")
    print(f"Found {len(seen_morphemes):,} unique morphemes")

    # Insert decompositions
    print("Building decompositions...")
    decomp_count = 0
    for word_text, morphs in word_decompositions.items():
        word_key = (word_text, 'word', 'en')
        if word_key not in seen_morphemes:
            continue

        word_id = seen_morphemes[word_key]

        # Sort by position
        morphs.sort(key=lambda x: x[1])

        for morph_text, pos, morph_type, lang in morphs:
            morph_key = (morph_text, morph_type, lang)
            if morph_key in seen_morphemes:
                morph_id = seen_morphemes[morph_key]
                try:
                    c.execute('''
                        INSERT OR IGNORE INTO decompositions
                        (word_id, morpheme_id, position, morph_type)
                        VALUES (?, ?, ?, ?)
                    ''', (word_id, morph_id, pos, morph_type))
                    decomp_count += 1
                except sqlite3.IntegrityError:
                    pass

    print(f"Created {decomp_count:,} decomposition links")

    conn.commit()

    # Stats
    c.execute("SELECT type, COUNT(*) FROM morphemes GROUP BY type")
    print("\nMorpheme counts by type:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    c.execute("SELECT COUNT(*) FROM decompositions")
    print(f"\nTotal decomposition links: {c.fetchone()[0]:,}")

    conn.close()
    print(f"\nDatabase saved to: {db_path}")

if __name__ == '__main__':
    csv_path = Path('etymology.csv')
    db_path = Path('tokenizer.db')

    if not csv_path.exists():
        print("Error: etymology.csv not found")
        print("Run: gunzip etymology.csv.gz")
        exit(1)

    extract_english_morphology(csv_path, db_path)
