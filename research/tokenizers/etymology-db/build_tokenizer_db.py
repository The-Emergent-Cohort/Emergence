#!/usr/bin/env python3
"""
Concept-based tokenizer database builder

Structure:
- concepts: Universal, language-independent (core primitives in 0-1M range)
- lang_en: English-specific expressions of concepts
- modals: Output routing wrappers

Each language gets its own table structure appropriate to its morphology.
All link back to concepts via concept_id.
"""

import csv
import sqlite3
import hashlib
from pathlib import Path

# ID ranges for semantic grouping
# 0 - 1,000,000: Core concepts (physics, body, system - from model/physics specialists)
# Language-specific ranges (English):
ID_EN_ROOTS = 1_000_000_000
ID_EN_PREFIXES = 2_000_000_000
ID_EN_SUFFIXES = 3_000_000_000
ID_EN_WORDS = 4_000_000_000
ID_EN_COMPOUNDS = 5_000_000_000

def stable_id(text, range_base):
    """Generate stable numeric ID from text within a range"""
    h = hashlib.sha256(text.encode()).hexdigest()[:15]
    offset = int(h, 16) % 900_000_000  # Leave room in range
    return range_base + offset

def create_schema(conn):
    """Create database schema"""
    c = conn.cursor()

    # =========================================
    # UNIVERSAL TABLES (language-independent)
    # =========================================

    # Core concepts - the universal primitives
    # IDs 0-1,000,000 reserved for core system concepts
    c.execute('''
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            canonical TEXT NOT NULL,           -- Canonical representation
            domain TEXT,                       -- 'physical', 'abstract', 'emotional', 'system', etc.
            subdomain TEXT,                    -- More specific category
            description TEXT,
            composable INTEGER DEFAULT 1,      -- Can this concept compose with others?
            core_level INTEGER DEFAULT 0,      -- 0=derived, 1-10=core primitives

            -- For physics/body concepts (0-1M range)
            physics_type TEXT,                 -- 'position', 'velocity', 'force', 'contact', etc.
            body_region TEXT,                  -- 'hand', 'arm', 'head', etc. if applicable
            sensory_modality TEXT,             -- 'visual', 'tactile', 'auditory', etc.

            -- System concepts
            system_type TEXT,                  -- 'command', 'modifier', 'state', etc.

            -- Learning annotations
            usage_notes TEXT,
            learned_associations TEXT,         -- JSON: related concepts learned through experience

            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT
        )
    ''')

    # Modal wrappers for output routing (universal)
    c.execute('''
        CREATE TABLE IF NOT EXISTS modals (
            id INTEGER PRIMARY KEY,
            wrapper TEXT NOT NULL UNIQUE,      -- e.g., 'text_en', 'code_python', 'thought'
            modal_type TEXT NOT NULL,          -- 'text', 'code', 'speech', 'thought', 'physics'
            lang TEXT,                         -- Language code if applicable
            format TEXT,                       -- Additional format info (e.g., 'python', 'json')
            target_table TEXT,                 -- Which lang table to query (e.g., 'lang_en')
            description TEXT
        )
    ''')

    # Nuance/synonym groups - links concepts to expressions across any language
    c.execute('''
        CREATE TABLE IF NOT EXISTS nuance_groups (
            id INTEGER PRIMARY KEY,
            concept_id INTEGER,                -- Base concept
            lang TEXT NOT NULL,                -- Which language
            expression_id INTEGER,             -- ID in that language's table
            nuance TEXT,                       -- 'intense', 'mild', 'formal', 'casual', etc.
            context_hint TEXT,                 -- When to prefer this form
            preference_weight REAL DEFAULT 0.0,
            FOREIGN KEY(concept_id) REFERENCES concepts(id)
        )
    ''')

    # =========================================
    # ENGLISH LANGUAGE TABLES (lang_en)
    # =========================================

    # English expressions - structure appropriate for English morphology
    c.execute('''
        CREATE TABLE IF NOT EXISTS lang_en (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            type TEXT NOT NULL,                -- 'root', 'prefix', 'suffix', 'word', 'compound'
            concept_id INTEGER,                -- Link to universal concept
            etymology_id TEXT,                 -- Original ID from etymology-db
            frequency INTEGER DEFAULT 0,       -- Usage count

            -- DI learning/annotation columns
            usage_notes TEXT,
            last_context TEXT,
            learned_nuance TEXT,
            preference_weight REAL DEFAULT 0.0,
            register TEXT,                     -- 'formal', 'casual', 'technical', etc.
            intensity REAL,                    -- Emotional/semantic intensity (0 to 1)

            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,

            FOREIGN KEY(concept_id) REFERENCES concepts(id),
            UNIQUE(text, type)
        )
    ''')

    # English decomposition - how English words break into morphemes
    c.execute('''
        CREATE TABLE IF NOT EXISTS decompositions_en (
            word_id INTEGER,
            morpheme_id INTEGER,
            position INTEGER,                  -- Order in the word (0, 1, 2...)
            morph_type TEXT,                   -- 'prefix', 'root', 'suffix'
            FOREIGN KEY(word_id) REFERENCES lang_en(id),
            FOREIGN KEY(morpheme_id) REFERENCES lang_en(id),
            PRIMARY KEY(word_id, position)
        )
    ''')

    # English etymology relationships
    c.execute('''
        CREATE TABLE IF NOT EXISTS etymology_en (
            term_id INTEGER,
            related_id INTEGER,
            relationship TEXT,                 -- 'borrowed_from', 'derived_from', etc.
            FOREIGN KEY(term_id) REFERENCES lang_en(id),
            FOREIGN KEY(related_id) REFERENCES lang_en(id)
        )
    ''')

    # =========================================
    # MIGRATION/MAPPING TABLE
    # =========================================

    # Token mappings: old BPE tokens -> new concept-based tokens (for Mistral retrofit)
    c.execute('''
        CREATE TABLE IF NOT EXISTS token_mappings (
            spelling TEXT PRIMARY KEY,         -- The actual character sequence
            old_tokens TEXT,                   -- JSON: how BPE tokenizes it
            new_tokens TEXT,                   -- JSON: concept-based tokenization
            concept_ids TEXT,                  -- JSON: list of concept IDs
            lang_ids TEXT                      -- JSON: list of lang table IDs
        )
    ''')

    # =========================================
    # INDEXES
    # =========================================

    # Concept indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_canonical ON concepts(canonical)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_domain ON concepts(domain)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_physics ON concepts(physics_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_system ON concepts(system_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_core ON concepts(core_level)')

    # English indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_en_text ON lang_en(text)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_en_type ON lang_en(type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_en_concept ON lang_en(concept_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_decomp_en_word ON decompositions_en(word_id)')

    # Other indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_modal_wrapper ON modals(wrapper)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_nuance_concept ON nuance_groups(concept_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_nuance_lang ON nuance_groups(lang)')

    # =========================================
    # DEFAULT DATA
    # =========================================

    # Default modal wrappers
    default_modals = [
        ('text_en', 'text', 'en', None, 'lang_en', 'English text output'),
        ('text_es', 'text', 'es', None, 'lang_es', 'Spanish text output'),
        ('text_fr', 'text', 'fr', None, 'lang_fr', 'French text output'),
        ('text_de', 'text', 'de', None, 'lang_de', 'German text output'),
        ('text_zh', 'text', 'zh', None, 'lang_zh', 'Chinese text output'),
        ('code_python', 'code', None, 'python', None, 'Python code'),
        ('code_js', 'code', None, 'javascript', None, 'JavaScript code'),
        ('code_sql', 'code', None, 'sql', None, 'SQL queries'),
        ('thought', 'thought', None, None, None, 'Internal reasoning'),
        ('physics_body', 'physics', None, 'body', None, 'Body sensation'),
        ('physics_env', 'physics', None, 'environment', None, 'Environment state'),
        ('physics_contact', 'physics', None, 'contact', None, 'Contact events'),
        ('speech_en', 'speech', 'en', None, 'lang_en', 'English speech'),
        ('tool_call', 'tool', None, None, None, 'Tool invocation'),
        ('tool_result', 'tool', None, None, None, 'Tool response'),
    ]
    c.executemany('''
        INSERT OR IGNORE INTO modals (wrapper, modal_type, lang, format, target_table, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', default_modals)

    # Some core system concepts (examples - will be expanded by other specialists)
    core_concepts = [
        (1, 'STATE_OF', 'abstract', 'modifier', 'Converts X to "state of being X"', 1, 10, None, None, None, 'modifier'),
        (2, 'QUALITY_OF', 'abstract', 'modifier', 'Converts X to quality/property', 1, 10, None, None, None, 'modifier'),
        (3, 'AGENT_OF', 'abstract', 'modifier', 'One who does X', 1, 10, None, None, None, 'modifier'),
        (4, 'ACT_OF', 'abstract', 'modifier', 'The act of doing X', 1, 10, None, None, None, 'modifier'),
        (5, 'NEGATION', 'abstract', 'modifier', 'Not X, opposite of X', 1, 10, None, None, None, 'modifier'),
        (6, 'SELF', 'abstract', 'identity', 'Self-reference', 1, 10, None, None, None, 'identity'),
        (7, 'OTHER', 'abstract', 'identity', 'Other-reference', 1, 10, None, None, None, 'identity'),
        (10, 'POSITION', 'physical', 'spatial', 'Location in space', 1, 10, 'position', None, 'visual', None),
        (11, 'VELOCITY', 'physical', 'motion', 'Rate of position change', 1, 10, 'velocity', None, None, None),
        (12, 'FORCE', 'physical', 'dynamics', 'Push or pull', 1, 10, 'force', None, 'tactile', None),
        (13, 'CONTACT', 'physical', 'interaction', 'Physical touching', 1, 10, 'contact', None, 'tactile', None),
    ]
    c.executemany('''
        INSERT OR IGNORE INTO concepts
        (id, canonical, domain, subdomain, description, composable, core_level,
         physics_type, body_region, sensory_modality, system_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', core_concepts)

    conn.commit()

def classify_morpheme(text, reltype):
    """Classify morpheme type based on text and relationship"""
    text = text.strip()

    if text.endswith('-') or reltype == 'has_prefix':
        return 'prefix', ID_EN_PREFIXES

    if text.startswith('-') or reltype == 'has_suffix':
        return 'suffix', ID_EN_SUFFIXES

    if reltype == 'has_root' or 'Proto-Indo-European' in str(reltype):
        return 'root', ID_EN_ROOTS

    if reltype == 'compound_of':
        return 'compound', ID_EN_COMPOUNDS

    return 'word', ID_EN_WORDS

def extract_english_morphology(csv_path, db_path):
    """Extract English morphological data from etymology CSV"""

    conn = sqlite3.connect(db_path)
    create_schema(conn)
    c = conn.cursor()

    seen = {}  # (text, type) -> id
    word_decomps = {}  # word_text -> [(morph_text, position, type)]

    print("Reading etymology data...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        row_count = 0
        english_count = 0

        for row in reader:
            row_count += 1
            if row_count % 100000 == 0:
                print(f"  Processed {row_count:,} rows...")

            if row['lang'] != 'English':
                continue

            english_count += 1

            term = row['term']
            reltype = row['reltype']
            related = row['related_term']
            position = int(row['position']) if row['position'] else 0
            etym_id = row['term_id']

            # Insert word into lang_en
            word_key = (term, 'word')
            if word_key not in seen:
                word_id = stable_id(term, ID_EN_WORDS)
                try:
                    c.execute('''
                        INSERT OR IGNORE INTO lang_en (id, text, type, etymology_id)
                        VALUES (?, ?, 'word', ?)
                    ''', (word_id, term, etym_id))
                    seen[word_key] = word_id
                except sqlite3.IntegrityError:
                    pass

            # Handle morphological relationships
            if reltype in ('has_affix', 'has_prefix', 'has_suffix', 'has_prefix_with_root'):
                if related:
                    morph_type, id_range = classify_morpheme(related, reltype)
                    morph_key = (related, morph_type)

                    if morph_key not in seen:
                        morph_id = stable_id(related + morph_type, id_range)
                        try:
                            c.execute('''
                                INSERT OR IGNORE INTO lang_en (id, text, type)
                                VALUES (?, ?, ?)
                            ''', (morph_id, related, morph_type))
                            seen[morph_key] = morph_id
                        except sqlite3.IntegrityError:
                            pass

                    if term not in word_decomps:
                        word_decomps[term] = []
                    word_decomps[term].append((related, position, morph_type))

    print(f"Processed {row_count:,} total rows, {english_count:,} English")
    print(f"Found {len(seen):,} unique entries in lang_en")

    # Insert decompositions
    print("Building decompositions...")
    decomp_count = 0
    for word_text, morphs in word_decomps.items():
        word_key = (word_text, 'word')
        if word_key not in seen:
            continue

        word_id = seen[word_key]
        morphs.sort(key=lambda x: x[1])

        for morph_text, pos, morph_type in morphs:
            morph_key = (morph_text, morph_type)
            if morph_key in seen:
                morph_id = seen[morph_key]
                try:
                    c.execute('''
                        INSERT OR IGNORE INTO decompositions_en
                        (word_id, morpheme_id, position, morph_type)
                        VALUES (?, ?, ?, ?)
                    ''', (word_id, morph_id, pos, morph_type))
                    decomp_count += 1
                except sqlite3.IntegrityError:
                    pass

    print(f"Created {decomp_count:,} decomposition links")

    conn.commit()

    # Stats
    c.execute("SELECT type, COUNT(*) FROM lang_en GROUP BY type")
    print("\nEnglish entries by type:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    c.execute("SELECT COUNT(*) FROM concepts")
    print(f"\nCore concepts: {c.fetchone()[0]:,}")

    c.execute("SELECT COUNT(*) FROM decompositions_en")
    print(f"Decomposition links: {c.fetchone()[0]:,}")

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
