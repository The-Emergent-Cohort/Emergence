#!/usr/bin/env python3
"""
Concept-based tokenizer database builder - ALL LANGUAGES

Structure:
- concepts: Universal, language-independent (core primitives in 0-1M range)
- lang_*: One table per language (dynamically created)
- etymology_links: Cross-language relationships that reveal shared concepts
- modals: Output routing wrappers

When words in different languages share etymology, they point to the same underlying concept.
This lets us extract the concept map from cross-linguistic patterns.
"""

import csv
import sqlite3
import hashlib
import re
from pathlib import Path
from collections import defaultdict

def sanitize_table_name(lang):
    """Convert language name to valid SQLite table name"""
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', lang.lower())
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = 'lang_' + name
    return f"lang_{name}" if name else "lang_unknown"

def stable_id(text, lang, item_type='word'):
    """Generate stable numeric ID from text, language, and type"""
    key = f"{lang}:{item_type}:{text}"
    h = hashlib.sha256(key.encode()).hexdigest()[:15]
    return int(h, 16) % (2**62)  # SQLite max int is 2^63-1

def create_core_schema(conn):
    """Create core universal tables"""
    c = conn.cursor()

    # =========================================
    # UNIVERSAL TABLES (language-independent)
    # =========================================

    # Core concepts - the universal primitives
    # IDs 0-1,000,000 reserved for core system concepts
    c.execute('''
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            canonical TEXT NOT NULL,
            domain TEXT,
            subdomain TEXT,
            description TEXT,
            composable INTEGER DEFAULT 1,
            core_level INTEGER DEFAULT 0,
            physics_type TEXT,
            body_region TEXT,
            sensory_modality TEXT,
            system_type TEXT,
            usage_notes TEXT,
            learned_associations TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT
        )
    ''')

    # Modal wrappers for output routing
    c.execute('''
        CREATE TABLE IF NOT EXISTS modals (
            id INTEGER PRIMARY KEY,
            wrapper TEXT NOT NULL UNIQUE,
            modal_type TEXT NOT NULL,
            lang TEXT,
            format TEXT,
            target_table TEXT,
            description TEXT
        )
    ''')

    # Nuance/synonym groups - links concepts to expressions across any language
    c.execute('''
        CREATE TABLE IF NOT EXISTS nuance_groups (
            id INTEGER PRIMARY KEY,
            concept_id INTEGER,
            lang TEXT NOT NULL,
            expression_id INTEGER,
            nuance TEXT,
            context_hint TEXT,
            preference_weight REAL DEFAULT 0.0,
            FOREIGN KEY(concept_id) REFERENCES concepts(id)
        )
    ''')

    # Cross-language etymology links - THIS IS THE KEY TABLE
    # When Finnish "talo" is related to Proto-Uralic "*talo" which is also
    # related to Hungarian "ház", we can infer they share a concept
    c.execute('''
        CREATE TABLE IF NOT EXISTS etymology_links (
            id INTEGER PRIMARY KEY,
            source_lang TEXT NOT NULL,
            source_term TEXT NOT NULL,
            source_id INTEGER,
            target_lang TEXT NOT NULL,
            target_term TEXT NOT NULL,
            target_id INTEGER,
            relationship TEXT,
            position INTEGER DEFAULT 0,
            UNIQUE(source_lang, source_term, target_lang, target_term, relationship)
        )
    ''')

    # Language registry - track all languages and their table names
    c.execute('''
        CREATE TABLE IF NOT EXISTS languages (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            table_name TEXT NOT NULL,
            entry_count INTEGER DEFAULT 0,
            has_morphology INTEGER DEFAULT 0
        )
    ''')

    # Token mappings: old BPE tokens -> new concept-based tokens
    c.execute('''
        CREATE TABLE IF NOT EXISTS token_mappings (
            spelling TEXT PRIMARY KEY,
            old_tokens TEXT,
            new_tokens TEXT,
            concept_ids TEXT,
            lang_ids TEXT
        )
    ''')

    # Core concept indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_canonical ON concepts(canonical)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_domain ON concepts(domain)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_physics ON concepts(physics_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_concept_system ON concepts(system_type)')

    # Etymology link indexes - crucial for concept discovery
    c.execute('CREATE INDEX IF NOT EXISTS idx_etym_source ON etymology_links(source_lang, source_term)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_etym_target ON etymology_links(target_lang, target_term)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_etym_rel ON etymology_links(relationship)')

    # Nuance indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_nuance_concept ON nuance_groups(concept_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_nuance_lang ON nuance_groups(lang)')

    # Seed core concepts
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

    # Seed some default modals
    default_modals = [
        ('code_python', 'code', None, 'python', None, 'Python code'),
        ('code_js', 'code', None, 'javascript', None, 'JavaScript code'),
        ('code_sql', 'code', None, 'sql', None, 'SQL queries'),
        ('thought', 'thought', None, None, None, 'Internal reasoning'),
        ('physics_body', 'physics', None, 'body', None, 'Body sensation'),
        ('physics_env', 'physics', None, 'environment', None, 'Environment state'),
        ('physics_contact', 'physics', None, 'contact', None, 'Contact events'),
        ('tool_call', 'tool', None, None, None, 'Tool invocation'),
        ('tool_result', 'tool', None, None, None, 'Tool response'),
    ]
    c.executemany('''
        INSERT OR IGNORE INTO modals (wrapper, modal_type, lang, format, target_table, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', default_modals)

    conn.commit()

def create_lang_table(conn, lang_name, table_name):
    """Create a language-specific table"""
    c = conn.cursor()

    # Each language gets its own expressions table
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            type TEXT NOT NULL,
            concept_id INTEGER,
            etymology_id TEXT,
            frequency INTEGER DEFAULT 0,
            usage_notes TEXT,
            last_context TEXT,
            learned_nuance TEXT,
            preference_weight REAL DEFAULT 0.0,
            register TEXT,
            intensity REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,
            FOREIGN KEY(concept_id) REFERENCES concepts(id),
            UNIQUE(text, type)
        )
    ''')

    # Decompositions table for this language
    c.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}_decomp" (
            word_id INTEGER,
            morpheme_id INTEGER,
            position INTEGER,
            morph_type TEXT,
            FOREIGN KEY(word_id) REFERENCES "{table_name}"(id),
            FOREIGN KEY(morpheme_id) REFERENCES "{table_name}"(id),
            PRIMARY KEY(word_id, position)
        )
    ''')

    # Indexes for this language
    c.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_text" ON "{table_name}"(text)')
    c.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_type" ON "{table_name}"(type)')
    c.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_concept" ON "{table_name}"(concept_id)')

    # Register in languages table and create modal
    c.execute('''
        INSERT OR IGNORE INTO languages (code, name, table_name)
        VALUES (?, ?, ?)
    ''', (lang_name.lower()[:10], lang_name, table_name))

    # Create text modal for this language
    modal_wrapper = f"text_{lang_name.lower()[:20].replace(' ', '_')}"
    c.execute('''
        INSERT OR IGNORE INTO modals (wrapper, modal_type, lang, target_table, description)
        VALUES (?, 'text', ?, ?, ?)
    ''', (modal_wrapper, lang_name, table_name, f'{lang_name} text output'))

    conn.commit()

def classify_morpheme(text, reltype):
    """Classify morpheme type based on text and relationship"""
    text = text.strip()

    if text.endswith('-') or reltype == 'has_prefix':
        return 'prefix'
    if text.startswith('-') or reltype == 'has_suffix':
        return 'suffix'
    if reltype == 'has_root' or 'Proto' in str(reltype):
        return 'root'
    if reltype == 'compound_of':
        return 'compound'
    return 'word'

def extract_all_languages(csv_path, db_path):
    """Extract ALL languages from etymology CSV"""

    conn = sqlite3.connect(db_path)
    create_core_schema(conn)
    c = conn.cursor()

    # Track created tables
    lang_tables = {}  # lang_name -> table_name
    lang_seen = {}    # (lang, text, type) -> id
    lang_decomps = defaultdict(list)  # (lang, word_text) -> [(morph_text, position, type)]
    lang_counts = defaultdict(int)

    print("Pass 1: Reading all etymology data...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        row_count = 0
        etym_links = []

        for row in reader:
            row_count += 1
            if row_count % 200000 == 0:
                print(f"  Processed {row_count:,} rows, {len(lang_tables)} languages...")

            lang = row['lang']
            term = row['term']
            reltype = row['reltype']
            related = row['related_term']
            related_lang = row.get('related_lang', lang)  # Some entries have related_lang
            position = int(row['position']) if row['position'] else 0
            etym_id = row['term_id']

            # Create table for this language if needed
            if lang not in lang_tables:
                table_name = sanitize_table_name(lang)
                lang_tables[lang] = table_name
                create_lang_table(conn, lang, table_name)

            table_name = lang_tables[lang]

            # Insert term into language table
            word_key = (lang, term, 'word')
            if word_key not in lang_seen:
                word_id = stable_id(term, lang, 'word')
                try:
                    c.execute(f'''
                        INSERT OR IGNORE INTO "{table_name}" (id, text, type, etymology_id)
                        VALUES (?, ?, 'word', ?)
                    ''', (word_id, term, etym_id))
                    lang_seen[word_key] = word_id
                    lang_counts[lang] += 1
                except sqlite3.IntegrityError:
                    pass

            # Track cross-language etymology links
            if related and related_lang:
                etym_links.append((
                    lang, term, lang_seen.get(word_key),
                    related_lang, related, None,
                    reltype, position
                ))

            # Handle morphological relationships within same language
            if reltype in ('has_affix', 'has_prefix', 'has_suffix', 'has_prefix_with_root'):
                if related:
                    morph_type = classify_morpheme(related, reltype)
                    morph_key = (lang, related, morph_type)

                    if morph_key not in lang_seen:
                        morph_id = stable_id(related, lang, morph_type)
                        try:
                            c.execute(f'''
                                INSERT OR IGNORE INTO "{table_name}" (id, text, type)
                                VALUES (?, ?, ?)
                            ''', (morph_id, related, morph_type))
                            lang_seen[morph_key] = morph_id
                            lang_counts[lang] += 1
                        except sqlite3.IntegrityError:
                            pass

                    decomp_key = (lang, term)
                    lang_decomps[decomp_key].append((related, position, morph_type))

    print(f"\nProcessed {row_count:,} total rows")
    print(f"Created {len(lang_tables)} language tables")

    # Insert etymology links
    print(f"\nInserting {len(etym_links):,} etymology links...")
    link_count = 0
    for link in etym_links:
        try:
            c.execute('''
                INSERT OR IGNORE INTO etymology_links
                (source_lang, source_term, source_id, target_lang, target_term, target_id, relationship, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', link)
            link_count += 1
        except sqlite3.IntegrityError:
            pass
    print(f"Inserted {link_count:,} unique etymology links")

    # Build decompositions for each language
    print("\nBuilding decompositions...")
    decomp_count = 0
    for (lang, word_text), morphs in lang_decomps.items():
        word_key = (lang, word_text, 'word')
        if word_key not in lang_seen:
            continue

        word_id = lang_seen[word_key]
        table_name = lang_tables[lang]
        morphs.sort(key=lambda x: x[1])

        for morph_text, pos, morph_type in morphs:
            morph_key = (lang, morph_text, morph_type)
            if morph_key in lang_seen:
                morph_id = lang_seen[morph_key]
                try:
                    c.execute(f'''
                        INSERT OR IGNORE INTO "{table_name}_decomp"
                        (word_id, morpheme_id, position, morph_type)
                        VALUES (?, ?, ?, ?)
                    ''', (word_id, morph_id, pos, morph_type))
                    decomp_count += 1
                except sqlite3.IntegrityError:
                    pass

    print(f"Created {decomp_count:,} decomposition links")

    # Update language entry counts
    for lang, count in lang_counts.items():
        c.execute('''
            UPDATE languages SET entry_count = ? WHERE name = ?
        ''', (count, lang))

    conn.commit()

    # Print stats
    print("\n=== FINAL STATISTICS ===")
    c.execute("SELECT COUNT(*) FROM languages")
    print(f"Languages: {c.fetchone()[0]:,}")

    c.execute("SELECT SUM(entry_count) FROM languages")
    total = c.fetchone()[0]
    print(f"Total entries across all languages: {total:,}")

    c.execute("SELECT COUNT(*) FROM etymology_links")
    print(f"Etymology links: {c.fetchone()[0]:,}")

    c.execute("SELECT COUNT(*) FROM concepts")
    print(f"Core concepts: {c.fetchone()[0]:,}")

    print("\nTop 20 languages by entry count:")
    c.execute("SELECT name, entry_count FROM languages ORDER BY entry_count DESC LIMIT 20")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    # Sample cross-language links
    print("\nSample etymology links (concept discovery candidates):")
    c.execute('''
        SELECT source_lang, source_term, target_lang, target_term, relationship
        FROM etymology_links
        WHERE source_lang != target_lang
        LIMIT 10
    ''')
    for row in c.fetchall():
        print(f"  {row[0]}:{row[1]} --[{row[4]}]--> {row[2]}:{row[3]}")

    conn.close()
    print(f"\nDatabase saved to: {db_path}")

if __name__ == '__main__':
    csv_path = Path('etymology.csv')
    db_path = Path('tokenizer.db')

    if not csv_path.exists():
        print("Error: etymology.csv not found")
        print("Run: gunzip etymology.csv.gz")
        exit(1)

    extract_all_languages(csv_path, db_path)
