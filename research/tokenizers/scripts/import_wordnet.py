#!/usr/bin/env python3
"""Import Open English WordNet synsets as concepts into tokenizer database

WordNet provides the middle layer we need:
- Synsets = concept clusters (words grouped by meaning)
- Glosses = concept definitions
- Relations = hypernym (is-a), meronym (part-of), etc.

This gives us ~117k distinct concepts with definitions and relationships.
"""

import sqlite3
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_PATH = BASE_DIR / "db" / "tokenizer.db"
WORDNET_PATH = BASE_DIR / "reference" / "wordnet" / "oewn.sqlite"

SOURCE_NAME = "wordnet"
SOURCE_CONFIDENCE = 0.90


def explore_schema(wn_conn):
    """Print the WordNet database schema to understand its structure"""
    cur = wn_conn.cursor()

    print("=== WordNet Database Schema ===\n")

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    print(f"Tables: {tables}\n")

    for table in tables:
        print(f"--- {table} ---")
        cur.execute(f"PRAGMA table_info({table})")
        cols = cur.fetchall()
        for col in cols:
            print(f"  {col[1]} ({col[2]})")

        # Sample row
        cur.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cur.fetchone()
        if sample:
            print(f"  Sample: {sample[:5]}..." if len(sample) > 5 else f"  Sample: {sample}")
        print()

    # Count synsets
    for table in tables:
        if 'synset' in table.lower():
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table} count: {count}")


def get_or_create_source(cur) -> int:
    """Register WordNet as a data source"""
    cur.execute("""
        INSERT OR IGNORE INTO data_sources (name, source_type, base_confidence, description, url)
        VALUES (?, 'lexicon', ?, 'Open English WordNet synsets and glosses',
                'https://github.com/globalwordnet/english-wordnet')
    """, (SOURCE_NAME, SOURCE_CONFIDENCE))
    cur.execute("SELECT id FROM data_sources WHERE name = ?", (SOURCE_NAME,))
    return cur.fetchone()[0]


def import_synsets(wn_conn, tok_conn):
    """Import WordNet synsets as concepts"""
    wn_cur = wn_conn.cursor()
    tok_cur = tok_conn.cursor()

    source_id = get_or_create_source(tok_cur)
    tok_conn.commit()

    # First, understand the schema
    wn_cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in wn_cur.fetchall()}

    print(f"  WordNet tables: {sorted(tables)}")

    # Try to find synsets table - OEWN uses different naming
    synset_table = None
    gloss_col = None
    pos_col = None

    # Check for common table names
    for candidate in ['synsets', 'Synsets', 'synset', 'Synset']:
        if candidate in tables:
            synset_table = candidate
            break

    if not synset_table:
        # Look for any table with 'synset' in name
        for t in tables:
            if 'synset' in t.lower():
                synset_table = t
                break

    if not synset_table:
        print("  ERROR: Could not find synsets table")
        print(f"  Available tables: {tables}")
        return 0, 0

    # Get columns
    wn_cur.execute(f"PRAGMA table_info({synset_table})")
    cols = {row[1].lower(): row[1] for row in wn_cur.fetchall()}
    print(f"  Synset columns: {list(cols.keys())}")

    # Find gloss column
    for candidate in ['gloss', 'definition', 'def', 'glosses']:
        if candidate in cols:
            gloss_col = cols[candidate]
            break

    # Find POS column
    for candidate in ['pos', 'part_of_speech', 'posid', 'pos_id']:
        if candidate in cols:
            pos_col = cols[candidate]
            break

    # Find ID column
    id_col = None
    for candidate in ['synsetid', 'synset_id', 'id', 'wnid']:
        if candidate in cols:
            id_col = cols[candidate]
            break

    print(f"  Using: table={synset_table}, id={id_col}, gloss={gloss_col}, pos={pos_col}")

    if not id_col:
        print("  ERROR: Could not find synset ID column")
        return 0, 0

    # Build query
    select_cols = [id_col]
    if gloss_col:
        select_cols.append(gloss_col)
    if pos_col:
        select_cols.append(pos_col)

    query = f"SELECT {', '.join(select_cols)} FROM {synset_table}"
    wn_cur.execute(query)

    # Check if we have a concepts table or should use morphemes
    tok_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tok_tables = {row[0] for row in tok_cur.fetchall()}

    use_concepts = 'concepts' in tok_tables

    synset_count = 0
    batch = []
    batch_size = 1000

    for row in wn_cur:
        synset_id = row[0]
        gloss = row[1] if len(row) > 1 and gloss_col else None
        pos = row[2] if len(row) > 2 and pos_col else None

        # Create a canonical form from synset_id (e.g., "wn:ball.n.01")
        canonical = f"wn:{synset_id}" if not synset_id.startswith('wn:') else synset_id

        batch.append((canonical, gloss, pos))

        if len(batch) >= batch_size:
            if use_concepts:
                tok_cur.executemany("""
                    INSERT OR IGNORE INTO concepts (concept, gloss, pos)
                    VALUES (?, ?, ?)
                """, batch)
            else:
                # Fall back to morphemes table
                tok_cur.executemany("""
                    INSERT OR IGNORE INTO morphemes (morpheme, category)
                    VALUES (?, ?)
                """, [(b[0], b[2] or 'synset') for b in batch])

            synset_count += tok_cur.rowcount
            tok_conn.commit()
            batch = []

    # Final batch
    if batch:
        if use_concepts:
            tok_cur.executemany("""
                INSERT OR IGNORE INTO concepts (concept, gloss, pos)
                VALUES (?, ?, ?)
            """, batch)
        else:
            tok_cur.executemany("""
                INSERT OR IGNORE INTO morphemes (morpheme, category)
                VALUES (?, ?)
            """, [(b[0], b[2] or 'synset') for b in batch])
        synset_count += tok_cur.rowcount
        tok_conn.commit()

    return synset_count


def import_words_to_synsets(wn_conn, tok_conn):
    """Import word-to-synset mappings as surface forms"""
    wn_cur = wn_conn.cursor()
    tok_cur = tok_conn.cursor()

    # Find the words/senses table
    wn_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in wn_cur.fetchall()}

    # Look for sense or word tables
    sense_table = None
    for candidate in ['senses', 'Senses', 'sense', 'words', 'Words', 'entries']:
        if candidate in tables:
            sense_table = candidate
            break

    if not sense_table:
        for t in tables:
            if 'sense' in t.lower() or 'word' in t.lower():
                sense_table = t
                break

    if not sense_table:
        print("  Could not find senses/words table, skipping word mappings")
        return 0

    wn_cur.execute(f"PRAGMA table_info({sense_table})")
    cols = {row[1].lower(): row[1] for row in wn_cur.fetchall()}
    print(f"  Sense table columns: {list(cols.keys())}")

    # This is database-structure dependent - we'll log what we find
    print(f"  Found sense table: {sense_table}")
    wn_cur.execute(f"SELECT * FROM {sense_table} LIMIT 3")
    samples = wn_cur.fetchall()
    for s in samples:
        print(f"    Sample: {s}")

    return 0  # TODO: implement based on actual schema


def main():
    print(f"Importing WordNet from {WORDNET_PATH}")

    if not WORDNET_PATH.exists():
        print(f"ERROR: WordNet database not found at {WORDNET_PATH}")
        print("Run: python download_sources.py --base-dir ../reference --sources wordnet")
        return 1

    wn_conn = sqlite3.connect(WORDNET_PATH)
    tok_conn = sqlite3.connect(DB_PATH)
    tok_conn.execute("PRAGMA foreign_keys = ON")

    try:
        # First, explore the schema
        print("\n--- Exploring WordNet Schema ---")
        explore_schema(wn_conn)

        # Import synsets
        print("\n--- Importing Synsets ---")
        synset_count = import_synsets(wn_conn, tok_conn)
        print(f"  Imported {synset_count} synsets")

        # Import word mappings
        print("\n--- Importing Word Mappings ---")
        word_count = import_words_to_synsets(wn_conn, tok_conn)
        print(f"  Imported {word_count} word-synset mappings")

        # Summary
        tok_cur = tok_conn.cursor()
        tok_cur.execute("SELECT COUNT(*) FROM morphemes")
        total = tok_cur.fetchone()[0]
        print(f"\n  Morphemes/concepts table now has: {total} entries")

    finally:
        wn_conn.close()
        tok_conn.close()

    print("\nWordNet import complete!")
    return 0


if __name__ == '__main__':
    exit(main())
