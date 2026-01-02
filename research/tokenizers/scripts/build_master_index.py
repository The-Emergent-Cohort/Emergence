#!/usr/bin/env python3
"""
Build master token index.

Aggregates token IDs from all databases into a single routing table.
Minimal data: idx, token_id, description, location

Run from: /usr/share/databases/scripts/
Requires: All imports to have been run first
"""

import sqlite3
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
PRIMITIVES_DB = DB_DIR / "primitives.db"
MASTER_INDEX_DB = DB_DIR / "master_index.db"

MASTER_SCHEMA = """
-- Master routing table for all tokens
CREATE TABLE IF NOT EXISTS token_index (
    idx INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    description TEXT,
    location TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_id ON token_index(token_id);
CREATE INDEX IF NOT EXISTS idx_location ON token_index(location);
"""


def index_primitives(cursor) -> int:
    """Index tokens from primitives.db."""
    if not PRIMITIVES_DB.exists():
        return 0

    conn = sqlite3.connect(PRIMITIVES_DB)
    prim_cursor = conn.cursor()

    # Get primitives
    prim_cursor.execute("""
        SELECT canonical_name, source, domain, category, primitive_id
        FROM primitives
    """)

    count = 0
    for row in prim_cursor.fetchall():
        name, source, domain, category, prim_id = row
        # Generate token_id for primitive
        token_id = f"1.{domain}.{category}.0.0.0.0.{prim_id}.0"
        description = f"{name} ({source})"

        cursor.execute("""
            INSERT OR IGNORE INTO token_index (token_id, description, location)
            VALUES (?, ?, ?)
        """, (token_id, description, "primitives.db"))
        count += 1

    conn.close()
    return count


def index_language_db(cursor, lang_db: Path) -> int:
    """Index tokens from a language database."""
    conn = sqlite3.connect(lang_db)
    lang_cursor = conn.cursor()

    lang_cursor.execute("""
        SELECT token_id, lemma, pos
        FROM concepts
        WHERE token_id IS NOT NULL
    """)

    count = 0
    location = f"lang/{lang_db.name}"

    for row in lang_cursor.fetchall():
        token_id, lemma, pos = row
        description = f"{lemma}"
        if pos:
            description += f" ({pos})"

        cursor.execute("""
            INSERT OR IGNORE INTO token_index (token_id, description, location)
            VALUES (?, ?, ?)
        """, (token_id, description, location))
        count += 1

    conn.close()
    return count


def main():
    print("=" * 60)
    print("Building Master Token Index")
    print("=" * 60)

    # Create master index database
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if MASTER_INDEX_DB.exists():
        print(f"\nRemoving existing {MASTER_INDEX_DB.name}...")
        MASTER_INDEX_DB.unlink()

    conn = sqlite3.connect(MASTER_INDEX_DB)
    conn.executescript(MASTER_SCHEMA)
    cursor = conn.cursor()

    total = 0

    # Index primitives
    print("\nIndexing primitives...")
    count = index_primitives(cursor)
    print(f"  Added {count} primitive tokens")
    total += count
    conn.commit()

    # Index language databases
    if LANG_DIR.exists():
        for lang_db in sorted(LANG_DIR.glob("*.db")):
            print(f"\nIndexing {lang_db.name}...")
            count = index_language_db(cursor, lang_db)
            print(f"  Added {count} tokens")
            total += count
            conn.commit()

    # Summary
    print("\n" + "=" * 60)
    print("Master index complete!")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) FROM token_index")
    final_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT location) FROM token_index")
    location_count = cursor.fetchone()[0]

    print(f"\nTotal tokens indexed: {final_count}")
    print(f"Source databases: {location_count}")
    print(f"Database: {MASTER_INDEX_DB}")
    print(f"Size: {MASTER_INDEX_DB.stat().st_size / 1024:.1f} KB")

    # Show distribution
    print("\nTokens by location:")
    cursor.execute("""
        SELECT location, COUNT(*) as cnt
        FROM token_index
        GROUP BY location
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
