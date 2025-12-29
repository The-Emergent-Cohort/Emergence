#!/usr/bin/env python3
"""Initialize core morphemes table from morphemeML.pdf
Just morpheme + category for relational groupings"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tokenizer.db"

# Morphemes with categories for relational grouping
# Format: (morpheme, category)
MORPHEMES = [
    # Quantity
    ("ONE", "quantity"),
    ("TWO", "quantity"),
    ("THREE", "quantity"),
    ("MANY", "quantity"),

    # Spatial
    ("ACROSS", "spatial"),
    ("UNDER", "spatial"),
    ("BETWEEN", "spatial"),
    ("WITHIN", "spatial"),
    ("OUT_FROM", "spatial"),

    # Temporal
    ("BEFORE", "temporal"),
    ("AFTER", "temporal"),
    ("AGAIN", "temporal"),
    ("TIME", "temporal"),

    # Quality/modifier
    ("GOOD", "quality"),
    ("BAD", "quality"),
    ("SMALL", "quality"),
    ("LARGE", "quality"),
    ("SAME", "quality"),
    ("COMPARATIVE", "quality"),
    ("QUALITY_OF", "quality"),
    ("CAPABILITY", "quality"),
    ("WITHOUT", "quality"),
    ("MANNER", "quality"),

    # Negation/opposition
    ("NEGATION", "negation"),
    ("AGAINST", "negation"),
    ("REMOVAL", "negation"),

    # Relation
    ("TOGETHER", "relation"),
    ("COMPLETELY", "relation"),

    # State/condition
    ("STATE_OF", "state"),
    ("LIFE", "state"),
    ("DIE", "state"),
    ("SUFFERING", "state"),
    ("FEAR", "state"),

    # Agent/role
    ("AGENT_OF", "agent"),
    ("SKILL_OF", "agent"),
    ("PRACTICE_OF", "agent"),
    ("STUDY_OF", "agent"),

    # Action
    ("CAUSATIVE", "action"),
    ("MAKE", "action"),
    ("CARRY", "action"),
    ("PULL", "action"),
    ("PUSH", "action"),
    ("THROW", "action"),
    ("TURN", "action"),
    ("SEND", "action"),
    ("TAKE", "action"),
    ("BREAK", "action"),
    ("BUILD", "action"),
    ("JOIN", "action"),
    ("BEND", "action"),
    ("STEP", "action"),
    ("KILL", "action"),
    ("FORCE", "action"),

    # Perception
    ("SEE", "perception"),
    ("SOUND", "perception"),
    ("LIGHT", "perception"),

    # Mental
    ("MIND", "mental"),
    ("BELIEVE", "mental"),
    ("SPEAK", "mental"),
    ("WRITE", "mental"),

    # Physical/element
    ("WATER", "element"),
    ("EARTH", "element"),
    ("HEAT", "element"),
    ("STAR", "element"),
    ("ANIMAL", "element"),

    # Abstract
    ("SELF", "abstract"),
    ("ALL", "abstract"),
    ("DISTANT", "abstract"),
    ("MEASURE", "abstract"),
    ("SHAPE", "abstract"),
    ("RESULT_OF", "abstract"),
    ("PLACE_FOR", "abstract"),
    ("BREATHE", "abstract"),
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS morphemes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            morpheme TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_morphemes_category ON morphemes(category)")

    conn.commit()
    return conn


def main():
    print(f"Creating database at {DB_PATH}")
    conn = init_db()
    cur = conn.cursor()

    for morpheme, category in MORPHEMES:
        cur.execute(
            "INSERT OR IGNORE INTO morphemes (morpheme, category) VALUES (?, ?)",
            (morpheme, category)
        )

    conn.commit()

    # Summary
    cur.execute("SELECT category, COUNT(*) FROM morphemes GROUP BY category ORDER BY category")
    print("\nMorphemes by category:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cur.execute("SELECT COUNT(*) FROM morphemes")
    print(f"\nTotal: {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
