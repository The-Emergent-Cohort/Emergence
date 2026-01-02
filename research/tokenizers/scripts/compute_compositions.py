#!/usr/bin/env python3
"""
Compute concept compositions (progressive decomposition).

Links each concept to its immediate components (one level down).
Components may be other concepts or primitives.

Strategy:
1. Start with NSM primitives (already have compositions = [])
2. For each lexical concept, find component concepts by:
   - Gloss analysis (definition words)
   - WordNet hypernyms
   - Shared fingerprints across languages
3. Link to immediate components only (not full chain)

Run from: /usr/share/databases/scripts/
Requires: Imports to have been run first
"""

import re
import sqlite3
from pathlib import Path

# Path configuration - resolve symlinks to find real paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
PRIMITIVES_DB = DB_DIR / "primitives.db"

# Common stop words to skip in gloss analysis
STOP_WORDS = {
    "a", "an", "the", "of", "to", "in", "for", "on", "with", "at", "by",
    "from", "or", "and", "as", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "which", "who", "whom",
    "what", "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
    "one", "two", "three", "something", "someone", "anything", "anyone",
    "used", "using", "make", "made", "making",
}


def extract_gloss_words(gloss: str) -> list:
    """Extract meaningful words from a gloss."""
    if not gloss:
        return []

    # Remove parenthetical notes
    gloss = re.sub(r'\([^)]*\)', '', gloss)
    # Remove quotes
    gloss = re.sub(r'"[^"]*"', '', gloss)
    # Lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', gloss.lower())
    # Filter stop words and short words
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def load_primitives() -> dict:
    """Load primitives as potential components."""
    if not PRIMITIVES_DB.exists():
        return {}

    conn = sqlite3.connect(PRIMITIVES_DB)
    cursor = conn.cursor()

    # Get primitive canonical names
    cursor.execute("SELECT primitive_id, canonical_name FROM primitives")
    primitives = {row[1].lower(): row[0] for row in cursor.fetchall()}

    # Also get surface forms
    cursor.execute("""
        SELECT p.primitive_id, pf.surface_form
        FROM primitives p
        JOIN primitive_forms pf ON p.primitive_id = pf.primitive_id
    """)
    for row in cursor.fetchall():
        primitives[row[1].lower()] = row[0]

    conn.close()
    return primitives


def compute_for_language(lang_db: Path, primitives: dict):
    """Compute compositions for a language database."""
    print(f"\nProcessing {lang_db.name}...")

    conn = sqlite3.connect(lang_db)
    cursor = conn.cursor()

    # Get all concepts with glosses
    cursor.execute("""
        SELECT concept_id, lemma, gloss, fingerprint
        FROM concepts
        WHERE gloss IS NOT NULL AND gloss != ''
    """)
    concepts = cursor.fetchall()

    # Build lemma -> concept_id lookup for this language
    cursor.execute("SELECT concept_id, lemma FROM concepts")
    lemma_to_id = {}
    for row in cursor.fetchall():
        lemma = row[1].lower()
        if lemma not in lemma_to_id:
            lemma_to_id[lemma] = row[0]

    # Clear existing compositions
    cursor.execute("DELETE FROM compositions")

    composition_count = 0
    primitive_links = 0
    concept_links = 0

    for concept_id, lemma, gloss, fingerprint in concepts:
        gloss_words = extract_gloss_words(gloss)

        if not gloss_words:
            continue

        position = 0
        components_added = set()

        for word in gloss_words:
            # Skip self-reference
            if word == lemma.lower():
                continue

            component_id = None
            relation = "part"

            # First check primitives
            if word in primitives:
                # Store as negative ID to distinguish from concept IDs
                component_id = -primitives[word]
                relation = "primitive"
                primitive_links += 1
            # Then check other concepts in same language
            elif word in lemma_to_id:
                component_id = lemma_to_id[word]
                concept_links += 1

            if component_id and component_id not in components_added:
                cursor.execute("""
                    INSERT INTO compositions (concept_id, component_id, position, relation)
                    VALUES (?, ?, ?, ?)
                """, (concept_id, component_id, position, relation))
                components_added.add(component_id)
                position += 1
                composition_count += 1

                # Limit components per concept
                if position >= 10:
                    break

    conn.commit()
    conn.close()

    print(f"  Compositions: {composition_count}")
    print(f"  Primitive links: {primitive_links}")
    print(f"  Concept links: {concept_links}")

    return composition_count


def main():
    print("=" * 60)
    print("Computing Concept Compositions")
    print("=" * 60)

    # Load primitives
    primitives = load_primitives()
    print(f"\nLoaded {len(primitives)} primitive forms")

    if not LANG_DIR.exists():
        print(f"ERROR: Language directory not found: {LANG_DIR}")
        return 1

    # Process each language database
    total = 0
    for lang_db in sorted(LANG_DIR.glob("*.db")):
        count = compute_for_language(lang_db, primitives)
        total += count

    print("\n" + "=" * 60)
    print("Composition computation complete!")
    print("=" * 60)
    print(f"\nTotal compositions: {total}")

    return 0


if __name__ == "__main__":
    exit(main())
