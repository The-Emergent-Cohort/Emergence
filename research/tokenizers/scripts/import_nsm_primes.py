#!/usr/bin/env python3
"""
Import NSM (Natural Semantic Metalanguage) Primes

The 65 semantic primes from Anna Wierzbicka's NSM theory.
These are the irreducible core of meaning - present in all human languages.

Sources:
- Goddard & Wierzbicka (2014) "Words and Meanings"
- NSM Homepage: https://nsm-approach.net/
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

# NSM Semantic Primes organized by category
# Format: (canonical_name, domain, category, description, example_forms)
NSM_PRIMES = [
    # === SUBSTANTIVES ===
    ("I", 3, 4, "First person singular", {"eng": "I", "fra": "je", "deu": "ich", "spa": "yo", "jpn": "私", "zho": "我"}),
    ("YOU", 3, 4, "Second person singular", {"eng": "you", "fra": "tu", "deu": "du", "spa": "tú", "jpn": "あなた", "zho": "你"}),
    ("SOMEONE", 4, 2, "Indefinite person", {"eng": "someone", "fra": "quelqu'un", "deu": "jemand", "spa": "alguien"}),
    ("SOMETHING", 5, 5, "Indefinite thing", {"eng": "something", "fra": "quelque chose", "deu": "etwas", "spa": "algo"}),
    ("PEOPLE", 4, 3, "Human collective", {"eng": "people", "fra": "gens", "deu": "Leute", "spa": "gente"}),
    ("BODY", 6, 2, "Physical body", {"eng": "body", "fra": "corps", "deu": "Körper", "spa": "cuerpo"}),

    # === RELATIONAL SUBSTANTIVES ===
    ("KIND", 5, 5, "Type/category", {"eng": "kind", "fra": "sorte", "deu": "Art", "spa": "tipo"}),
    ("PART", 5, 6, "Part of whole", {"eng": "part", "fra": "partie", "deu": "Teil", "spa": "parte"}),

    # === DETERMINERS ===
    ("THIS", 5, 5, "Proximal demonstrative", {"eng": "this", "fra": "ce", "deu": "dies", "spa": "esto"}),
    ("THE_SAME", 5, 3, "Identity", {"eng": "the same", "fra": "le même", "deu": "derselbe", "spa": "el mismo"}),
    ("OTHER", 5, 3, "Difference/alterity", {"eng": "other", "fra": "autre", "deu": "andere", "spa": "otro"}),

    # === QUANTIFIERS ===
    ("ONE", 5, 1, "Singular quantity", {"eng": "one", "fra": "un", "deu": "ein", "spa": "uno"}),
    ("TWO", 5, 1, "Dual quantity", {"eng": "two", "fra": "deux", "deu": "zwei", "spa": "dos"}),
    ("SOME", 5, 1, "Partial quantity", {"eng": "some", "fra": "quelques", "deu": "einige", "spa": "algunos"}),
    ("ALL", 5, 1, "Total quantity", {"eng": "all", "fra": "tout", "deu": "alle", "spa": "todo"}),
    ("MUCH_MANY", 5, 1, "Large quantity", {"eng": "much/many", "fra": "beaucoup", "deu": "viel", "spa": "mucho"}),

    # === EVALUATORS ===
    ("GOOD", 9, 1, "Positive evaluation", {"eng": "good", "fra": "bon", "deu": "gut", "spa": "bueno"}),
    ("BAD", 9, 1, "Negative evaluation", {"eng": "bad", "fra": "mauvais", "deu": "schlecht", "spa": "malo"}),

    # === DESCRIPTORS ===
    ("BIG", 1, 6, "Large size", {"eng": "big", "fra": "grand", "deu": "groß", "spa": "grande"}),
    ("SMALL", 1, 6, "Small size", {"eng": "small", "fra": "petit", "deu": "klein", "spa": "pequeño"}),

    # === MENTAL PREDICATES ===
    ("THINK", 3, 2, "Cognitive process", {"eng": "think", "fra": "penser", "deu": "denken", "spa": "pensar"}),
    ("KNOW", 3, 2, "Epistemic state", {"eng": "know", "fra": "savoir", "deu": "wissen", "spa": "saber"}),
    ("WANT", 3, 4, "Volition", {"eng": "want", "fra": "vouloir", "deu": "wollen", "spa": "querer"}),
    ("DONT_WANT", 3, 4, "Negative volition", {"eng": "don't want", "fra": "ne pas vouloir", "deu": "nicht wollen", "spa": "no querer"}),
    ("FEEL", 3, 3, "Emotional state", {"eng": "feel", "fra": "sentir", "deu": "fühlen", "spa": "sentir"}),
    ("SEE", 3, 1, "Visual perception", {"eng": "see", "fra": "voir", "deu": "sehen", "spa": "ver"}),
    ("HEAR", 3, 1, "Auditory perception", {"eng": "hear", "fra": "entendre", "deu": "hören", "spa": "oír"}),

    # === SPEECH ===
    ("SAY", 4, 1, "Speech act", {"eng": "say", "fra": "dire", "deu": "sagen", "spa": "decir"}),
    ("WORDS", 4, 1, "Linguistic units", {"eng": "words", "fra": "mots", "deu": "Wörter", "spa": "palabras"}),
    ("TRUE", 5, 4, "Truth value", {"eng": "true", "fra": "vrai", "deu": "wahr", "spa": "verdadero"}),

    # === ACTIONS, EVENTS, MOVEMENT ===
    ("DO", 1, 4, "Action/agency", {"eng": "do", "fra": "faire", "deu": "tun", "spa": "hacer"}),
    ("HAPPEN", 2, 4, "Event occurrence", {"eng": "happen", "fra": "arriver", "deu": "geschehen", "spa": "pasar"}),
    ("MOVE", 1, 1, "Motion", {"eng": "move", "fra": "bouger", "deu": "bewegen", "spa": "mover"}),

    # === EXISTENCE, POSSESSION ===
    ("BE", 1, 5, "Existence/state", {"eng": "be", "fra": "être", "deu": "sein", "spa": "ser/estar"}),
    ("THERE_IS", 1, 5, "Existential", {"eng": "there is", "fra": "il y a", "deu": "es gibt", "spa": "hay"}),
    ("BE_SOMEONES", 4, 4, "Possession", {"eng": "be someone's", "fra": "être à", "deu": "gehören", "spa": "ser de"}),

    # === LIFE AND DEATH ===
    ("LIVE", 6, 1, "Being alive", {"eng": "live", "fra": "vivre", "deu": "leben", "spa": "vivir"}),
    ("DIE", 6, 1, "Death", {"eng": "die", "fra": "mourir", "deu": "sterben", "spa": "morir"}),

    # === TIME ===
    ("WHEN", 2, 1, "Temporal question", {"eng": "when", "fra": "quand", "deu": "wann", "spa": "cuándo"}),
    ("NOW", 2, 1, "Present time", {"eng": "now", "fra": "maintenant", "deu": "jetzt", "spa": "ahora"}),
    ("BEFORE", 2, 1, "Prior time", {"eng": "before", "fra": "avant", "deu": "vorher", "spa": "antes"}),
    ("AFTER", 2, 1, "Posterior time", {"eng": "after", "fra": "après", "deu": "nachher", "spa": "después"}),
    ("A_LONG_TIME", 2, 2, "Extended duration", {"eng": "a long time", "fra": "longtemps", "deu": "lange", "spa": "mucho tiempo"}),
    ("A_SHORT_TIME", 2, 2, "Brief duration", {"eng": "a short time", "fra": "peu de temps", "deu": "kurz", "spa": "poco tiempo"}),
    ("FOR_SOME_TIME", 2, 2, "Indefinite duration", {"eng": "for some time", "fra": "pendant quelque temps", "deu": "eine Weile", "spa": "por un tiempo"}),
    ("MOMENT", 2, 2, "Instant", {"eng": "moment", "fra": "moment", "deu": "Moment", "spa": "momento"}),

    # === SPACE ===
    ("WHERE", 1, 2, "Spatial question", {"eng": "where", "fra": "où", "deu": "wo", "spa": "dónde"}),
    ("HERE", 1, 2, "Proximal location", {"eng": "here", "fra": "ici", "deu": "hier", "spa": "aquí"}),
    ("ABOVE", 1, 2, "Superior position", {"eng": "above", "fra": "au-dessus", "deu": "über", "spa": "arriba"}),
    ("BELOW", 1, 2, "Inferior position", {"eng": "below", "fra": "en-dessous", "deu": "unter", "spa": "abajo"}),
    ("FAR", 1, 2, "Distant", {"eng": "far", "fra": "loin", "deu": "weit", "spa": "lejos"}),
    ("NEAR", 1, 2, "Proximate", {"eng": "near", "fra": "près", "deu": "nah", "spa": "cerca"}),
    ("SIDE", 1, 2, "Lateral position", {"eng": "side", "fra": "côté", "deu": "Seite", "spa": "lado"}),
    ("INSIDE", 1, 2, "Interior", {"eng": "inside", "fra": "dedans", "deu": "innen", "spa": "dentro"}),
    ("TOUCH", 1, 3, "Physical contact", {"eng": "touch", "fra": "toucher", "deu": "berühren", "spa": "tocar"}),

    # === LOGICAL CONCEPTS ===
    ("NOT", 5, 4, "Negation", {"eng": "not", "fra": "ne...pas", "deu": "nicht", "spa": "no"}),
    ("MAYBE", 5, 4, "Possibility", {"eng": "maybe", "fra": "peut-être", "deu": "vielleicht", "spa": "quizás"}),
    ("CAN", 5, 4, "Ability/possibility", {"eng": "can", "fra": "pouvoir", "deu": "können", "spa": "poder"}),
    ("BECAUSE", 5, 4, "Causation", {"eng": "because", "fra": "parce que", "deu": "weil", "spa": "porque"}),
    ("IF", 5, 4, "Conditional", {"eng": "if", "fra": "si", "deu": "wenn", "spa": "si"}),

    # === INTENSIFIER, AUGMENTOR ===
    ("VERY", 5, 2, "Intensifier", {"eng": "very", "fra": "très", "deu": "sehr", "spa": "muy"}),
    ("MORE", 5, 3, "Comparative", {"eng": "more", "fra": "plus", "deu": "mehr", "spa": "más"}),

    # === SIMILARITY ===
    ("LIKE", 5, 3, "Similarity", {"eng": "like", "fra": "comme", "deu": "wie", "spa": "como"}),
]

# Language code mapping (matching language_registry)
LANG_CODES = {
    "eng": 100,
    "fra": 450,
    "deu": 400,
    "spa": 200,
    "jpn": 350,
    "zho": 150,
    "ara": 250,
    "hin": 300,
    "por": 500,
    "rus": 550,
}


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create primitives database with schema."""
    conn = sqlite3.connect(db_path)

    # Read and execute schema
    schema_path = db_path.parent / "SCHEMA-primitives.sql"
    if schema_path.exists():
        with open(schema_path) as f:
            conn.executescript(f.read())

    return conn


def import_nsm_primes(conn: sqlite3.Connection) -> int:
    """Import NSM primes into the primitives table."""
    cursor = conn.cursor()

    primes_added = 0
    forms_added = 0

    for i, (name, domain, category, description, forms) in enumerate(NSM_PRIMES, start=1):
        # Insert primitive
        cursor.execute("""
            INSERT OR IGNORE INTO primitives
            (primitive_id, canonical_name, source, domain, category, description)
            VALUES (?, ?, 'nsm', ?, ?, ?)
        """, (i, name, domain, category, description))

        if cursor.rowcount > 0:
            primes_added += 1

        # Insert language forms
        for lang_iso, surface_form in forms.items():
            lang_code = LANG_CODES.get(lang_iso, 9999)
            cursor.execute("""
                INSERT OR IGNORE INTO primitive_forms
                (primitive_id, lang_code, surface_form, source)
                VALUES (?, ?, ?, 'nsm_research')
            """, (i, lang_code, surface_form))

            if cursor.rowcount > 0:
                forms_added += 1

    # Record import metadata
    cursor.execute("""
        INSERT INTO import_metadata (source, record_count, notes)
        VALUES ('nsm_primes', ?, ?)
    """, (primes_added, f"65 NSM semantic primes, {forms_added} language forms"))

    conn.commit()
    return primes_added


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import NSM semantic primes")
    parser.add_argument("--db", type=Path,
                       default=Path(__file__).parent.parent / "db" / "primitives.db",
                       help="Database path")
    parser.add_argument("--schema", type=Path,
                       default=Path(__file__).parent.parent / "db" / "SCHEMA-primitives.sql",
                       help="Schema file path")
    args = parser.parse_args()

    # Ensure db directory exists
    args.db.parent.mkdir(parents=True, exist_ok=True)

    # Create database with schema
    conn = sqlite3.connect(args.db)

    if args.schema.exists():
        print(f"Applying schema from {args.schema}")
        with open(args.schema) as f:
            conn.executescript(f.read())

    # Import primes
    print("Importing NSM semantic primes...")
    count = import_nsm_primes(conn)

    # Summary
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM primitives WHERE source = 'nsm'")
    total_primes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM primitive_forms")
    total_forms = cursor.fetchone()[0]

    print(f"\nImport complete:")
    print(f"  Primitives: {total_primes}")
    print(f"  Language forms: {total_forms}")
    print(f"  Database: {args.db}")

    conn.close()


if __name__ == "__main__":
    main()
