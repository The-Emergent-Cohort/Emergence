#!/usr/bin/env python3
"""
Import NSM (Natural Semantic Metalanguage) semantic primes.

65 universal semantic primes from Wierzbicka's NSM theory.
These are irreducible meaning atoms found across all languages.

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py to have been run first
"""

import sqlite3
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
PRIMITIVES_DB = DB_DIR / "primitives.db"

# NSM Semantic Primes organized by domain and category
# Format: (name, domain, category, description, examples)
NSM_PRIMES = [
    # === SUBSTANTIVES ===
    ("I", 3, 14, "First person singular", "I, me"),
    ("YOU", 3, 14, "Second person", "you"),
    ("SOMEONE", 3, 14, "Indefinite person", "someone, person"),
    ("SOMETHING", 5, 33, "Indefinite thing", "something, thing"),
    ("PEOPLE", 4, 23, "Collective persons", "people"),
    ("BODY", 6, 5, "Physical body", "body"),

    # === RELATIONAL SUBSTANTIVES ===
    ("KIND", 5, 33, "Type or category", "kind, sort, type"),
    ("PART", 5, 34, "Component", "part"),

    # === DETERMINERS ===
    ("THIS", 5, 33, "Proximal demonstrative", "this"),
    ("THE_SAME", 5, 31, "Identity", "the same"),
    ("OTHER", 5, 31, "Difference", "other, another, else"),

    # === QUANTIFIERS ===
    ("ONE", 5, 29, "Singular quantity", "one"),
    ("TWO", 5, 29, "Dual quantity", "two"),
    ("SOME", 5, 29, "Partial quantity", "some"),
    ("ALL", 5, 29, "Total quantity", "all, every"),
    ("MUCH_MANY", 5, 29, "Large quantity", "much, many"),

    # === EVALUATORS ===
    ("GOOD", 9, 35, "Positive evaluation", "good"),
    ("BAD", 9, 35, "Negative evaluation", "bad"),

    # === DESCRIPTORS ===
    ("BIG", 1, 6, "Large size", "big, large"),
    ("SMALL", 1, 6, "Small size", "small, little"),

    # === MENTAL PREDICATES ===
    ("THINK", 3, 14, "Cognitive process", "think"),
    ("KNOW", 3, 19, "Knowledge state", "know"),
    ("WANT", 3, 16, "Desire", "want"),
    ("DONT_WANT", 3, 16, "Negative desire", "don't want, refuse"),
    ("FEEL", 3, 15, "Emotional experience", "feel"),
    ("SEE", 3, 13, "Visual perception", "see"),
    ("HEAR", 3, 13, "Auditory perception", "hear"),

    # === SPEECH ===
    ("SAY", 4, 21, "Verbal communication", "say"),
    ("WORDS", 4, 21, "Linguistic units", "words"),
    ("TRUE", 5, 32, "Truth value", "true"),

    # === ACTIONS, EVENTS, MOVEMENT ===
    ("DO", 1, 4, "Action", "do"),
    ("HAPPEN", 1, 4, "Event occurrence", "happen"),
    ("MOVE", 1, 1, "Physical movement", "move"),
    ("TOUCH", 1, 3, "Physical contact", "touch"),

    # === LOCATION, EXISTENCE, SPECIFICATION ===
    ("BE_SOMEWHERE", 1, 2, "Location", "be somewhere"),
    ("THERE_IS", 1, 5, "Existence", "there is, exist"),
    ("BE_SOMEONE_SOMETHING", 5, 33, "Identity/specification", "be"),

    # === POSSESSION ===
    ("HAVE", 4, 24, "Possession", "have"),

    # === LIFE AND DEATH ===
    ("LIVE", 6, 5, "Being alive", "live, alive"),
    ("DIE", 6, 5, "Death", "die"),

    # === TIME ===
    ("WHEN", 2, 12, "Temporal reference", "when, time"),
    ("NOW", 2, 12, "Present time", "now"),
    ("BEFORE", 2, 9, "Temporal precedence", "before"),
    ("AFTER", 2, 9, "Temporal succession", "after"),
    ("A_LONG_TIME", 2, 10, "Extended duration", "a long time"),
    ("A_SHORT_TIME", 2, 10, "Brief duration", "a short time"),
    ("FOR_SOME_TIME", 2, 10, "Duration", "for some time"),
    ("MOMENT", 2, 10, "Instant", "moment"),

    # === SPACE ===
    ("WHERE", 1, 2, "Spatial reference", "where, place"),
    ("HERE", 1, 2, "Proximal location", "here"),
    ("ABOVE", 1, 2, "Vertical superior", "above"),
    ("BELOW", 1, 2, "Vertical inferior", "below"),
    ("FAR", 1, 2, "Distant", "far"),
    ("NEAR", 1, 2, "Proximate", "near"),
    ("SIDE", 1, 2, "Lateral", "side"),
    ("INSIDE", 1, 2, "Interior", "inside"),

    # === LOGICAL CONCEPTS ===
    ("NOT", 5, 32, "Negation", "not"),
    ("MAYBE", 5, 32, "Possibility", "maybe, perhaps"),
    ("CAN", 5, 32, "Ability/possibility", "can"),
    ("BECAUSE", 5, 32, "Causation", "because"),
    ("IF", 5, 32, "Condition", "if"),

    # === AUGMENTOR ===
    ("VERY", 5, 30, "Intensifier", "very"),
    ("MORE", 5, 30, "Comparative", "more"),

    # === SIMILARITY ===
    ("LIKE", 5, 31, "Similarity", "like, as"),
]

# Cross-linguistic forms for major languages
# Format: (prime_name, lang_genomic, surface_form)
# Lang genomic codes: FF.SS.LLL.DD
#   1.8.127.0 = Indo-European.Germanic.English.core
#   1.8.200.0 = Indo-European.Germanic.German.core
#   1.12.100.0 = Indo-European.Romance.French.core
#   1.12.150.0 = Indo-European.Romance.Spanish.core
#   9.0.100.0 = Japonic.-.Japanese.core
#   2.1.100.0 = Sino-Tibetan.Sinitic.Mandarin.core

PRIME_FORMS = [
    # English (1.8.127.0) - generate from canonical names
    *[(p[0], "1.8.127.0", p[0].lower().replace("_", " ")) for p in NSM_PRIMES],

    # German (1.8.200.0)
    ("I", "1.8.200.0", "ich"),
    ("YOU", "1.8.200.0", "du"),
    ("SOMEONE", "1.8.200.0", "jemand"),
    ("SOMETHING", "1.8.200.0", "etwas"),
    ("PEOPLE", "1.8.200.0", "Leute"),
    ("THINK", "1.8.200.0", "denken"),
    ("KNOW", "1.8.200.0", "wissen"),
    ("WANT", "1.8.200.0", "wollen"),
    ("FEEL", "1.8.200.0", "fühlen"),
    ("SEE", "1.8.200.0", "sehen"),
    ("HEAR", "1.8.200.0", "hören"),
    ("SAY", "1.8.200.0", "sagen"),
    ("DO", "1.8.200.0", "tun"),
    ("GOOD", "1.8.200.0", "gut"),
    ("BAD", "1.8.200.0", "schlecht"),
    ("BIG", "1.8.200.0", "groß"),
    ("SMALL", "1.8.200.0", "klein"),
    ("NOT", "1.8.200.0", "nicht"),
    ("NOW", "1.8.200.0", "jetzt"),
    ("HERE", "1.8.200.0", "hier"),
    ("MOVE", "1.8.200.0", "bewegen"),
    ("LIVE", "1.8.200.0", "leben"),
    ("DIE", "1.8.200.0", "sterben"),
    ("TRUE", "1.8.200.0", "wahr"),

    # French (1.12.100.0)
    ("I", "1.12.100.0", "je"),
    ("YOU", "1.12.100.0", "tu"),
    ("SOMEONE", "1.12.100.0", "quelqu'un"),
    ("SOMETHING", "1.12.100.0", "quelque chose"),
    ("PEOPLE", "1.12.100.0", "gens"),
    ("THINK", "1.12.100.0", "penser"),
    ("KNOW", "1.12.100.0", "savoir"),
    ("WANT", "1.12.100.0", "vouloir"),
    ("FEEL", "1.12.100.0", "sentir"),
    ("SEE", "1.12.100.0", "voir"),
    ("HEAR", "1.12.100.0", "entendre"),
    ("SAY", "1.12.100.0", "dire"),
    ("DO", "1.12.100.0", "faire"),
    ("GOOD", "1.12.100.0", "bon"),
    ("BAD", "1.12.100.0", "mauvais"),
    ("BIG", "1.12.100.0", "grand"),
    ("SMALL", "1.12.100.0", "petit"),
    ("NOT", "1.12.100.0", "ne pas"),
    ("NOW", "1.12.100.0", "maintenant"),
    ("HERE", "1.12.100.0", "ici"),
    ("MOVE", "1.12.100.0", "bouger"),
    ("LIVE", "1.12.100.0", "vivre"),
    ("DIE", "1.12.100.0", "mourir"),
    ("TRUE", "1.12.100.0", "vrai"),

    # Spanish (1.12.150.0)
    ("I", "1.12.150.0", "yo"),
    ("YOU", "1.12.150.0", "tú"),
    ("SOMEONE", "1.12.150.0", "alguien"),
    ("SOMETHING", "1.12.150.0", "algo"),
    ("PEOPLE", "1.12.150.0", "gente"),
    ("THINK", "1.12.150.0", "pensar"),
    ("KNOW", "1.12.150.0", "saber"),
    ("WANT", "1.12.150.0", "querer"),
    ("FEEL", "1.12.150.0", "sentir"),
    ("SEE", "1.12.150.0", "ver"),
    ("HEAR", "1.12.150.0", "oír"),
    ("SAY", "1.12.150.0", "decir"),
    ("DO", "1.12.150.0", "hacer"),
    ("GOOD", "1.12.150.0", "bueno"),
    ("BAD", "1.12.150.0", "malo"),
    ("BIG", "1.12.150.0", "grande"),
    ("SMALL", "1.12.150.0", "pequeño"),
    ("NOT", "1.12.150.0", "no"),
    ("NOW", "1.12.150.0", "ahora"),
    ("HERE", "1.12.150.0", "aquí"),
    ("MOVE", "1.12.150.0", "mover"),
    ("LIVE", "1.12.150.0", "vivir"),
    ("DIE", "1.12.150.0", "morir"),
    ("TRUE", "1.12.150.0", "verdadero"),

    # Japanese (9.0.100.0)
    ("I", "9.0.100.0", "私"),
    ("YOU", "9.0.100.0", "あなた"),
    ("SOMEONE", "9.0.100.0", "誰か"),
    ("SOMETHING", "9.0.100.0", "何か"),
    ("PEOPLE", "9.0.100.0", "人々"),
    ("THINK", "9.0.100.0", "思う"),
    ("KNOW", "9.0.100.0", "知る"),
    ("WANT", "9.0.100.0", "欲しい"),
    ("SEE", "9.0.100.0", "見る"),
    ("HEAR", "9.0.100.0", "聞く"),
    ("SAY", "9.0.100.0", "言う"),
    ("DO", "9.0.100.0", "する"),
    ("GOOD", "9.0.100.0", "良い"),
    ("BAD", "9.0.100.0", "悪い"),
    ("BIG", "9.0.100.0", "大きい"),
    ("SMALL", "9.0.100.0", "小さい"),
    ("NOT", "9.0.100.0", "ない"),
    ("NOW", "9.0.100.0", "今"),
    ("HERE", "9.0.100.0", "ここ"),

    # Mandarin (2.1.100.0)
    ("I", "2.1.100.0", "我"),
    ("YOU", "2.1.100.0", "你"),
    ("SOMEONE", "2.1.100.0", "有人"),
    ("SOMETHING", "2.1.100.0", "东西"),
    ("PEOPLE", "2.1.100.0", "人"),
    ("THINK", "2.1.100.0", "想"),
    ("KNOW", "2.1.100.0", "知道"),
    ("WANT", "2.1.100.0", "要"),
    ("SEE", "2.1.100.0", "看"),
    ("HEAR", "2.1.100.0", "听"),
    ("SAY", "2.1.100.0", "说"),
    ("DO", "2.1.100.0", "做"),
    ("GOOD", "2.1.100.0", "好"),
    ("BAD", "2.1.100.0", "坏"),
    ("BIG", "2.1.100.0", "大"),
    ("SMALL", "2.1.100.0", "小"),
    ("NOT", "2.1.100.0", "不"),
    ("NOW", "2.1.100.0", "现在"),
    ("HERE", "2.1.100.0", "这里"),
]


def generate_token_id(primitive_id: int, domain: int, category: int) -> str:
    """
    Generate genomic token ID for a primitive.

    Format: A.D.C.FF.SS.LLL.DD.FP.COL

    For primitives:
    - A = 1 (abstraction level 1 = primitive)
    - D = domain
    - C = category
    - FF.SS.LLL.DD = 0.0.0.0 (universal)
    - FP = primitive_id (unique identifier)
    - COL = 0 (no collision for primitives)
    """
    return f"1.{domain}.{category}.0.0.0.0.{primitive_id}.0"


def main():
    print("=" * 60)
    print("Importing NSM Semantic Primes")
    print("=" * 60)

    if not PRIMITIVES_DB.exists():
        print(f"ERROR: {PRIMITIVES_DB} not found.")
        print("Run init_schemas.py first.")
        return 1

    conn = sqlite3.connect(PRIMITIVES_DB)
    cursor = conn.cursor()

    # Clear existing NSM primes
    cursor.execute("DELETE FROM primitive_forms WHERE primitive_id IN (SELECT primitive_id FROM primitives WHERE source = 'nsm')")
    cursor.execute("DELETE FROM primitives WHERE source = 'nsm'")
    cursor.execute("DELETE FROM token_index WHERE location = 'primitives' AND token_id LIKE '1.%'")

    print(f"\nInserting {len(NSM_PRIMES)} NSM primes...")

    # Insert primes
    for i, (name, domain, category, description, examples) in enumerate(NSM_PRIMES, 1):
        cursor.execute("""
            INSERT INTO primitives (primitive_id, canonical_name, source, domain, category, description, examples)
            VALUES (?, ?, 'nsm', ?, ?, ?, ?)
        """, (i, name, domain, category, description, examples))

        # Generate and insert token_id
        token_id = generate_token_id(i, domain, category)
        cursor.execute("""
            INSERT INTO token_index (token_id, description, location)
            VALUES (?, ?, 'primitives')
        """, (token_id, f"NSM:{name}"))

    print(f"  ✓ Inserted {len(NSM_PRIMES)} primitives")

    # Build name->id mapping
    cursor.execute("SELECT canonical_name, primitive_id FROM primitives WHERE source = 'nsm'")
    name_to_id = {row[0]: row[1] for row in cursor.fetchall()}

    # Insert forms
    print(f"\nInserting {len(PRIME_FORMS)} surface forms...")
    forms_inserted = 0
    for prime_name, lang_genomic, surface_form in PRIME_FORMS:
        if prime_name in name_to_id:
            cursor.execute("""
                INSERT INTO primitive_forms (primitive_id, lang_genomic, surface_form, source)
                VALUES (?, ?, ?, 'nsm')
            """, (name_to_id[prime_name], lang_genomic, surface_form))
            forms_inserted += 1

    print(f"  ✓ Inserted {forms_inserted} surface forms")

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("NSM import complete!")
    print("=" * 60)
    print(f"\nPrimitives: {len(NSM_PRIMES)}")
    print(f"Surface forms: {forms_inserted}")
    print(f"\nDatabase: {PRIMITIVES_DB}")

    return 0


if __name__ == "__main__":
    exit(main())
