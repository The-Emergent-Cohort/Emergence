#!/usr/bin/env python3
"""Initialize concepts and morpheme_forms tables from morphemeML.pdf"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tokenizer.db"

# CONCEPTS - the abstract meanings (language-independent)
# Format: (concept_name, description)
CONCEPTS = [
    # From suffixes
    ("AGENT_OF", "one who does or is associated with"),
    ("COMPARATIVE", "more of a quality"),
    ("MANNER", "in a way that is"),
    ("CAPABILITY", "capable of, worthy of"),
    ("STATE_OF", "condition or state of being"),
    ("QUALITY_OF", "having the quality of, relating to"),
    ("WITHOUT", "lacking, absence of"),
    ("STUDY_OF", "study or science of"),
    ("SKILL_OF", "art, skill, condition, rank of"),
    ("RESULT_OF", "result of an action"),
    ("PLACE_FOR", "place for, collection of"),
    ("CAUSATIVE", "to make, to cause to be"),
    ("PRACTICE_OF", "practice, belief, system"),

    # From prefixes
    ("REMOVAL", "from, reduce, remove"),
    ("NEGATION", "not, opposite of"),
    ("ACROSS", "across, through, over"),
    ("OUT_FROM", "out, from"),
    ("ONE", "one, single"),
    ("TWO", "two, pair"),
    ("THREE", "three"),
    ("MANY", "many, much, multiple"),
    ("BEFORE", "before, forward"),
    ("AFTER", "after"),
    ("BAD", "bad, wrong, evil"),
    ("GOOD", "good, well"),
    ("UNDER", "under, beneath"),
    ("AGAIN", "back, again, repetition"),
    ("BETWEEN", "among, between"),
    ("WITHIN", "within, inside"),
    ("TOGETHER", "together, with"),
    ("COMPLETELY", "to completion"),
    ("AGAINST", "against, opposite"),

    # From Greek roots
    ("STAR", "stars, heavens, celestial"),
    ("LIFE", "life, living"),
    ("EARTH", "earth, ground, rocks"),
    ("HEAT", "heat, warmth, temperature"),
    ("SELF", "self, automatic"),
    ("SAME", "same, alike, similar"),
    ("WATER", "water, liquid"),
    ("SMALL", "small, tiny"),
    ("LARGE", "large, great"),
    ("SOUND", "sound, speech, voice"),
    ("SEE", "to see, observe, watch"),
    ("WRITE", "to write, written"),
    ("LIGHT", "light, illumination"),
    ("DISTANT", "distant, far, remote"),
    ("MEASURE", "to measure, measurement"),
    ("SUFFERING", "suffering, disease, feeling"),
    ("MIND", "mind, mental, soul"),
    ("ALL", "all, whole, every"),
    ("ANIMAL", "animal, living creature"),
    ("TIME", "time, duration"),
    ("FEAR", "fear, intense dislike"),

    # From Latin roots
    ("CARRY", "to carry, transport"),
    ("SHAPE", "to shape, form"),
    ("PULL", "to pull, draw, drag"),
    ("BREAK", "to break, burst"),
    ("BUILD", "to build, construct"),
    ("SPEAK", "to tell, say, speak"),
    ("BEND", "to bend, flex"),
    ("BELIEVE", "to believe, trust"),
    ("PUSH", "to drive, push, compel"),
    ("MAKE", "to make, do, create"),
    ("THROW", "to throw, cast"),
    ("TURN", "to turn, change"),
    ("SEND", "to send, transmit"),
    ("DIE", "to die, death"),
    ("JOIN", "to join, connect"),
    ("KILL", "to kill, cut"),
    ("FORCE", "to force, press, squeeze"),
    ("BREATHE", "to breathe, spirit"),
    ("STEP", "to step, walk, go"),
    ("TAKE", "to take, seize, receive"),
]

# MORPHEME FORMS - surface forms that express concepts
# Format: (concept_name, morpheme, morpheme_type, origin)
MORPHEME_FORMS = [
    # AGENT_OF
    ("AGENT_OF", "-er", "suffix", None),
    ("AGENT_OF", "-ist", "suffix", None),
    ("AGENT_OF", "-ian", "suffix", None),
    ("AGENT_OF", "-or", "suffix", None),
    ("AGENT_OF", "-eer", "suffix", None),
    ("AGENT_OF", "-ent", "suffix", None),
    ("AGENT_OF", "-ant", "suffix", None),

    # COMPARATIVE
    ("COMPARATIVE", "-er", "suffix", None),

    # MANNER
    ("MANNER", "-ly", "suffix", None),

    # CAPABILITY
    ("CAPABILITY", "-able", "suffix", None),
    ("CAPABILITY", "-ible", "suffix", None),

    # STATE_OF
    ("STATE_OF", "-hood", "suffix", None),
    ("STATE_OF", "-ness", "suffix", None),
    ("STATE_OF", "-ment", "suffix", None),
    ("STATE_OF", "-ure", "suffix", None),
    ("STATE_OF", "-ion", "suffix", None),
    ("STATE_OF", "-ation", "suffix", None),
    ("STATE_OF", "-ance", "suffix", None),
    ("STATE_OF", "-ence", "suffix", None),
    ("STATE_OF", "-ity", "suffix", None),
    ("STATE_OF", "-tude", "suffix", None),

    # QUALITY_OF
    ("QUALITY_OF", "-ful", "suffix", None),
    ("QUALITY_OF", "-ous", "suffix", None),
    ("QUALITY_OF", "-ive", "suffix", None),
    ("QUALITY_OF", "-ic", "suffix", None),
    ("QUALITY_OF", "-al", "suffix", None),
    ("QUALITY_OF", "-ish", "suffix", None),
    ("QUALITY_OF", "-ary", "suffix", None),

    # WITHOUT
    ("WITHOUT", "-less", "suffix", None),

    # STUDY_OF
    ("STUDY_OF", "-logy", "suffix", "Greek"),

    # SKILL_OF
    ("SKILL_OF", "-ship", "suffix", None),

    # RESULT_OF
    ("RESULT_OF", "-age", "suffix", None),

    # PLACE_FOR
    ("PLACE_FOR", "-ary", "suffix", None),

    # CAUSATIVE
    ("CAUSATIVE", "-ize", "suffix", None),
    ("CAUSATIVE", "-ise", "suffix", None),
    ("CAUSATIVE", "-ate", "suffix", None),
    ("CAUSATIVE", "en-", "prefix", None),
    ("CAUSATIVE", "em-", "prefix", None),

    # PRACTICE_OF
    ("PRACTICE_OF", "-ism", "suffix", None),

    # REMOVAL
    ("REMOVAL", "de-", "prefix", "Latin"),

    # NEGATION
    ("NEGATION", "dis-", "prefix", "Latin"),
    ("NEGATION", "non-", "prefix", "Latin"),
    ("NEGATION", "un-", "prefix", None),
    ("NEGATION", "in-", "prefix", "Latin"),
    ("NEGATION", "im-", "prefix", "Latin"),
    ("NEGATION", "il-", "prefix", "Latin"),
    ("NEGATION", "ir-", "prefix", "Latin"),
    ("NEGATION", "a-", "prefix", "Greek"),
    ("NEGATION", "an-", "prefix", "Greek"),

    # ACROSS
    ("ACROSS", "trans-", "prefix", "Latin"),
    ("ACROSS", "dia-", "prefix", "Greek"),

    # OUT_FROM
    ("OUT_FROM", "ex-", "prefix", "Latin"),
    ("OUT_FROM", "e-", "prefix", "Latin"),

    # ONE
    ("ONE", "mono-", "prefix", "Greek"),
    ("ONE", "uni-", "prefix", "Latin"),

    # TWO
    ("TWO", "bi-", "prefix", "Latin"),
    ("TWO", "di-", "prefix", "Greek"),

    # THREE
    ("THREE", "tri-", "prefix", None),

    # MANY
    ("MANY", "multi-", "prefix", "Latin"),
    ("MANY", "poly-", "prefix", "Greek"),

    # BEFORE
    ("BEFORE", "pre-", "prefix", "Latin"),
    ("BEFORE", "pro-", "prefix", "Latin"),

    # AFTER
    ("AFTER", "post-", "prefix", "Latin"),

    # BAD
    ("BAD", "mal-", "prefix", "Latin"),
    ("BAD", "mis-", "prefix", None),

    # GOOD
    ("GOOD", "bene-", "prefix", "Latin"),

    # UNDER
    ("UNDER", "sub-", "prefix", "Latin"),

    # AGAIN
    ("AGAIN", "re-", "prefix", "Latin"),

    # BETWEEN
    ("BETWEEN", "inter-", "prefix", "Latin"),

    # WITHIN
    ("WITHIN", "intra-", "prefix", "Latin"),

    # TOGETHER
    ("TOGETHER", "co-", "prefix", "Latin"),
    ("TOGETHER", "com-", "prefix", "Latin"),
    ("TOGETHER", "con-", "prefix", "Latin"),
    ("TOGETHER", "col-", "prefix", "Latin"),

    # COMPLETELY
    ("COMPLETELY", "be-", "prefix", None),

    # AGAINST
    ("AGAINST", "anti-", "prefix", "Greek"),
    ("AGAINST", "contra-", "prefix", "Latin"),
    ("AGAINST", "counter-", "prefix", None),

    # STAR
    ("STAR", "astr", "root", "Greek"),

    # LIFE
    ("LIFE", "bio", "root", "Greek"),

    # EARTH
    ("EARTH", "geo", "root", "Greek"),

    # HEAT
    ("HEAT", "therm", "root", "Greek"),

    # SELF
    ("SELF", "auto", "root", "Greek"),

    # SAME
    ("SAME", "homo", "root", "Greek"),

    # WATER
    ("WATER", "hydr", "root", "Greek"),
    ("WATER", "aqua", "root", "Latin"),

    # SMALL
    ("SMALL", "micro", "root", "Greek"),

    # LARGE
    ("LARGE", "macro", "root", "Greek"),

    # SOUND
    ("SOUND", "phon", "root", "Greek"),

    # SEE
    ("SEE", "scope", "root", "Greek"),
    ("SEE", "spect", "root", "Latin"),
    ("SEE", "spec", "root", "Latin"),

    # WRITE
    ("WRITE", "graph", "root", "Greek"),
    ("WRITE", "script", "root", "Latin"),
    ("WRITE", "scrib", "root", "Latin"),

    # LIGHT
    ("LIGHT", "phot", "root", "Greek"),

    # DISTANT
    ("DISTANT", "tele", "root", "Greek"),

    # MEASURE
    ("MEASURE", "meter", "root", "Greek"),
    ("MEASURE", "metr", "root", "Greek"),

    # SUFFERING
    ("SUFFERING", "path", "root", "Greek"),
    ("SUFFERING", "pass", "root", "Greek"),

    # MIND
    ("MIND", "psych", "root", "Greek"),

    # ALL
    ("ALL", "pan", "root", "Greek"),

    # ANIMAL
    ("ANIMAL", "zoo", "root", "Greek"),

    # TIME
    ("TIME", "chron", "root", "Greek"),

    # FEAR
    ("FEAR", "phobia", "root", "Greek"),

    # CARRY
    ("CARRY", "port", "root", "Latin"),

    # SHAPE
    ("SHAPE", "form", "root", "Latin"),

    # PULL
    ("PULL", "tract", "root", "Latin"),

    # BREAK
    ("BREAK", "rupt", "root", "Latin"),

    # BUILD
    ("BUILD", "struct", "root", "Latin"),
    ("BUILD", "stru", "root", "Latin"),

    # SPEAK
    ("SPEAK", "dict", "root", "Latin"),
    ("SPEAK", "dic", "root", "Latin"),

    # BEND
    ("BEND", "flec", "root", "Latin"),
    ("BEND", "flex", "root", "Latin"),

    # BELIEVE
    ("BELIEVE", "cred", "root", "Latin"),

    # PUSH
    ("PUSH", "pel", "root", "Latin"),
    ("PUSH", "puls", "root", "Latin"),

    # MAKE
    ("MAKE", "fact", "root", "Latin"),
    ("MAKE", "fac", "root", "Latin"),

    # THROW
    ("THROW", "ject", "root", "Latin"),

    # TURN
    ("TURN", "vert", "root", "Latin"),
    ("TURN", "vers", "root", "Latin"),

    # SEND
    ("SEND", "mit", "root", "Latin"),
    ("SEND", "mis", "root", "Latin"),

    # DIE
    ("DIE", "mort", "root", "Latin"),

    # JOIN
    ("JOIN", "junct", "root", "Latin"),

    # KILL
    ("KILL", "cide", "root", "Latin"),

    # FORCE
    ("FORCE", "press", "root", "Latin"),

    # BREATHE
    ("BREATHE", "spire", "root", "Latin"),

    # STEP
    ("STEP", "grad", "root", "Latin"),
    ("STEP", "gress", "root", "Latin"),

    # TAKE
    ("TAKE", "cept", "root", "Latin"),
    ("TAKE", "capt", "root", "Latin"),
]


def init_db():
    """Create database and tables"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Concepts table - the core, language-independent meanings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL UNIQUE,
            description TEXT
        )
    """)

    # Morpheme forms table - surface forms linked to concepts
    cur.execute("""
        CREATE TABLE IF NOT EXISTS morpheme_forms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id INTEGER NOT NULL,
            morpheme TEXT NOT NULL,
            morpheme_type TEXT,
            origin TEXT,
            FOREIGN KEY (concept_id) REFERENCES concepts(id),
            UNIQUE(concept_id, morpheme)
        )
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forms_concept ON morpheme_forms(concept_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forms_morpheme ON morpheme_forms(morpheme)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forms_type ON morpheme_forms(morpheme_type)")

    conn.commit()
    return conn


def insert_data(conn):
    """Insert concepts and morpheme forms"""
    cur = conn.cursor()

    # Insert concepts
    for concept, description in CONCEPTS:
        cur.execute(
            "INSERT OR IGNORE INTO concepts (concept, description) VALUES (?, ?)",
            (concept, description)
        )
    conn.commit()

    # Get concept IDs
    cur.execute("SELECT id, concept FROM concepts")
    concept_ids = {row[1]: row[0] for row in cur.fetchall()}

    # Insert morpheme forms
    inserted = 0
    for concept_name, morpheme, mtype, origin in MORPHEME_FORMS:
        concept_id = concept_ids.get(concept_name)
        if concept_id:
            try:
                cur.execute("""
                    INSERT INTO morpheme_forms (concept_id, morpheme, morpheme_type, origin)
                    VALUES (?, ?, ?, ?)
                """, (concept_id, morpheme, mtype, origin))
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # Skip duplicates

    conn.commit()
    return inserted


def main():
    print(f"Creating database at {DB_PATH}")
    conn = init_db()

    inserted = insert_data(conn)
    print(f"Inserted {inserted} morpheme forms")

    cur = conn.cursor()

    # Show concept counts
    cur.execute("SELECT COUNT(*) FROM concepts")
    print(f"Total concepts: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM morpheme_forms")
    print(f"Total morpheme forms: {cur.fetchone()[0]}")

    # Sample output
    print("\nSample concepts with their forms:")
    cur.execute("""
        SELECT c.concept, c.description, GROUP_CONCAT(m.morpheme, ', ')
        FROM concepts c
        LEFT JOIN morpheme_forms m ON c.id = m.concept_id
        GROUP BY c.id
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[2]}")

    conn.close()


if __name__ == "__main__":
    main()
