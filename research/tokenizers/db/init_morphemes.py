#!/usr/bin/env python3
"""Initialize morphemes table with data from morphemeML.pdf"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tokenizer.db"

# Morpheme data extracted from morphemeML.pdf
# Format: (morpheme, meaning, morpheme_type, origin, pos_tendency)

SUFFIXES = [
    ("-er", "one who, that which", "suffix", None, "noun"),
    ("-er", "more", "suffix", None, "adjective"),  # Note: same form, different meaning
    ("-ly", "to act in a way that is", "suffix", None, "adverb"),
    ("-able", "capable of, or worthy of", "suffix", None, "adjective"),
    ("-ible", "capable of, or worthy of", "suffix", None, "adjective"),
    ("-hood", "condition of being", "suffix", None, "noun"),
    ("-ful", "full of, having", "suffix", None, "adjective"),
    ("-less", "without", "suffix", None, "adjective"),
    ("-ish", "somewhat like", "suffix", None, "adjective"),
    ("-ness", "condition or state of", "suffix", None, "noun"),
    ("-ic", "relating to", "suffix", None, "adjective"),
    ("-ist", "one who", "suffix", None, "noun"),
    ("-ian", "one who", "suffix", None, "noun"),
    ("-or", "one who", "suffix", None, "noun"),
    ("-eer", "one who", "suffix", None, "noun"),
    ("-logy", "study of", "suffix", None, "noun"),
    ("-ship", "art or skill of, condition, rank, group of", "suffix", None, "noun"),
    ("-ous", "full of, having, possessing", "suffix", None, "adjective"),
    ("-ive", "tending to", "suffix", None, "adjective"),
    ("-age", "result of an action", "suffix", None, "noun"),
    ("-ant", "a condition or state", "suffix", None, "adjective"),
    ("-ent", "someone who, something that", "suffix", None, "noun"),
    ("-ment", "state or act of", "suffix", None, "noun"),
    ("-ary", "place for, collection of", "suffix", None, "noun"),
    ("-ize", "to make", "suffix", None, "verb"),
    ("-ise", "to make", "suffix", None, "verb"),
    ("-ure", "action or condition of", "suffix", None, "noun"),
    ("-ion", "act or condition", "suffix", None, "noun"),
    ("-ation", "act or condition", "suffix", None, "noun"),
    ("-ance", "act or condition of", "suffix", None, "noun"),
    ("-ence", "act or condition of", "suffix", None, "noun"),
    ("-ity", "state or quality of", "suffix", None, "noun"),
    ("-al", "relating to", "suffix", None, "adjective"),
    ("-ate", "to make", "suffix", None, "verb"),
    ("-tude", "condition of", "suffix", None, "noun"),
    ("-ism", "practice, belief", "suffix", None, "noun"),
]

PREFIXES = [
    ("de-", "from, reduce, or opposite", "prefix", None, None),
    ("dis-", "opposite", "prefix", None, None),
    ("trans-", "across, over, through", "prefix", None, None),
    ("dia-", "across, through", "prefix", "Greek", None),
    ("ex-", "out, from", "prefix", "Latin", None),
    ("e-", "out, from", "prefix", "Latin", None),
    ("mono-", "one, single", "prefix", "Greek", None),
    ("uni-", "one, single", "prefix", "Latin", None),
    ("bi-", "two", "prefix", "Latin", None),
    ("di-", "two, or in parts", "prefix", "Greek", None),
    ("tri-", "three", "prefix", None, None),
    ("multi-", "many, much", "prefix", "Latin", None),
    ("poly-", "many, much", "prefix", "Greek", None),
    ("pre-", "before", "prefix", "Latin", None),
    ("post-", "after", "prefix", "Latin", None),
    ("mal-", "bad, evil", "prefix", "Latin", None),
    ("mis-", "wrong, bad", "prefix", None, None),
    ("bene-", "good, well", "prefix", "Latin", None),
    ("pro-", "forward, forth, before", "prefix", "Latin", None),
    ("sub-", "under, beneath", "prefix", "Latin", None),
    ("re-", "back, again", "prefix", "Latin", None),
    ("inter-", "among, between", "prefix", "Latin", None),
    ("intra-", "within", "prefix", "Latin", None),
    ("co-", "together, with", "prefix", "Latin", None),
    ("com-", "together, with", "prefix", "Latin", None),
    ("con-", "together, with", "prefix", "Latin", None),
    ("col-", "together, with", "prefix", "Latin", None),
    ("be-", "to, completely", "prefix", None, None),
    ("non-", "not", "prefix", None, None),
    ("un-", "not", "prefix", None, None),
    ("in-", "not", "prefix", "Latin", None),
    ("im-", "not", "prefix", "Latin", None),
    ("il-", "not", "prefix", "Latin", None),
    ("ir-", "not", "prefix", "Latin", None),
    ("a-", "not, negative", "prefix", "Greek", None),
    ("an-", "not, negative", "prefix", "Greek", None),
    ("anti-", "against, opposite", "prefix", "Greek", None),
    ("contra-", "against, opposite", "prefix", "Latin", None),
    ("counter-", "against, opposite", "prefix", None, None),
    ("en-", "to cause to be, to put or go into or onto", "prefix", None, None),
    ("em-", "to cause to be, to put or go into or onto", "prefix", None, None),
]

GREEK_ROOTS = [
    ("astr", "stars, heavens", "root", "Greek", None),
    ("bio", "life", "root", "Greek", None),
    ("geo", "earth, rocks", "root", "Greek", None),
    ("therm", "heat, warm", "root", "Greek", None),
    ("auto", "self", "root", "Greek", None),
    ("homo", "same, alike", "root", "Greek", None),
    ("hydr", "water", "root", "Greek", None),
    ("micro", "small", "root", "Greek", None),
    ("macro", "large", "root", "Greek", None),
    ("phon", "sound, speech", "root", "Greek", None),
    ("scope", "instrument used to observe, to see", "root", "Greek", None),
    ("graph", "written", "root", "Greek", None),
    ("phot", "light", "root", "Greek", None),
    ("tele", "distant, far", "root", "Greek", None),
    ("meter", "instrument used to measure", "root", "Greek", None),
    ("path", "suffering, disease", "root", "Greek", None),
    ("psych", "mind, mental", "root", "Greek", None),
    ("pan", "all, whole", "root", "Greek", None),
    ("zoo", "animal", "root", "Greek", None),
    ("chron", "time", "root", "Greek", None),
    ("phobia", "fear, intense dislike", "root", "Greek", None),
]

LATIN_ROOTS = [
    ("port", "to carry", "root", "Latin", None),
    ("form", "to shape", "root", "Latin", None),
    ("tract", "to pull", "root", "Latin", None),
    ("rupt", "to break", "root", "Latin", None),
    ("spect", "to see, to watch", "root", "Latin", None),
    ("spec", "to see, to watch", "root", "Latin", None),
    ("struct", "to build", "root", "Latin", None),
    ("stru", "to build", "root", "Latin", None),
    ("dict", "to tell, to say", "root", "Latin", None),
    ("dic", "to tell, to say", "root", "Latin", None),
    ("flec", "to bend", "root", "Latin", None),
    ("flex", "to bend", "root", "Latin", None),
    ("cred", "to believe", "root", "Latin", None),
    ("aqua", "water", "root", "Latin", None),
    ("pel", "to drive, push", "root", "Latin", None),
    ("puls", "to drive, push", "root", "Latin", None),
    ("fact", "to make, to do", "root", "Latin", None),
    ("fac", "to make, to do", "root", "Latin", None),
    ("ject", "to throw, to throw down", "root", "Latin", None),
    ("vert", "to turn", "root", "Latin", None),
    ("vers", "to turn", "root", "Latin", None),
    ("mit", "to send", "root", "Latin", None),
    ("mis", "to send", "root", "Latin", None),
    ("mort", "to die", "root", "Latin", None),
    ("script", "to write", "root", "Latin", None),
    ("scrib", "to write", "root", "Latin", None),
    ("junct", "to join", "root", "Latin", None),
    ("cide", "to kill, a killer", "root", "Latin", None),
    ("press", "to force, squeeze", "root", "Latin", None),
    ("spire", "to breathe", "root", "Latin", None),
    ("grad", "to step", "root", "Latin", None),
    ("gress", "to step", "root", "Latin", None),
    ("cept", "to take, seize, receive", "root", "Latin", None),
    ("capt", "to take, seize, receive", "root", "Latin", None),
]


def init_db():
    """Create database and morphemes table"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS morphemes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            morpheme TEXT NOT NULL,
            meaning TEXT,
            morpheme_type TEXT,
            origin TEXT,
            pos_tendency TEXT,
            UNIQUE(morpheme, meaning)
        )
    """)

    # Create indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_morphemes_type ON morphemes(morpheme_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_morphemes_pos ON morphemes(pos_tendency)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_morphemes_origin ON morphemes(origin)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_morphemes_morpheme ON morphemes(morpheme)")

    conn.commit()
    return conn


def insert_morphemes(conn, data):
    """Insert morpheme data, skip duplicates"""
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for row in data:
        try:
            cur.execute("""
                INSERT INTO morphemes (morpheme, meaning, morpheme_type, origin, pos_tendency)
                VALUES (?, ?, ?, ?, ?)
            """, row)
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    return inserted, skipped


def main():
    print(f"Creating database at {DB_PATH}")
    conn = init_db()

    all_data = SUFFIXES + PREFIXES + GREEK_ROOTS + LATIN_ROOTS
    inserted, skipped = insert_morphemes(conn, all_data)

    print(f"Inserted: {inserted}, Skipped (duplicates): {skipped}")

    # Show counts
    cur = conn.cursor()
    cur.execute("SELECT morpheme_type, COUNT(*) FROM morphemes GROUP BY morpheme_type")
    print("\nCounts by type:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cur.execute("SELECT COUNT(*) FROM morphemes")
    print(f"\nTotal morphemes: {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
