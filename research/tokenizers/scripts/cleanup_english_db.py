#!/usr/bin/env python3
"""
Cleanup and migration script for English DB.

Handles three tasks:
1. Add missing capitalized common nouns (January, Monday, etc.) with capitalize_output=1
2. Identify jargon terms that should be in jargon DBs and move them
3. Delete moved terms from eng.db

Run after import_kaikki.py to fix data imported before routing was added.

Usage:
    python cleanup_english_db.py --add-capitalized   # Add missing capitalized nouns
    python cleanup_english_db.py --move-jargon       # Move jargon to separate DBs
    python cleanup_english_db.py --all               # Do both
    python cleanup_english_db.py --dry-run --all     # Preview without changes
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
LANG_DIR = DB_DIR / "lang"
JARGON_DIR = DB_DIR / "jargon"
KAIKKI_DIR = BASE_DIR / "reference" / "kaikki"

# Import jargon detection from import_kaikki
sys.path.insert(0, str(SCRIPT_DIR))
from import_kaikki import (
    JARGON_GENOMIC, CAPITALIZE_PATTERNS, POS_DOMAIN,
    detect_jargon_genomic, is_capitalized_common_noun,
    generate_fingerprint, LANGUAGE_DB_SCHEMA
)


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def get_or_create_jargon_db(family: int) -> sqlite3.Connection:
    """Get or create a jargon domain database."""
    family_names = {51: "math", 52: "chemistry", 53: "physics", 54: "computing", 55: "music"}
    name = family_names.get(family, f"domain_{family}")

    JARGON_DIR.mkdir(parents=True, exist_ok=True)
    jargon_db = JARGON_DIR / f"{name}.db"

    conn = sqlite3.connect(jargon_db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(LANGUAGE_DB_SCHEMA)

    return conn


def add_capitalized_nouns(dry_run: bool = False) -> dict:
    """
    Scan Kaikki for capitalized common nouns and add missing ones to eng.db.
    """
    stats = {"scanned": 0, "matched": 0, "already_exists": 0, "added": 0}

    kaikki_file = KAIKKI_DIR / "kaikki.org-dictionary-English.jsonl"
    if not kaikki_file.exists():
        # Try alternate name
        kaikki_file = KAIKKI_DIR / "english.jsonl"
    if not kaikki_file.exists():
        log(f"English Kaikki file not found in {KAIKKI_DIR}", "ERROR")
        return stats

    eng_db = LANG_DIR / "eng.db"
    if not eng_db.exists():
        log(f"English DB not found: {eng_db}", "ERROR")
        return stats

    log(f"Scanning {kaikki_file.name} for capitalized common nouns...")

    conn = sqlite3.connect(eng_db)
    cursor = conn.cursor()

    # Get existing fingerprints
    cursor.execute("SELECT fingerprint FROM concepts WHERE source = 'kaikki'")
    existing_fps = {row[0] for row in cursor.fetchall()}

    # Also check which capitalized words we already have
    cursor.execute("SELECT lemma FROM concepts WHERE capitalize_output = 1")
    existing_cap = {row[0] for row in cursor.fetchall()}

    with open(kaikki_file, "r", encoding="utf-8") as f:
        for line in f:
            stats["scanned"] += 1
            if stats["scanned"] % 100000 == 0:
                log(f"  Scanned {stats['scanned']:,} entries...")

            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "")
            if not word:
                continue

            # Check if this is a capitalized common noun
            if not is_capitalized_common_noun(word):
                continue

            stats["matched"] += 1

            # Skip proper nouns
            pos = entry.get("pos", "")
            if pos in ("name", "proper noun", "prop"):
                continue

            # Check if already exists
            if word in existing_cap:
                stats["already_exists"] += 1
                continue

            # Process senses
            for sense in entry.get("senses", []):
                glosses = sense.get("glosses", [])
                if not glosses:
                    continue

                gloss = glosses[0]
                fingerprint = generate_fingerprint(word, gloss)

                if fingerprint in existing_fps:
                    # Update existing entry to set capitalize_output
                    if not dry_run:
                        cursor.execute(
                            "UPDATE concepts SET capitalize_output = 1 WHERE fingerprint = ?",
                            (fingerprint,)
                        )
                    stats["already_exists"] += 1
                else:
                    # Insert new entry
                    domain = POS_DOMAIN.get(pos, 5)
                    token_id = f"2.{domain}.1.1.8.127.0.{fingerprint}.0"

                    if not dry_run:
                        cursor.execute("""
                            INSERT OR IGNORE INTO concepts
                            (token_id, lemma, gloss, pos, abstraction, fingerprint, capitalize_output, source)
                            VALUES (?, ?, ?, ?, 2, ?, 1, 'kaikki')
                        """, (token_id, word, gloss[:500], pos, fingerprint))
                    existing_fps.add(fingerprint)
                    stats["added"] += 1

                break  # One sense per word for now

    if not dry_run:
        conn.commit()
    conn.close()

    return stats


def move_jargon_to_dbs(dry_run: bool = False) -> dict:
    """
    Identify jargon terms in eng.db and move them to appropriate jargon DBs.
    """
    stats = {"scanned": 0, "jargon_found": 0, "moved": 0, "deleted": 0, "by_family": {}}

    kaikki_file = KAIKKI_DIR / "kaikki.org-dictionary-English.jsonl"
    if not kaikki_file.exists():
        kaikki_file = KAIKKI_DIR / "english.jsonl"
    if not kaikki_file.exists():
        log(f"English Kaikki file not found in {KAIKKI_DIR}", "ERROR")
        return stats

    eng_db = LANG_DIR / "eng.db"
    if not eng_db.exists():
        log(f"English DB not found: {eng_db}", "ERROR")
        return stats

    log(f"Scanning {kaikki_file.name} for jargon terms to move...")

    # Open eng.db
    conn = sqlite3.connect(eng_db)
    cursor = conn.cursor()

    # Get existing fingerprints in eng.db
    cursor.execute("SELECT fingerprint, concept_id, lemma FROM concepts WHERE source = 'kaikki'")
    eng_concepts = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    # Track jargon DBs
    jargon_conns = {}  # family -> conn
    jargon_fps = {}    # family -> set of fingerprints

    # Fingerprints to delete from eng.db
    to_delete = []

    with open(kaikki_file, "r", encoding="utf-8") as f:
        for line in f:
            stats["scanned"] += 1
            if stats["scanned"] % 100000 == 0:
                log(f"  Scanned {stats['scanned']:,} entries...")

            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "")
            pos = entry.get("pos", "")
            if not word:
                continue

            # Skip proper nouns
            if pos in ("name", "proper noun", "prop"):
                continue

            # Check if jargon
            jargon_genomic = detect_jargon_genomic(entry)
            if not jargon_genomic:
                continue

            stats["jargon_found"] += 1
            family = jargon_genomic[0]

            # Initialize jargon DB connection if needed
            if family not in jargon_conns:
                jargon_conns[family] = get_or_create_jargon_db(family)
                jcursor = jargon_conns[family].cursor()
                jcursor.execute("SELECT fingerprint FROM concepts WHERE source = 'kaikki'")
                jargon_fps[family] = {row[0] for row in jcursor.fetchall()}
                stats["by_family"][family] = 0

            # Process senses
            for sense in entry.get("senses", []):
                glosses = sense.get("glosses", [])
                if not glosses:
                    continue

                gloss = glosses[0]
                fingerprint = generate_fingerprint(word, gloss)

                # Check if in eng.db
                if fingerprint not in eng_concepts:
                    continue

                # Check if already in jargon DB
                if fingerprint in jargon_fps[family]:
                    # Just mark for deletion from eng.db
                    to_delete.append(fingerprint)
                    continue

                # Move to jargon DB
                fam, sub, lng, dial = jargon_genomic
                domain = POS_DOMAIN.get(pos, 5)
                token_id = f"2.{domain}.1.{fam}.{sub}.{lng}.{dial}.{fingerprint}.0"

                if not dry_run:
                    jcursor = jargon_conns[family].cursor()
                    jcursor.execute("""
                        INSERT OR IGNORE INTO concepts
                        (token_id, lemma, gloss, pos, abstraction, fingerprint, capitalize_output, source)
                        VALUES (?, ?, ?, ?, 2, ?, 0, 'kaikki')
                    """, (token_id, word, gloss[:500], pos, fingerprint))

                jargon_fps[family].add(fingerprint)
                to_delete.append(fingerprint)
                stats["moved"] += 1
                stats["by_family"][family] = stats["by_family"].get(family, 0) + 1

                break  # One sense per word

    # Delete moved entries from eng.db
    if to_delete and not dry_run:
        log(f"Deleting {len(to_delete):,} entries from eng.db...")
        # Delete in batches
        for i in range(0, len(to_delete), 500):
            batch = to_delete[i:i+500]
            placeholders = ",".join("?" * len(batch))
            cursor.execute(f"DELETE FROM concepts WHERE fingerprint IN ({placeholders})", batch)
        conn.commit()
        stats["deleted"] = len(to_delete)

    # Commit and close jargon DBs
    for family, jconn in jargon_conns.items():
        if not dry_run:
            jconn.commit()
        jconn.close()

    conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Cleanup English DB")
    parser.add_argument("--add-capitalized", action="store_true",
                        help="Add missing capitalized common nouns")
    parser.add_argument("--move-jargon", action="store_true",
                        help="Move jargon terms to separate DBs")
    parser.add_argument("--all", action="store_true",
                        help="Do both operations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying DBs")

    args = parser.parse_args()

    if not (args.add_capitalized or args.move_jargon or args.all):
        parser.print_help()
        return 1

    log("=" * 60)
    log("English DB Cleanup")
    if args.dry_run:
        log("DRY RUN - no changes will be made")
    log("=" * 60)

    if args.add_capitalized or args.all:
        log("\n--- Adding Capitalized Common Nouns ---")
        stats = add_capitalized_nouns(dry_run=args.dry_run)
        log(f"  Scanned: {stats['scanned']:,}")
        log(f"  Matched patterns: {stats['matched']:,}")
        log(f"  Already exists: {stats['already_exists']:,}")
        log(f"  Added: {stats['added']:,}")

    if args.move_jargon or args.all:
        log("\n--- Moving Jargon Terms ---")
        stats = move_jargon_to_dbs(dry_run=args.dry_run)
        log(f"  Scanned: {stats['scanned']:,}")
        log(f"  Jargon found: {stats['jargon_found']:,}")
        log(f"  Moved: {stats['moved']:,}")
        log(f"  Deleted from eng.db: {stats['deleted']:,}")

        family_names = {51: "math", 52: "chemistry", 53: "physics", 54: "computing", 55: "music"}
        for family, count in stats.get("by_family", {}).items():
            log(f"    {family_names.get(family, family)}: {count:,}")

    log("\n" + "=" * 60)
    log("Cleanup complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
