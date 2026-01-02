#!/usr/bin/env python3
"""
Import Glottolog CLDF data into language_registry.db.

Assigns genomic FF.SS.LLL.DD codes based on family hierarchy.

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py to have been run first
"""

import csv
import sqlite3
from pathlib import Path
from collections import defaultdict

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
LANGUAGE_REGISTRY_DB = DB_DIR / "language_registry.db"


def find_csv_file(directory: Path, patterns: list) -> Path:
    """Find a CSV file matching one of the patterns."""
    for pattern in patterns:
        matches = list(directory.rglob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


# Major language family assignments (FF codes)
FAMILY_CODES = {
    "indo1319": (1, "Indo-European"),
    "sino1245": (2, "Sino-Tibetan"),
    "afro1255": (3, "Afro-Asiatic"),
    "atla1278": (4, "Atlantic-Congo"),
    "aust1307": (5, "Austronesian"),
    "drav1251": (6, "Dravidian"),
    "ural1272": (7, "Uralic"),
    "turk1311": (8, "Turkic"),
    "japo1237": (9, "Japonic"),
    "kore1284": (10, "Koreanic"),
    "aust1305": (11, "Austroasiatic"),
    "taik1256": (12, "Tai-Kadai"),
    "nilo1247": (13, "Nilo-Saharan"),
    "pama1250": (14, "Pama-Nyungan"),
    "quec1387": (15, "Quechuan"),
    "tuca1253": (16, "Tucanoan"),
    "maya1287": (17, "Mayan"),
    "otom1299": (18, "Oto-Manguean"),
    "utos1215": (19, "Uto-Aztecan"),
    "algo1256": (20, "Algic"),
    "araw1281": (21, "Arawakan"),
    "cadd1255": (22, "Caddoan"),
    "carb1283": (23, "Cariban"),
    "chib1249": (24, "Chibchan"),
    "choc1280": (25, "Chocoan"),
    "guah1252": (26, "Guahiboan"),
    "jivu1234": (27, "Jivaro"),
    "pano1259": (28, "Panoan"),
    "tupi1275": (29, "Tupian"),
    "mong1349": (30, "Mongolic"),
    "tung1282": (31, "Tungusic"),
    "sepi1257": (32, "Sepik"),
    "nucl1709": (33, "Nuclear Trans New Guinea"),
    "mand1469": (34, "Mande"),
    "afro-asiatic_unclassified": (35, "Afro-Asiatic unclassified"),
    "khoe1240": (36, "Khoe-Kwadi"),
    "koma1264": (37, "Koman"),
    "sonl1242": (38, "Songhay"),
    "nakh1245": (39, "Nakh-Daghestanian"),
    "kart1248": (40, "Kartvelian"),
}

# Indo-European subfamily assignments (SS codes for FF=1)
IE_SUBFAMILY_CODES = {
    "germ1287": (8, "Germanic"),
    "roma1334": (12, "Romance"),
    "slav1255": (15, "Slavic"),
    "indo1321": (20, "Indo-Iranian"),
    "celt1248": (25, "Celtic"),
    "balt1263": (28, "Baltic"),
    "gree1276": (30, "Hellenic"),
    "alba1267": (32, "Albanian"),
    "arme1241": (34, "Armenian"),
}

# Sino-Tibetan subfamily assignments (SS codes for FF=2)
ST_SUBFAMILY_CODES = {
    "sini1245": (1, "Sinitic"),
    "tibe1272": (5, "Tibeto-Burman"),
}


def main():
    print("=" * 60)
    print("Importing Glottolog Language Data")
    print("=" * 60)

    if not LANGUAGE_REGISTRY_DB.exists():
        print(f"ERROR: {LANGUAGE_REGISTRY_DB} not found.")
        print("Run init_schemas.py first.")
        return 1

    # Find glottolog data
    glottolog_dir = REF_DIR / "glottolog"
    if not glottolog_dir.exists():
        print(f"ERROR: {glottolog_dir} not found.")
        return 1

    lang_file = find_csv_file(glottolog_dir, ["languages.csv", "*languages*.csv"])
    if not lang_file:
        print(f"ERROR: No languages.csv found in {glottolog_dir}")
        return 1

    print(f"\nFound: {lang_file}")

    conn = sqlite3.connect(LANGUAGE_REGISTRY_DB)
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM language_codes")
    cursor.execute("DELETE FROM language_families WHERE family_id > 0")

    # Read languages file
    print("\nParsing Glottolog data...")
    languages = []
    families_seen = set()
    subfamily_counters = defaultdict(lambda: 50)  # Start subfamilies at 50 for unmapped
    language_counters = defaultdict(lambda: 100)  # Start languages at 100 for unmapped

    with open(lang_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Map columns
        id_col = next((c for c in fieldnames if c.lower() in ['id', 'glottocode']), None)
        name_col = next((c for c in fieldnames if c.lower() == 'name'), None)
        iso_col = next((c for c in fieldnames if 'iso' in c.lower()), None)
        family_col = next((c for c in fieldnames if 'family' in c.lower()), None)
        parent_col = next((c for c in fieldnames if 'parent' in c.lower()), None)
        level_col = next((c for c in fieldnames if 'level' in c.lower()), None)

        for row in reader:
            glottocode = row.get(id_col, "") if id_col else ""
            name = row.get(name_col, "") if name_col else ""
            iso = row.get(iso_col, "") if iso_col else ""
            family_id = row.get(family_col, "") if family_col else ""
            parent_id = row.get(parent_col, "") if parent_col else ""
            level = row.get(level_col, "") if level_col else ""

            if not glottocode or not name:
                continue

            # Skip non-languages for now (families handled separately)
            if level == "family":
                families_seen.add(glottocode)
                continue

            # Determine FF (family code)
            ff = 99  # Unknown family
            family_name = "Unknown"
            if family_id in FAMILY_CODES:
                ff, family_name = FAMILY_CODES[family_id]

            # Determine SS (subfamily code)
            ss = 0  # No subfamily
            if ff == 1 and parent_id in IE_SUBFAMILY_CODES:
                ss, _ = IE_SUBFAMILY_CODES[parent_id]
            elif ff == 2 and parent_id in ST_SUBFAMILY_CODES:
                ss, _ = ST_SUBFAMILY_CODES[parent_id]

            # Assign LLL (language code within subfamily)
            key = (ff, ss)
            lll = language_counters[key]
            language_counters[key] += 1

            # DD = 0 for core
            dd = 0

            languages.append({
                "family": ff,
                "subfamily": ss,
                "language": lll,
                "dialect": dd,
                "iso639_3": iso if iso else None,
                "glottocode": glottocode,
                "name": name,
                "status": "living",
                "parent_glottocode": parent_id if parent_id else None,
                "level": level,
            })

    print(f"  Found {len(languages)} languages")

    # Update language_families table
    print("\nUpdating language families...")
    for glottocode, (ff, name) in FAMILY_CODES.items():
        cursor.execute("""
            INSERT OR REPLACE INTO language_families (family_id, name, glottocode)
            VALUES (?, ?, ?)
        """, (ff, name, glottocode))

    # Insert languages
    print("Inserting language codes...")
    cursor.executemany("""
        INSERT INTO language_codes
        (family, subfamily, language, dialect, iso639_3, glottocode, name, status, parent_glottocode, level)
        VALUES (:family, :subfamily, :language, :dialect, :iso639_3, :glottocode, :name, :status, :parent_glottocode, :level)
    """, languages)

    conn.commit()

    # Summary
    cursor.execute("SELECT COUNT(*) FROM language_codes")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT family) FROM language_codes")
    families_count = cursor.fetchone()[0]

    print("\n" + "=" * 60)
    print("Glottolog import complete!")
    print("=" * 60)
    print(f"\nLanguages: {total}")
    print(f"Families used: {families_count}")
    print(f"\nDatabase: {LANGUAGE_REGISTRY_DB}")

    # Show top families
    print("\nTop families by language count:")
    cursor.execute("""
        SELECT lf.name, COUNT(*) as cnt
        FROM language_codes lc
        JOIN language_families lf ON lc.family = lf.family_id
        GROUP BY lc.family
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for name, cnt in cursor.fetchall():
        print(f"  {name}: {cnt}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
