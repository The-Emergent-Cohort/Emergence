#!/usr/bin/env python3
"""
Import Open Multilingual Wordnet (OMW) translations.

OMW provides translations of WordNet synsets into 29+ languages.
Links multilingual lemmas to English synsets via shared fingerprints.

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py and import_wordnet.py to have been run first
"""

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
LANG_DIR = DB_DIR / "lang"
LANG_REGISTRY_DB = DB_DIR / "language_registry.db"

# Language DB schema
LANGUAGE_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,
    abstraction INTEGER NOT NULL DEFAULT 99,
    fingerprint INTEGER,
    source TEXT
);

CREATE TABLE IF NOT EXISTS surface_forms (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    form TEXT NOT NULL,
    form_type TEXT,
    features TEXT,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    component_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    relation TEXT DEFAULT 'part',
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,
    target_lemma TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE IF NOT EXISTS token_index (
    idx INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,
    description TEXT,
    location TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_surface_form ON surface_forms(form);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_token ON concepts(token_id);
CREATE INDEX IF NOT EXISTS idx_concept_fp ON concepts(fingerprint);
"""

# OMW language code to ISO 639-3 mapping
OMW_LANG_MAP = {
    "als": "als",  # Alemannic
    "arb": "arb",  # Arabic
    "bul": "bul",  # Bulgarian
    "cat": "cat",  # Catalan
    "cmn": "cmn",  # Mandarin Chinese
    "dan": "dan",  # Danish
    "ell": "ell",  # Greek
    "eng": "eng",  # English
    "eus": "eus",  # Basque
    "fas": "fas",  # Persian
    "fin": "fin",  # Finnish
    "fra": "fra",  # French
    "glg": "glg",  # Galician
    "heb": "heb",  # Hebrew
    "hrv": "hrv",  # Croatian
    "ind": "ind",  # Indonesian
    "ita": "ita",  # Italian
    "jpn": "jpn",  # Japanese
    "nld": "nld",  # Dutch
    "nno": "nno",  # Norwegian Nynorsk
    "nob": "nob",  # Norwegian Bokmål
    "pol": "pol",  # Polish
    "por": "por",  # Portuguese
    "ron": "ron",  # Romanian
    "slv": "slv",  # Slovenian
    "spa": "spa",  # Spanish
    "swe": "swe",  # Swedish
    "tha": "tha",  # Thai
    "zsm": "zsm",  # Malay
}


def get_lang_genomic(iso_code: str) -> str:
    """Get genomic code for a language from registry."""
    if not LANG_REGISTRY_DB.exists():
        return f"0.0.0.0"  # Unknown

    conn = sqlite3.connect(LANG_REGISTRY_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT genomic_code FROM language_codes WHERE iso639_3 = ?",
        (iso_code,)
    )
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else "0.0.0.0"


def parse_omw_tab(filepath: Path) -> dict:
    """Parse OMW tab-separated format."""
    entries = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                synset_id = parts[0]
                relation = parts[1]
                value = parts[2]

                if relation in ("lemma", "word"):
                    if synset_id not in entries:
                        entries[synset_id] = []
                    entries[synset_id].append(value)

    return entries


def parse_omw_xml(filepath: Path) -> dict:
    """Parse OMW LMF XML format."""
    entries = {}

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Handle namespace if present
        ns = {"": ""}
        if root.tag.startswith("{"):
            ns_end = root.tag.find("}")
            ns = {"lmf": root.tag[1:ns_end]}

        for lexentry in root.findall(".//LexicalEntry", ns) + root.findall(".//lexentry", ns):
            # Get synset reference
            sense = lexentry.find(".//Sense", ns) or lexentry.find(".//sense", ns)
            if sense is None:
                continue

            synset_id = sense.get("synset", "")
            if not synset_id:
                continue

            # Get lemma
            lemma_elem = lexentry.find(".//Lemma", ns) or lexentry.find(".//lemma", ns)
            if lemma_elem is None:
                continue

            lemma = lemma_elem.get("writtenForm", "")
            if not lemma:
                continue

            if synset_id not in entries:
                entries[synset_id] = []
            entries[synset_id].append(lemma)

    except ET.ParseError:
        pass

    return entries


def synset_to_fingerprint(synset_id: str) -> int:
    """Convert synset ID to fingerprint."""
    # Extract offset from synset ID (e.g., "00001740-n" -> 1740)
    parts = synset_id.replace("eng-", "").split("-")
    if parts:
        try:
            return int(parts[0]) % 1000000
        except ValueError:
            pass
    return 0


def generate_token_id(lang_genomic: str, pos: str, fingerprint: int) -> str:
    """Generate genomic token ID."""
    domain = {"n": 5, "v": 1, "a": 9, "r": 2, "s": 9}.get(pos, 5)
    return f"2.{domain}.1.{lang_genomic}.{fingerprint}.0"


def main():
    print("=" * 60)
    print("Importing Open Multilingual Wordnet")
    print("=" * 60)

    # Find OMW data
    omw_dir = REF_DIR / "omw"
    if not omw_dir.exists():
        # Try alternatives
        for alt in ["open-multilingual-wordnet", "OMW"]:
            alt_dir = REF_DIR / alt
            if alt_dir.exists():
                omw_dir = alt_dir
                break

    if not omw_dir.exists():
        print(f"ERROR: OMW directory not found in {REF_DIR}")
        print("Run unpack_tarballs.py first or download OMW.")
        return 1

    # Find language directories/files
    lang_sources = {}

    # Check for tab files
    for tab_file in omw_dir.rglob("*.tab"):
        lang_code = tab_file.stem.split("-")[0] if "-" in tab_file.stem else tab_file.stem[:3]
        if lang_code in OMW_LANG_MAP or len(lang_code) == 3:
            lang_sources[lang_code] = ("tab", tab_file)

    # Check for XML files
    for xml_file in omw_dir.rglob("*.xml"):
        lang_code = xml_file.stem.split("-")[0] if "-" in xml_file.stem else xml_file.stem[:3]
        if lang_code in OMW_LANG_MAP or len(lang_code) == 3:
            if lang_code not in lang_sources:  # Prefer tab
                lang_sources[lang_code] = ("xml", xml_file)

    # Check for language subdirectories
    for lang_dir in omw_dir.iterdir():
        if lang_dir.is_dir() and len(lang_dir.name) == 3:
            for f in lang_dir.glob("*"):
                if f.suffix in (".tab", ".xml"):
                    lang_sources[lang_dir.name] = (f.suffix[1:], f)
                    break

    if not lang_sources:
        print(f"ERROR: No OMW language files found in {omw_dir}")
        return 1

    print(f"\nFound {len(lang_sources)} languages in OMW")
    LANG_DIR.mkdir(parents=True, exist_ok=True)

    total_concepts = 0
    languages_processed = 0

    for lang_code, (file_type, filepath) in sorted(lang_sources.items()):
        if lang_code == "eng":
            continue  # Skip English, already have WordNet

        iso_code = OMW_LANG_MAP.get(lang_code, lang_code)
        print(f"\nProcessing {iso_code}...")

        # Parse entries
        if file_type == "tab":
            entries = parse_omw_tab(filepath)
        else:
            entries = parse_omw_xml(filepath)

        if not entries:
            print(f"  No entries found, skipping")
            continue

        # Get genomic code
        lang_genomic = get_lang_genomic(iso_code)

        # Create/open language database
        lang_db = LANG_DIR / f"{iso_code}.db"
        conn = sqlite3.connect(lang_db)
        conn.executescript(LANGUAGE_DB_SCHEMA)
        cursor = conn.cursor()

        # Track existing to avoid duplicates
        cursor.execute("SELECT fingerprint FROM concepts WHERE source = 'omw'")
        existing_fps = {row[0] for row in cursor.fetchall()}

        inserted = 0
        for synset_id, lemmas in entries.items():
            fingerprint = synset_to_fingerprint(synset_id)

            if fingerprint in existing_fps:
                continue

            # Determine POS from synset ID
            pos = "noun"
            if "-" in synset_id:
                pos_code = synset_id.split("-")[-1]
                pos = {"n": "noun", "v": "verb", "a": "adj", "r": "adv", "s": "adj"}.get(pos_code, "noun")

            # Primary lemma
            primary_lemma = lemmas[0] if lemmas else ""
            if not primary_lemma:
                continue

            token_id = generate_token_id(lang_genomic, pos[0], fingerprint)

            cursor.execute("""
                INSERT OR IGNORE INTO concepts
                (token_id, lemma, pos, abstraction, fingerprint, source)
                VALUES (?, ?, ?, 2, ?, 'omw')
            """, (token_id, primary_lemma, pos, fingerprint))

            if cursor.rowcount > 0:
                concept_id = cursor.lastrowid

                # Add all lemmas as surface forms
                for lemma in lemmas:
                    cursor.execute("""
                        INSERT OR IGNORE INTO surface_forms
                        (concept_id, form, form_type)
                        VALUES (?, ?, 'lemma')
                    """, (concept_id, lemma))

                inserted += 1
                existing_fps.add(fingerprint)

        conn.commit()
        conn.close()

        if inserted > 0:
            print(f"  Added {inserted} concepts to {lang_db.name}")
            total_concepts += inserted
            languages_processed += 1

    print("\n" + "=" * 60)
    print("OMW import complete!")
    print("=" * 60)
    print(f"\nLanguages processed: {languages_processed}")
    print(f"Total concepts added: {total_concepts}")
    print(f"\nDatabases in: {LANG_DIR}")

    return 0


if __name__ == "__main__":
    exit(main())
