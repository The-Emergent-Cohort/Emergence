# Import Pipeline Plan
## Concept-Based Tokenizer Database Build

**Date:** 2026-01-01
**Status:** Active
**Location:** frankenputer `/usr/share/databases/`

---

## 1. Data Directory Structure (frankenputer)

```
/usr/share/databases/
├── db/                         # Output databases go here
│   ├── primitives.db           # Core primitives (always loaded)
│   ├── language_registry.db    # Language codes (always loaded)
│   ├── lang/                   # Per-language DBs
│   │   ├── eng.db
│   │   ├── deu.db
│   │   └── ...
│   └── families/               # Minor language family DBs
│
├── reference/                  # Source data (read-only)
│   ├── glottolog/              # ✓ Already present
│   │   └── glottolog-glottolog-cldf-4dbf078/
│   │       └── cldf/
│   │
│   ├── grambank/               # ✓ Already present
│   │   └── grambank-grambank-9e0f341/
│   │       └── cldf/
│   │
│   ├── wals/                   # ✓ Already present
│   │   └── cldf-datasets-wals-0f5cd82/
│   │       └── cldf/
│   │
│   ├── kaikki/                 # ✓ Directory exists
│   │   ├── en/                 # Add: kaikki.org-dictionary-English.jsonl
│   │   ├── de/                 # Add more languages as needed
│   │   └── ...
│   │
│   ├── verbnet/                # ← ADD: Unpack verbnet-3.4.tar.gz
│   │   └── verbnet3.4/
│   │       ├── accept-77.xml
│   │       └── ... (~270 XML files)
│   │
│   ├── wordnet/                # ← ADD: Unpack wn3.1.dict.tar.gz
│   │   └── dict/
│   │       ├── data.noun
│   │       ├── data.verb
│   │       └── ...
│   │
│   └── omw/                    # ← ADD: Unpack omw-1.4.tar.xz
│       └── omw-1.4/
│           ├── als/            # Albanian
│           ├── arb/            # Arabic
│           └── ... (~31 languages)
│
├── scripts -> /mnt/project/tokenizers/scripts  # Symlink to project
│
└── venv/                       # Python environment
```

### Unpack Commands

```bash
cd /usr/share/databases/reference

# VerbNet
mkdir -p verbnet && cd verbnet
tar -xzf ~/Downloads/verbnet-3.4.tar.gz
cd ..

# WordNet
mkdir -p wordnet && cd wordnet
tar -xzf ~/Downloads/wn3.1.dict.tar.gz
cd ..

# OMW
mkdir -p omw && cd omw
tar -xJf ~/Downloads/omw-1.4.tar.xz
cd ..
```

---

## 2. Database Architecture

### 2.1 Core DBs (Always Loaded)

```
db/
├── primitives.db               # ~5MB - semantic primitives
│   ├── primitives              # 425 base concepts
│   ├── primitive_forms         # Cross-linguistic surface forms
│   └── primitive_relations     # How primitives relate
│
├── language_registry.db        # ~1MB - language codes & metadata
│   ├── language_codes          # 4-digit code assignments
│   ├── language_families       # Family hierarchy
│   └── language_features       # Typological features (WALS/Grambank)
│
└── modifiers.db                # ~100KB - grammar tokens
    └── modifiers               # Tense, case, number, etc.
```

### 2.2 Language DBs (Loaded on Demand)

```
db/lang/
├── eng.db                      # English (largest)
├── zho.db                      # Chinese
├── spa.db                      # Spanish
├── ara.db                      # Arabic
├── hin.db                      # Hindi
├── jpn.db                      # Japanese
├── deu.db                      # German
├── fra.db                      # French
├── por.db                      # Portuguese
├── rus.db                      # Russian
└── ...                         # Top 20-30 by speaker count

db/families/
├── indo-european-minor.db      # Smaller IE languages
├── austronesian.db             # Tagalog, Malay, etc.
├── uralic.db                   # Finnish, Hungarian, Estonian
├── niger-congo.db              # Swahili, Yoruba, etc.
└── ...
```

### 2.3 Per-Language DB Schema

Each language DB contains:
```sql
-- Concepts specific to this language
CREATE TABLE concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id INTEGER NOT NULL,          -- Full encoded token ID
    synset_hash TEXT NOT NULL,          -- Links to primitives.db
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,
    frequency INTEGER DEFAULT 0
);

-- Surface forms for this language
CREATE TABLE surface_forms (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,
    concept_id INTEGER NOT NULL,
    form_type TEXT,                     -- lemma, inflected, variant
    pos_features TEXT,
    frequency INTEGER DEFAULT 0
);

-- Compositions: how concepts decompose to primitives
CREATE TABLE compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    primitive_id INTEGER NOT NULL,      -- From primitives.db
    position INTEGER NOT NULL,
    weight REAL DEFAULT 1.0
);
```

---

## 3. Schema Updates Needed

### 3.1 New Tables for primitives.db

```sql
-- Core semantic primitives (NSM + image schemas + VerbNet frames)
CREATE TABLE primitives (
    primitive_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,       -- "KNOW", "MOVE", "CONTAINER"
    source TEXT NOT NULL,               -- 'nsm', 'image_schema', 'verbnet'
    domain INTEGER NOT NULL,            -- Matches token encoding domains
    category INTEGER NOT NULL,          -- Matches token encoding categories
    description TEXT,
    UNIQUE(canonical_name, source)
);

-- Cross-linguistic surface forms for primitives
CREATE TABLE primitive_forms (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    lang_code INTEGER NOT NULL,         -- 4-digit language code
    surface_form TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,                        -- 'nsm_research', 'omw', 'manual'
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    UNIQUE(primitive_id, lang_code, surface_form)
);

-- Relations between primitives
CREATE TABLE primitive_relations (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    related_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,        -- 'entails', 'contrasts', 'part_of'
    weight REAL DEFAULT 1.0,
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    FOREIGN KEY (related_id) REFERENCES primitives(primitive_id)
);
```

### 3.2 New Tables for language_registry.db

```sql
-- 4-digit language code assignments
CREATE TABLE language_codes (
    code INTEGER PRIMARY KEY,           -- 0000-9999
    iso639_3 TEXT,                      -- 'eng', 'deu', 'jpn'
    iso639_1 TEXT,                      -- 'en', 'de', 'ja'
    name TEXT NOT NULL,
    parent_code INTEGER,                -- For dialects (en-US -> en)
    family_code INTEGER,                -- Top-level family
    level TEXT,                         -- 'family', 'language', 'dialect'
    db_file TEXT,                       -- Which DB file contains this
    speaker_count INTEGER,
    FOREIGN KEY (parent_code) REFERENCES language_codes(code),
    FOREIGN KEY (family_code) REFERENCES language_codes(code)
);

-- Code ranges by family (for efficient queries)
CREATE TABLE code_ranges (
    family_code INTEGER NOT NULL,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    description TEXT,
    PRIMARY KEY (family_code, range_start)
);
```

---

## 4. Import Pipeline (Execution Order)

### Phase 1: Foundation (No Dependencies)

```
1.1 NSM Primes → primitives.db
    - Source: Hardcoded list (65 primes from Wierzbicka)
    - Creates: primitives table entries with source='nsm'
    - Script: scripts/import_nsm_primes.py

1.2 Image Schemas → primitives.db
    - Source: Hardcoded list (~27 schemas from Lakoff/Johnson)
    - Creates: primitives table entries with source='image_schema'
    - Script: scripts/import_image_schemas.py

1.3 Language Codes → language_registry.db
    - Source: Glottolog data (already have import script)
    - Creates: language_codes table with 4-digit assignments
    - Script: scripts/import_language_codes.py
```

### Phase 2: VerbNet (Depends on: Phase 1)

```
2.1 VerbNet Frames → primitives.db
    - Source: data/verbnet/verbnet3.4/*.xml
    - Creates: primitives table entries with source='verbnet'
    - Extracts: ~270 verb classes as semantic frames
    - Script: scripts/import_verbnet.py
```

### Phase 3: WordNet (Depends on: Phase 2)

```
3.1 WordNet Synsets → Cross-reference
    - Source: data/wordnet/dict/
    - Creates: Synset ID → primitive mapping
    - Links WordNet synsets to our primitives where possible
    - Script: scripts/import_wordnet.py
```

### Phase 4: OMW (Depends on: Phase 3)

```
4.1 OMW Translations → primitive_forms
    - Source: data/omw/omw-1.4/
    - Creates: Cross-linguistic primitive forms
    - ~31 languages of translations
    - Script: scripts/import_omw.py
```

### Phase 5: Kaikki (Depends on: Phase 4)

```
5.1 Kaikki English → db/lang/eng.db
    - Source: data/kaikki/en/*.jsonl
    - Creates: Full English vocabulary with compositions
    - Script: scripts/import_kaikki_lang.py --lang en

5.2 Kaikki Other Languages → db/lang/*.db
    - Repeat for each downloaded language
    - Script: scripts/import_kaikki_lang.py --lang <code>
```

### Phase 6: Composition Analysis (Depends on: Phase 5)

```
6.1 Decompose concepts to primitives
    - For each concept, analyze gloss to find primitive components
    - Compute fingerprints
    - Assign token IDs
    - Script: scripts/compute_compositions.py
```

---

## 5. Token ID Encoding (Reference)

```
Format: AADDCCLLLLFFFFFFCCC (19 digits)

AA     = Abstraction level (01-99)
        01 = primitive
        02 = direct composition
        03+ = compound compositions

DD     = Domain (01-99)
        01 = physical
        02 = temporal
        03 = mental
        04 = social
        05 = abstract
        ...

CC     = Category within domain (01-99)

LLLL   = Language code (0000-9999)
        0000 = universal/cross-linguistic
        0100-0149 = English variants
        0150-0199 = Chinese variants
        ...

FFFFFF = Fingerprint (000000-999999)
        Weighted sum of primitive IDs × positions

CCC    = Collision counter (000-999)
        Disambiguates within same fingerprint slot
```

---

## 6. Language Code Allocation

```
0000        = Universal (primitives, cross-linguistic concepts)

0001-0099   = Reserved for language families
              0001 = Indo-European
              0002 = Sino-Tibetan
              0003 = Afro-Asiatic
              0004 = Niger-Congo
              ...

0100-0999   = Major languages (with dialect sub-ranges)
              0100-0149 = English
                0100 = English (universal)
                0101 = en-GB
                0102 = en-US
                0103 = en-AU
                0104 = en-IN
                ...
              0150-0199 = Chinese
                0150 = Chinese (universal)
                0151 = cmn (Mandarin)
                0152 = yue (Cantonese)
                ...
              0200-0249 = Spanish
              0250-0299 = Arabic
              0300-0349 = Hindi
              0350-0399 = Japanese
              0400-0449 = German
              0450-0499 = French
              ...

1000-9999   = Minor languages (grouped by family)
              1000-1999 = Indo-European minor
              2000-2999 = Austronesian
              3000-3999 = Uralic
              ...
```

---

## 7. Execution Commands (on frankenputer)

```bash
cd /usr/share/databases
source venv/bin/activate

# Phase 1: Foundation
python scripts/import_nsm_primes.py --db db/primitives.db
python scripts/import_image_schemas.py --db db/primitives.db
python scripts/import_glottolog.py --source reference/glottolog/ --db db/language_registry.db
python scripts/import_wals.py --source reference/wals/ --db db/language_registry.db
python scripts/import_grambank.py --source reference/grambank/ --db db/language_registry.db

# Phase 2: VerbNet
python scripts/import_verbnet.py --source reference/verbnet/verbnet3.4/ --db db/primitives.db

# Phase 3: WordNet
python scripts/import_wordnet.py --source reference/wordnet/dict/ --db db/primitives.db

# Phase 4: OMW
python scripts/import_omw.py --source reference/omw/omw-1.4/ --db db/primitives.db

# Phase 5: Kaikki (per language)
python scripts/import_kaikki_lang.py --lang en --source reference/kaikki/ --db db/lang/eng.db
python scripts/import_kaikki_lang.py --lang de --source reference/kaikki/ --db db/lang/deu.db
# ... repeat for other languages

# Phase 6: Composition analysis
python scripts/compute_compositions.py --primitives db/primitives.db --lang-dir db/lang/
```

---

## 8. Script Status

```
scripts/
├── download_kaikki.py          # ✓ EXISTS - Download Kaikki data
├── download_typology.py        # ✓ EXISTS - Download WALS/Grambank
├── import_glottolog.py         # ✓ EXISTS - Language metadata
├── import_grambank.py          # ✓ EXISTS - Typological features
├── import_wals.py              # ✓ EXISTS - Typological features
├── import_kaikki.py            # ✓ EXISTS - Dictionary import
├── import_nsm_primes.py        # ✓ EXISTS - NSM semantic primes
│
├── import_image_schemas.py     # ○ TO CREATE - Embodied image schemas
├── import_verbnet.py           # ○ TO CREATE - Verb frames
├── import_wordnet.py           # ○ TO CREATE - Synset structure
├── import_omw.py               # ○ TO CREATE - Multilingual forms
├── import_language_codes.py    # ○ TO CREATE - 4-digit code assignments
├── compute_compositions.py     # ○ TO CREATE - Primitive decomposition
└── build_all.py                # ○ TO CREATE - Full pipeline orchestrator

db/
├── SCHEMA-primitives.sql       # ✓ EXISTS
├── SCHEMA-language-registry.sql # ✓ EXISTS
├── SCHEMA-language-template.sql # ✓ EXISTS
└── primitives.db               # ✓ EXISTS (seeded with NSM primes)

lib/
└── token_encoder.py            # ✓ EXISTS - Token ID encoding/decoding
```

---

## 9. Notes

- **Iteration Expected**: First pass won't be perfect. We'll likely need to:
  - Refine primitive decomposition heuristics
  - Adjust domain/category assignments
  - Handle edge cases in each data source

- **Compositionality**: The key insight is that most concepts decompose to primitives.
  Kaikki glosses like "to move quickly" map to MOVE + FAST primitives.

- **Language Loading**: At runtime, only load primitives.db + current language.
  Cross-language queries route through primitives as the hub.

- **Fingerprint Collisions**: Expected to be rare. The 3-digit collision counter
  handles chemistry/physics cases where many compounds share similar compositions.
