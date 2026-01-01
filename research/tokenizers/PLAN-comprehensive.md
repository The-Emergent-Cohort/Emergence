# Comprehensive Implementation Plan: Tokenizer Database System

**Date:** 2026-01-01
**Status:** Clean initialization - nothing built yet
**Purpose:** Conceptual and linguistic backbone for Digital Intelligence

---

## Frankenputer Environment

### Directory Structure
```
/usr/share/databases/
├── db/                          # OUTPUT: Generated databases go here
│   ├── primitives.db
│   ├── language_registry.db
│   └── lang/
│       ├── eng.db
│       └── ...
├── reference/                   # INPUT: Unpacked data sources
│   ├── glottolog/
│   ├── grambank/
│   ├── kaikki/
│   └── wals/
├── scripts/                     # SYMLINK -> /mnt/project/tokenizers/scripts
│   ├── data_tarballs/           # Archived sources (VerbNet, WordNet, OMW)
│   └── *.py                     # Import scripts
└── venv/                        # Python environment
```

### Key Paths (from scripts perspective)
```python
# Relative to /usr/share/databases/scripts/
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent                    # /usr/share/databases/
DB_DIR = BASE_DIR / "db"                        # /usr/share/databases/db/
REF_DIR = BASE_DIR / "reference"                # /usr/share/databases/reference/
TARBALL_DIR = SCRIPT_DIR / "data_tarballs"      # /usr/share/databases/scripts/data_tarballs/
```

### Important Notes
- Scripts must SEARCH for data files (csv, json, etc.) - don't hardcode exact paths
- Data files may be nested in subdirectories with version numbers
- Always use glob patterns to find files

---

## Executive Summary

The database IS the tokenizer. Every word is a query. Token IDs are derived from semantic structure using genomic notation, not arbitrarily assigned. The architecture supports differential computation (like game engines), language preservation (endangered languages as full deltas), and scales through assembly rather than brute-force parameter count.

---

## 1. Master Index Structure

### 1.1 The Minimal Routing Table

```sql
CREATE TABLE token_index (
    idx INTEGER PRIMARY KEY,              -- Sequential position
    token_id TEXT NOT NULL UNIQUE,        -- Genomic notation: A.D.C.FF.SS.LLL.DD.FP.COL
    description TEXT,                     -- Optional English gloss
    location TEXT NOT NULL                -- Container: 'primitives', 'lang/eng', etc.
);

CREATE INDEX idx_token_location ON token_index(location);
CREATE INDEX idx_token_prefix ON token_index(substr(token_id, 1, 10));
```

This replaces the traditional tokenizer's fixed vocabulary. No predetermined size.

---

## 2. Output Database Structure

### 2.1 Files to Generate

| Database | Purpose | Size Estimate |
|----------|---------|---------------|
| `db/primitives.db` | Semantic atoms (NSM, verb classes) | ~5MB |
| `db/language_registry.db` | Language codes, typology features | ~2MB |
| `db/lang/eng.db` | English vocabulary, compositions | ~500MB |
| `db/lang/{iso}.db` | Other languages | Varies |

**Note:** var.db excluded until abstraction 00 items discussed.

### 2.2 primitives.db Schema

```sql
CREATE TABLE primitives (
    primitive_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,           -- "KNOW", "MOVE", "CONTAINER"
    source TEXT NOT NULL,                   -- 'nsm', 'image_schema', 'verbnet'
    domain INTEGER NOT NULL,                -- Semantic domain (1-99)
    category INTEGER NOT NULL,              -- Category within domain (1-99)
    description TEXT,
    examples TEXT,
    UNIQUE(canonical_name, source)
);

CREATE TABLE primitive_forms (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    lang_genomic TEXT NOT NULL,             -- FF.SS.LLL.DD format
    surface_form TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id)
);

CREATE TABLE primitive_relations (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    related_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,            -- 'entails', 'contrasts', 'part_of'
    weight REAL DEFAULT 1.0,
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    FOREIGN KEY (related_id) REFERENCES primitives(primitive_id)
);

CREATE TABLE domains (
    domain_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (domain_id) REFERENCES domains(domain_id),
    UNIQUE(domain_id, name)
);
```

### 2.3 language_registry.db Schema

```sql
CREATE TABLE language_codes (
    id INTEGER PRIMARY KEY,
    family INTEGER NOT NULL,                -- FF (0-99)
    subfamily INTEGER NOT NULL,             -- SS (0-99)
    language INTEGER NOT NULL,              -- LLL (0-999)
    dialect INTEGER DEFAULT 0,              -- DD (0-999, 0=core)
    genomic_code TEXT GENERATED ALWAYS AS (
        family || '.' || subfamily || '.' || language || '.' || dialect
    ) STORED,
    iso639_3 TEXT,
    iso639_1 TEXT,
    glottocode TEXT UNIQUE,
    name TEXT NOT NULL,
    native_name TEXT,
    speaker_count INTEGER,
    status TEXT DEFAULT 'living',           -- 'living', 'extinct', 'historical', 'constructed'
    UNIQUE(family, subfamily, language, dialect)
);

CREATE TABLE language_families (
    family_id INTEGER PRIMARY KEY,          -- FF code
    name TEXT NOT NULL UNIQUE,
    glottocode TEXT,
    description TEXT
);

CREATE TABLE language_features (
    id INTEGER PRIMARY KEY,
    lang_id INTEGER NOT NULL,
    feature_id TEXT NOT NULL,               -- 'WALS_81A', 'GB020'
    value TEXT NOT NULL,
    value_name TEXT,
    source TEXT NOT NULL,                   -- 'wals', 'grambank'
    FOREIGN KEY (lang_id) REFERENCES language_codes(id),
    UNIQUE(lang_id, feature_id, source)
);

CREATE INDEX idx_lang_genomic ON language_codes(genomic_code);
CREATE INDEX idx_lang_iso ON language_codes(iso639_3);
CREATE INDEX idx_lang_glotto ON language_codes(glottocode);
```

### 2.4 lang/{iso}.db Schema (per-language)

```sql
CREATE TABLE concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL UNIQUE,          -- Full genomic notation
    lemma TEXT NOT NULL,
    gloss TEXT,
    pos TEXT,                               -- Part of speech
    abstraction INTEGER NOT NULL,           -- 1=primitive, 2+=compositions
    fingerprint INTEGER,
    UNIQUE(lemma, pos)
);

CREATE TABLE surface_forms (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    form TEXT NOT NULL,
    form_type TEXT,                         -- 'lemma', 'inflected', 'variant'
    features TEXT,                          -- JSON: {"tense": "past", "number": "plural"}
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    primitive_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    relation TEXT DEFAULT 'part',
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE TABLE dialect_status (
    concept_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,                   -- 'shared', 'divergent', 'unique'
    divergent_dialects TEXT,                -- JSON: [1, 2, 3] for divergent terms
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE INDEX idx_surface_form ON surface_forms(form);
CREATE INDEX idx_concept_lemma ON concepts(lemma);
CREATE INDEX idx_concept_abstraction ON concepts(abstraction);
```

---

## 3. Data Sources

### 3.1 Available in reference/

| Source | Location | Format | Purpose |
|--------|----------|--------|---------|
| Glottolog | `reference/glottolog/` | CLDF (csv) | Language tree → FF.SS.LLL |
| WALS | `reference/wals/` | CLDF (csv) | Typological features |
| Grambank | `reference/grambank/` | CLDF (csv) | Grammar features |
| Kaikki | `reference/kaikki/` | JSONL | Wiktionary extracts |

### 3.2 In data_tarballs/ (need unpacking)

| Source | Archive | Purpose |
|--------|---------|---------|
| VerbNet | verbnet*.tar.gz | Verb semantic classes |
| WordNet | wn*.tar.gz | Synset structure |
| OMW | omw*.tar.gz | Multilingual wordnet |

### 3.3 Finding Data Files

Scripts must search for files:
```python
def find_data_file(base_dir, pattern):
    """Find data file matching pattern, searching recursively."""
    matches = list(Path(base_dir).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {base_dir}")
    # Return most recently modified if multiple
    return max(matches, key=lambda p: p.stat().st_mtime)
```

---

## 4. Import Process

### Phase 1: Foundation

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `init_schemas.py` | None | Create empty DBs with schemas | DONE |
| `import_nsm_primes.py` | Hardcoded data | primitives.db | DONE |
| `import_glottolog.py` | reference/glottolog/ | language_registry.db | DONE |
| `import_wals.py` | reference/wals/ | language_registry.db (features) | DONE |
| `import_grambank.py` | reference/grambank/ | language_registry.db (features) | DONE |

### Phase 2: Extended Primitives

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `unpack_tarballs.py` | data_tarballs/ | reference/{verbnet,wordnet,omw}/ | TODO |
| `import_verbnet.py` | reference/verbnet/ | primitives.db | TODO |
| `import_wordnet.py` | reference/wordnet/ | primitives.db + synset mapping | TODO |

### Phase 3: Multilingual

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `import_omw.py` | reference/omw/ | primitives.db (forms) | TODO |

### Phase 4: Lexical Data

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `import_kaikki.py` | reference/kaikki/en*.jsonl | db/lang/eng.db | TODO |

### Phase 5: Composition Analysis

| Script | Input | Output | Status |
|--------|-------|--------|--------|
| `compute_compositions.py` | All DBs | Updates lang/*.db with fingerprints | TODO |
| `build_master_index.py` | All DBs | Populates token_index in each | TODO |

---

## 5. Layering Process (Progressive Abstraction)

**This is NOT import phases.** Layering is semantic resolution.

```
Abstraction 1: Primitives          KNOW, MOVE, GOOD...
              ↓
Abstraction 2: Direct compositions comprehend = KNOW + INSIDE + COMPLETE
              ↓
Abstraction 3: Compositions²       philosophy = KNOW + LOVE + WISDOM
              ↓
Abstraction N: Complex concepts
```

### Algorithm

1. Map primitive surface forms (abstraction 1)
2. Analyze glosses for primitive references
3. If all components resolved → compute fingerprint, assign abstraction N+1
4. Iterate until no more progress
5. Flag unresolved for review

---

## 6. Core/Delta Structure

### Dialect 0 = Core (Shared Truth)

**Non-divergent terms:**
- Full entry in core (dialect 0)
- Available everywhere

**Divergent terms:**
- Core has NO token/meaning entry
- Core only has reference list pointing to deltas
- Each delta has full token with own fingerprint

Example for "biscuit":
```
Core (eng.db, dialect=0):
  surface_forms: "biscuit" exists
  dialect_status: divergent, dialects=[1, 2, 3]
  concepts: NO entry for biscuit meaning

Delta 1 (GB): "biscuit" → token ...1.8.127.1... → COOKIE_SWEET
Delta 2 (US): "biscuit" → token ...1.8.127.2... → BREAD_ROLL
Delta 3 (CA): BOTH mappings with own tokens
```

---

## 7. Detection System

### Multi-Signal Detection

1. Character script → narrow to script family
2. N-gram frequency → statistical fingerprint
3. Surface form matches → known vocabulary
4. Typological features → grammar patterns (WALS/Grambank)
5. Dialect markers → vocabulary-based dialect ID

---

## 8. Genomic Notation Reference

```
Format: A.D.C.FF.SS.LLL.DD.FP.COL

A   = Abstraction level (1-99)
D   = Domain (1-99)
C   = Category (1-99)
FF  = Language Family (0-99)
SS  = Subfamily (0-99)
LLL = Language (0-999)
DD  = Dialect (0-999, 0=core)
FP  = Fingerprint (0-999999)
COL = Collision (0-999)

Example: 2.3.7.1.8.127.0.248.0
         │ │ │ │ │ │   │ │   └─ Collision 0
         │ │ │ │ │ │   │ └───── Fingerprint 248
         │ │ │ │ │ │   └─────── Dialect 0 (core)
         │ │ │ │ │ └─────────── Language 127 (English)
         │ │ │ │ └───────────── Subfamily 8 (Germanic)
         │ │ │ └─────────────── Family 1 (Indo-European)
         │ │ └───────────────── Category 7 (understanding)
         │ └─────────────────── Domain 3 (mental)
         └───────────────────── Abstraction 2
```

---

## 9. Execution Order

```bash
# Run from /usr/share/databases/scripts/ with venv activated

# Phase 1: Foundation
python init_schemas.py
python import_nsm_primes.py
python import_glottolog.py
python import_wals.py
python import_grambank.py

# Phase 2: Extended primitives
python unpack_tarballs.py
python import_verbnet.py
python import_wordnet.py

# Phase 3: Multilingual
python import_omw.py

# Phase 4: Lexical (English first)
python import_kaikki.py --lang en

# Phase 5: Analysis
python compute_compositions.py --lang en
python build_master_index.py
```

---

## 10. Script Checklist

All scripts go in `/home/user/Emergence/research/tokenizers/scripts/` (symlinked to frankenputer).

| Script | Purpose | Status |
|--------|---------|--------|
| `init_schemas.py` | Create empty DBs with all schemas | DONE |
| `import_nsm_primes.py` | 65 NSM semantic primes | DONE |
| `import_glottolog.py` | Language tree → FF.SS.LLL codes | DONE |
| `import_wals.py` | Typological features | DONE |
| `import_grambank.py` | Grammar features | DONE |
| `unpack_tarballs.py` | Extract VerbNet, WordNet, OMW | TODO |
| `import_verbnet.py` | Verb semantic classes | TODO |
| `import_wordnet.py` | Synset structure | TODO |
| `import_omw.py` | Multilingual forms | TODO |
| `import_kaikki.py` | Wiktionary vocabulary | TODO |
| `compute_compositions.py` | Primitive decomposition | TODO |
| `build_master_index.py` | Aggregate token_index | TODO |
| `setup.sh` | Full pipeline orchestrator | DONE |
