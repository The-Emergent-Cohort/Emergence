# Tokenizer Database Working Plan

**Date:** 2026-01-01
**Status:** Active
**Previous:** PLAN-import-pipeline.md (partially stale)

---

## Current State Summary

### Working Components
| Component | Status | Notes |
|-----------|--------|-------|
| primitives.db | DONE | 65 NSM primes, domain/category seeded |
| etymology.db | DONE | 4.2M rows, 2265 languages (in etymology-db/) |
| token_encoder.py | DONE | Genomic notation: A.D.C.FF.SS.LLL.DD.FP.COL |
| import_nsm_primes.py | DONE | Imports 65 NSM primes |
| import_glottolog.py | EXISTS | Language tree import |
| import_wals.py | EXISTS | Typological features |
| import_grambank.py | EXISTS | Grammar features |
| import_kaikki.py | EXISTS | Dictionary import |
| download_*.py | EXISTS | Download scripts |

### Missing Scripts
| Script | Priority | Complexity |
|--------|----------|------------|
| import_image_schemas.py | P1 | Low (hardcoded list) |
| import_language_codes.py | P1 | Medium (FF.SS.LLL.DD format) |
| import_verbnet.py | P2 | Medium (XML parsing) |
| import_wordnet.py | P2 | Medium (WordNet format) |
| import_omw.py | P3 | Medium (31 languages) |
| compute_compositions.py | P3 | High (semantic analysis) |
| build_all.py | P4 | Low (orchestrator) |

---

## Architecture Updates from 2026-01-01

### 1. Genomic Notation (DONE in token_encoder.py)
```
A.D.C.FF.SS.LLL.DD.FP.COL

A   = Abstraction level (1-99)
D   = Domain (1-99)
C   = Category (1-99)
FF  = Family (0-99)
SS  = Subfamily (0-99)
LLL = Language (0-999)
DD  = Dialect (0-999, where 0 = core)
FP  = Fingerprint (0-999999)
COL = Collision (0-999)
```

### 2. Dialect 0 = Core
- Dialect 0 is the canonical reference, not a variant
- Dialect 1+ are delta overlays
- Core contains:
  - Full entries for shared/unambiguous tokens
  - Token + delta references for divergent meanings

Example divergent entry:
```
core "biscuit":
  status: divergent
  meanings:
    - delta.1 (GB): COOKIE_SWEET
    - delta.2 (US): BREAD_ROLL_LEAVENED
```

### 3. Common/Delta Structure
- Reference DBs = source of truth (primitives.db, lang/eng.db, etc.)
- Focused DBs = compiled artifacts for specific tasks
- Like shader cache: pay prep cost once, reuse until context changes

### 4. Var Database (DESIGN ONLY)
Separate database for working state:
- Inference chains
- Physics data
- Temporary structures
- Different access patterns (write-heavy, temporal)

---

## Phase 1: Schema Alignment (BLOCKING)

### 1.1 Update SCHEMA-language-registry.sql
Current schema uses flat 4-digit codes. Needs update for FF.SS.LLL.DD structure.

**Changes needed:**
```sql
-- CURRENT (flat)
CREATE TABLE language_codes (
    code INTEGER PRIMARY KEY,  -- 0000-9999
    ...
);

-- UPDATED (genomic)
CREATE TABLE language_codes (
    id INTEGER PRIMARY KEY,
    family INTEGER NOT NULL,      -- FF (0-99)
    subfamily INTEGER NOT NULL,   -- SS (0-99)
    language INTEGER NOT NULL,    -- LLL (0-999)
    dialect INTEGER DEFAULT 0,    -- DD (0-999, 0=core)
    genomic_code TEXT GENERATED ALWAYS AS (
        family || '.' || subfamily || '.' || language || '.' || dialect
    ) STORED,
    ...
    UNIQUE(family, subfamily, language, dialect)
);
```

**Files to update:**
- `db/SCHEMA-language-registry.sql`
- `db/SCHEMA-language-template.sql`
- `scripts/import_glottolog.py`

### 1.2 Add Dialect Delta Support
New table for dialect divergence tracking:

```sql
CREATE TABLE dialect_divergence (
    id INTEGER PRIMARY KEY,
    token_fingerprint INTEGER NOT NULL,  -- FP component
    core_meaning_id INTEGER,             -- NULL if fully divergent
    divergent_dialects TEXT,             -- JSON: [1, 2, 3]
    notes TEXT
);
```

---

## Phase 2: Foundation Primitives (P1)

### 2.1 import_image_schemas.py (CREATE)
~27 embodied image schemas from Lakoff/Johnson:

```python
IMAGE_SCHEMAS = [
    ("CONTAINER", 1, 7, "Bounded region with interior/exterior"),
    ("PATH", 1, 1, "Source-route-goal trajectory"),
    ("FORCE", 1, 4, "Directed energy transfer"),
    ("BALANCE", 1, 5, "Symmetric equilibrium"),
    ("UP_DOWN", 1, 2, "Vertical orientation"),
    ("FRONT_BACK", 1, 2, "Sagittal orientation"),
    ("PART_WHOLE", 5, 6, "Component structure"),
    ("CENTER_PERIPHERY", 1, 2, "Radial organization"),
    ("LINK", 4, 2, "Connection between entities"),
    ("NEAR_FAR", 1, 2, "Proximity scale"),
    ("SCALE", 5, 2, "Ordered magnitude"),
    ("CYCLE", 2, 3, "Recurring pattern"),
    ("ITERATION", 2, 3, "Repeated action"),
    ("SURFACE", 1, 7, "2D extent"),
    ("FULL_EMPTY", 1, 5, "Containment degree"),
    ("MATCHING", 5, 3, "Structural correspondence"),
    ("SUPERIMPOSITION", 1, 2, "Layered position"),
    ("PROCESS", 2, 1, "Temporal unfolding"),
    ("COLLECTION", 5, 1, "Aggregate structure"),
    ("SPLITTING", 1, 4, "Division action"),
    ("MERGING", 1, 4, "Combination action"),
    ("CONTACT", 1, 3, "Surface adjacency"),
    ("REMOVAL", 1, 4, "Extraction from container"),
    ("MASS_COUNT", 5, 1, "Individuation"),
    ("ENABLEMENT", 5, 4, "Potential activation"),
    ("ATTRACTION", 1, 1, "Force toward"),
    ("BLOCKAGE", 1, 1, "Force opposition"),
]
```

**Output:** primitives.db entries with source='image_schema'

### 2.2 import_language_codes.py (CREATE)
Generate FF.SS.LLL.DD codes from Glottolog hierarchy.

**Key logic:**
1. Parse Glottolog family tree
2. Assign family codes (FF) to top-level families
3. Assign subfamily codes (SS) within families
4. Assign language codes (LLL) within subfamilies
5. Dialect 0 = canonical, 1+ = variants

**Output:** language_registry.db with genomic codes

---

## Phase 3: Verb Semantics (P2)

### 3.1 import_verbnet.py (CREATE)
Parse VerbNet 3.4 XML files (~270 verb classes).

**Source:** /usr/share/databases/reference/verbnet/
**Local:** data/verbnet/ (needs extraction)

**Extract:**
- Verb class names as primitives
- Thematic roles (Agent, Patient, Theme, etc.)
- Selectional restrictions
- Syntactic frames

**Output:** primitives.db entries with source='verbnet'

### 3.2 import_wordnet.py (CREATE)
Parse WordNet 3.1 dict files.

**Source:** /usr/share/databases/reference/wordnet/
**Local:** data/wordnet/ (needs extraction)

**Extract:**
- Synset IDs for cross-reference
- Hypernym/hyponym relations
- Synset glosses

**Output:** Synset mapping table for primitive linking

---

## Phase 4: Multilingual Forms (P3)

### 4.1 import_omw.py (CREATE)
Import Open Multilingual Wordnet (31 languages).

**Source:** /usr/share/databases/reference/omw/
**Local:** data/omw/ (needs extraction)

**Process:**
1. Map OMW synsets to our primitives
2. Extract translations as primitive_forms
3. Handle language code mapping

**Output:** primitive_forms entries with source='omw'

---

## Phase 5: Composition Analysis (P3)

### 5.1 compute_compositions.py (CREATE)
Analyze concepts to find primitive decompositions.

**Strategy:**
1. Parse Kaikki glosses for primitive keywords
2. Use WordNet hypernym chains
3. Apply image schema patterns
4. Compute fingerprints

**Output:** compositions table entries in lang/*.db

---

## Phase 6: Orchestration (P4)

### 6.1 build_all.py (CREATE)
Full pipeline orchestrator.

```python
PIPELINE = [
    ("Phase 1: Schema", [
        "python scripts/init_schemas.py",
    ]),
    ("Phase 2: Primitives", [
        "python scripts/import_nsm_primes.py",
        "python scripts/import_image_schemas.py",
    ]),
    ("Phase 3: Language Codes", [
        "python scripts/import_language_codes.py",
        "python scripts/import_glottolog.py",
        "python scripts/import_wals.py",
        "python scripts/import_grambank.py",
    ]),
    ("Phase 4: Verb Semantics", [
        "python scripts/import_verbnet.py",
        "python scripts/import_wordnet.py",
    ]),
    ("Phase 5: Multilingual", [
        "python scripts/import_omw.py",
    ]),
    ("Phase 6: Lexical Data", [
        "python scripts/import_kaikki.py --lang en",
    ]),
    ("Phase 7: Composition", [
        "python scripts/compute_compositions.py",
    ]),
]
```

---

## Immediate Next Steps (Priority Order)

### Step 1: Schema Update (30 min)
Update SCHEMA-language-registry.sql to use FF.SS.LLL.DD format.
- Add family/subfamily/language/dialect columns
- Add genomic_code generated column
- Add dialect_divergence table

### Step 2: import_image_schemas.py (1 hr)
Create script following import_nsm_primes.py pattern.
- Hardcoded list of ~27 schemas
- Insert into primitives table with source='image_schema'
- No external data needed

### Step 3: import_language_codes.py (2 hr)
Create script to generate genomic codes.
- Read Glottolog hierarchy
- Assign FF.SS.LLL codes
- Handle dialect 0 = core convention

### Step 4: Data Extraction (frankenputer)
Extract source archives:
```bash
cd /usr/share/databases/reference

# VerbNet (if not extracted)
# WordNet (if not extracted)
# OMW (if not extracted)
```

### Step 5: import_verbnet.py (2 hr)
Create script to parse VerbNet XML.
- Extract verb classes as primitives
- Domain 1 (physical) categories 1-4 (motion, change, etc.)

---

## File Update Checklist

### Must Update
- [ ] `db/SCHEMA-language-registry.sql` - FF.SS.LLL.DD format
- [ ] `db/SCHEMA-language-template.sql` - genomic code references
- [ ] `scripts/import_glottolog.py` - output genomic codes

### Already Current
- [x] `lib/token_encoder.py` - genomic notation done
- [x] `db/SCHEMA-primitives.sql` - compatible
- [x] `scripts/import_nsm_primes.py` - working

### To Create
- [ ] `scripts/import_image_schemas.py`
- [ ] `scripts/import_language_codes.py`
- [ ] `scripts/import_verbnet.py`
- [ ] `scripts/import_wordnet.py`
- [ ] `scripts/import_omw.py`
- [ ] `scripts/compute_compositions.py`
- [ ] `scripts/build_all.py`

---

## Data Locations

### frankenputer (/usr/share/databases/)
```
reference/
├── glottolog/glottolog-glottolog-cldf-4dbf078/cldf/
├── grambank/grambank-grambank-9e0f341/cldf/
├── wals/cldf-datasets-wals-0f5cd82/cldf/
├── kaikki/                    # Wiktionary extracts
├── verbnet/                   # VerbNet 3.4
├── wordnet/                   # WordNet 3.1
└── omw/                       # Open Multilingual Wordnet
```

### Local (research/tokenizers/)
```
db/
├── primitives.db              # WORKING (65 primes)
├── language_registry.db       # TO CREATE
└── lang/                      # TO CREATE
    ├── eng.db
    └── ...

data/                          # Placeholder dirs
├── kaikki/
├── omw/
├── primitives/
├── verbnet/
└── wordnet/

etymology-db/
├── etymology.csv              # 4.2M rows
└── tokenizer.db               # 80MB, 2265 languages
```

---

## Design Principles (from Architecture Insights)

1. **Data structure efficiency first** - Well-structured on modest hardware beats brute-force on expensive hardware

2. **Differential calculation** - Only compute what changed, like game engines

3. **Dialect 0 = core** - Canonical reference, deltas overlay

4. **Everything is containers** - Primitives, physics, inference - all just entries with token references

5. **Assembly for scaling** - Same sources, different outputs (full/common/minimal)

6. **The database IS the tokenizer** - Every word is a query
