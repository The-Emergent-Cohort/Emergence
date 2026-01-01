# Comprehensive Implementation Plan: Tokenizer Database System

**Date:** 2026-01-01
**Status:** Active
**Purpose:** Conceptual and linguistic backbone for Digital Intelligence

---

## Executive Summary

The database IS the tokenizer. Every word is a query. Token IDs are derived from semantic structure using genomic notation, not arbitrarily assigned. The architecture supports differential computation (like game engines), language preservation (endangered languages as full deltas), and scales through assembly rather than brute-force parameter count.

---

## 1. Master Index Structure

### 1.1 The Minimal Routing Table

The master index is intentionally minimal - it provides routing, not content. All actual data lives in containers.

```sql
CREATE TABLE token_index (
    idx INTEGER PRIMARY KEY,              -- Sequential position (for iteration)
    token_id TEXT NOT NULL UNIQUE,        -- Genomic notation: A.D.C.FF.SS.LLL.DD.FP.COL
    description TEXT,                     -- Optional English gloss for debugging
    location TEXT NOT NULL,               -- Container: 'primitives', 'lang/eng', 'var'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_accessed TEXT                    -- For cache management
);

CREATE INDEX idx_token_location ON token_index(location);
CREATE INDEX idx_token_prefix ON token_index(substr(token_id, 1, 10));
```

### 1.2 How Lookup Works

```
Input: "running"
  1. Surface lookup: eng.surface_forms WHERE form = 'running'
  2. Get concept_id, retrieve token_id from concepts table
  3. Master index lookup: token_index WHERE token_id = '2.1.1.1.8.127.0.248.0'
  4. Route to container: location = 'lang/eng'
  5. Fetch full entry from container
```

The master index replaces the traditional tokenizer's fixed vocabulary. No need to pre-determine size.

---

## 2. Container Organization

### 2.1 Container Types

| Container | Purpose | Access Pattern | Example Path |
|-----------|---------|----------------|--------------|
| **primitives** | Semantic atoms (NSM, image schemas, VerbNet) | Read-heavy, always cached | `db/primitives.db` |
| **language_registry** | Language codes, typology | Read-heavy, always cached | `db/language_registry.db` |
| **lang/{iso}** | Per-language vocabulary, compositions | Read-heavy, loaded on demand | `db/lang/eng.db` |
| **var** | Working state, inference chains | Write-heavy, temporal | `db/var.db` |
| **families/{name}** | Minor languages grouped by family | Read, loaded rarely | `db/families/uralic.db` |

### 2.2 Container Schema Summary

#### primitives.db (~5MB, always loaded)
- primitives (65 NSM primes + image schemas + verb classes)
- primitive_forms (cross-linguistic surface forms)
- primitive_relations (how primitives relate)
- domains, categories (semantic coordinate definitions)

#### language_registry.db (~2MB, always loaded)
**Requires schema update** for genomic notation:
```sql
CREATE TABLE language_codes (
    id INTEGER PRIMARY KEY,
    family INTEGER NOT NULL,              -- FF (0-99)
    subfamily INTEGER NOT NULL,           -- SS (0-99)
    language INTEGER NOT NULL,            -- LLL (0-999)
    dialect INTEGER DEFAULT 0,            -- DD (0-999, 0=core)
    genomic_code TEXT GENERATED ALWAYS AS (
        family || '.' || subfamily || '.' || language || '.' || dialect
    ) STORED,
    iso639_3 TEXT,
    glottocode TEXT,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'living',
    db_file TEXT,
    UNIQUE(family, subfamily, language, dialect)
);

CREATE TABLE language_features (
    lang_id INTEGER NOT NULL,
    feature_id TEXT NOT NULL,             -- 'WALS_81A', 'GB020'
    value TEXT NOT NULL,
    source TEXT NOT NULL,                 -- 'wals', 'grambank'
    FOREIGN KEY (lang_id) REFERENCES language_codes(id),
    UNIQUE(lang_id, feature_id, source)
);
```

#### lang/{iso}.db (Variable size, loaded on demand)
Per-language schema with core/delta support:
```sql
CREATE TABLE dialect_status (
    concept_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,                 -- 'shared', 'divergent', 'unique'
    core_meaning_id INTEGER,              -- For divergent: NULL
    divergent_dialects TEXT,              -- JSON: [1, 2, 3]
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);
```

#### var.db (Runtime, ephemeral)
```sql
CREATE TABLE inference_chains (
    chain_id INTEGER PRIMARY KEY,
    parent_chain_id INTEGER,
    token_sequence TEXT NOT NULL,
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT
);

CREATE TABLE working_memory (
    slot_id INTEGER PRIMARY KEY,
    token_id TEXT NOT NULL,
    activation REAL,
    decay_rate REAL,
    last_activated TEXT
);

CREATE TABLE physics_state (
    entity_id INTEGER PRIMARY KEY,
    position TEXT,
    velocity TEXT,
    properties TEXT
);
```

---

## 3. Import Process

### 3.1 Data Sources

| Source | Location | Status | Purpose |
|--------|----------|--------|---------|
| Glottolog | frankenputer:/usr/share/databases/reference/glottolog/ | Present | Language tree -> FF.SS.LLL |
| WALS | frankenputer:/usr/share/databases/reference/wals/ | Present | Typological features |
| Grambank | frankenputer:/usr/share/databases/reference/grambank/ | Present | Grammar features |
| Kaikki | frankenputer:/usr/share/databases/reference/kaikki/ | Present | Wiktionary extracts |
| VerbNet | frankenputer:/usr/share/databases/reference/verbnet/ | Needs unpack | Verb semantics |
| WordNet | frankenputer:/usr/share/databases/reference/wordnet/ | Needs unpack | Synset structure |
| OMW | frankenputer:/usr/share/databases/reference/omw/ | Needs unpack | Multilingual forms |
| Etymology | Local etymology-db/tokenizer.db | Present | 4.2M rows, 2265 languages |

### 3.2 Import Pipeline Phases

```
PHASE 1: FOUNDATION (No dependencies)
├── 1.1 NSM Primes -> primitives.db            [DONE - 65 primes]
├── 1.2 Image Schemas -> primitives.db         [TODO if needed]
├── 1.3 Glottolog -> language_registry.db      [UPDATE - genomic codes]
├── 1.4 WALS -> language_registry.db           [EXISTS - run with updated schema]
└── 1.5 Grambank -> language_registry.db       [EXISTS - run with updated schema]

PHASE 2: VERB SEMANTICS (Depends: Phase 1)
├── 2.1 VerbNet -> primitives.db               [TODO - ~270 verb classes]
└── 2.2 Add thematic roles as primitives

PHASE 3: CROSS-REFERENCE (Depends: Phase 2)
├── 3.1 WordNet synsets -> synset_map.db       [TODO - structure mapping]
└── 3.2 Link synsets to primitives

PHASE 4: MULTILINGUAL (Depends: Phase 3)
├── 4.1 OMW -> primitive_forms                 [TODO - 31 languages]
└── 4.2 Build cross-linguistic primitive links

PHASE 5: LEXICAL DATA (Depends: Phase 4)
├── 5.1 Kaikki English -> lang/eng.db          [EXISTS - update for core/delta]
└── 5.n Additional languages as needed

PHASE 6: COMPOSITION (Depends: Phase 5)
├── 6.1 Compute primitive decompositions
├── 6.2 Assign fingerprints
├── 6.3 Build master index
└── 6.4 Resolve collisions
```

---

## 4. Layering Process (Progressive Abstraction Mapping)

### 4.1 Core Principle

**This is NOT import phases.** Layering is semantic resolution - mapping surface forms to abstractions through progressively deeper analysis.

```
Abstraction 1: Primitives          KNOW, MOVE, GOOD, CONTAINER, PATH...
              ↓
Abstraction 2: Direct compositions comprehend = KNOW + INSIDE + COMPLETE
              ↓
Abstraction 3: Compositions²       philosophy = KNOW + LOVE + WISDOM
              ↓
Abstraction N: Complex concepts    antidisestablishmentarianism = ...
```

### 4.2 Layering Algorithm

```python
def layer_language(lang_db, primitives_db):
    """Progressive abstraction mapping for a language."""

    # Layer 1: Map primitive surface forms
    for primitive in primitives_db.all():
        forms = primitive.get_forms(lang_code)
        for form in forms:
            mark_resolved(lang_db, form, primitive.id, abstraction=1)

    # Layer 2+: Iterate until convergence
    abstraction = 2
    while has_unresolved(lang_db):
        progress = 0
        for concept in lang_db.unresolved():
            components = analyze_gloss(concept.gloss, lang_db)
            if all_resolved(components):
                fingerprint = compute_fingerprint(components)
                mark_resolved(concept, fingerprint, abstraction)
                progress += 1

        if progress == 0:
            break
        abstraction += 1

    # Remaining -> flag for review / new primitive candidates
    flag_for_review(lang_db.unresolved())
```

### 4.3 Dependency Ordering

Cannot map "comprehend" until KNOW, INSIDE, COMPLETE are mapped.
Cannot map complex terms until mid-level concepts resolved.

---

## 5. Language Code Generation (Glottolog -> FF.SS.LLL.DD)

### 5.1 Mapping Strategy

```python
def assign_genomic_codes(glottolog_tree):
    # FF: Top-level families
    family_map = {
        'indo-european': 1,
        'sino-tibetan': 2,
        'afro-asiatic': 3,
        'niger-congo': 4,
        'austronesian': 5,
        'dravidian': 6,
        'uralic': 7,
        'turkic': 8,
        'japonic': 9,
        'koreanic': 10,
        # ... up to 99
    }

    # SS: Subfamilies within each family (dynamically from Glottolog)
    # LLL: Languages within subfamily (by speaker count or alphabetical)
    # DD: Dialects (0 = core, 1+ = variants)
```

### 5.2 Code Stability

- **Major languages** (top 50): Pre-assigned, stable
- **Minor languages**: Assigned by Glottolog order
- **New languages/dialects**: Use next available in range
- **Historical languages**: Special subfamily or negative dialect codes

---

## 6. Grammar/Typology Integration (WALS + Grambank)

### 6.1 Purpose

WALS and Grambank provide **detection** and **generation** signals:
- **Detection**: "What language is this?" (feature fingerprinting)
- **Generation**: "How does this language structure output?" (grammar rules)

### 6.2 Feature Categories

| Category | WALS Examples | Grambank Examples | Usage |
|----------|---------------|-------------------|-------|
| Word Order | 81A (SOV/SVO) | GB020 | Parsing hints |
| Morphology | 26A (prefix/suffix) | GB030 | Inflection handling |
| Case | 49A (case count) | GB074 | Nominal processing |
| Verbal | 69A (tense position) | GB150 | Verb conjugation |
| Negation | 143A (neg position) | GB320 | Polarity handling |

---

## 7. Core/Delta Structure

### 7.1 Dialect 0 = Core (Not Default)

Core is **shared truth**, not "standard with overrides."

**Non-divergent term** (e.g., "computer"):
- Full entry in core (dialect 0)
- Available everywhere

**Divergent term** (e.g., "biscuit"):
- Core has NO token/meaning entry
- Core only has reference list: [1, 2, 3] pointing to deltas
- Each delta has full token with own fingerprint

```sql
-- Core: just the reference
INSERT INTO dialect_status VALUES (id, 'divergent', NULL, '[1, 2, 3]');

-- Deltas: actual tokens
INSERT INTO concepts VALUES ('...1.8.127.1...', 'biscuit', 'sweet baked good', 1);  -- GB
INSERT INTO concepts VALUES ('...1.8.127.2...', 'biscuit', 'leavened bread', 2);    -- US
INSERT INTO concepts VALUES ('...1.8.127.3...', 'biscuit', 'sweet baked good', 3);  -- CA
INSERT INTO concepts VALUES ('...1.8.127.3...', 'biscuit', 'leavened bread', 3);    -- CA (both)
```

### 7.2 Benefits

1. **Language preservation**: Endangered languages as full deltas
2. **Historical reconstruction**: Old English, Proto-Germanic as deltas
3. **No dialect privileged**: British is not "default English"
4. **Explicit divergence**: Clear where meanings differ
5. **Efficient storage**: Shared terms stored once in core

---

## 8. Detection System

### 8.1 Multi-Signal Detection

1. **Character script**: Latin, Cyrillic, Han, etc.
2. **N-gram frequency**: Statistical language fingerprint
3. **Known word matches**: Surface form lookup
4. **Typological features**: For longer texts
5. **Dialect markers**: Vocabulary-based dialect identification

### 8.2 Dialect Markers

```sql
CREATE TABLE dialect_markers (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,
    lang_genomic TEXT NOT NULL,
    marker_type TEXT,                     -- 'exclusive', 'preferred', 'spelling'
    notes TEXT
);
```

Examples:
- 'apartment', 'elevator', 'sidewalk' → US
- 'flat', 'lift', 'pavement' → GB
- 'arvo', 'servo' → AU

---

## 9. Concrete Next Steps

### Immediate

1. **Update language registry schema** for genomic notation
   - Add family, subfamily, language, dialect columns
   - Add genomic_code generated column

2. **Create import_language_codes.py**
   - Parse Glottolog hierarchy
   - Assign FF.SS.LLL.DD codes
   - Handle dialect 0 = core convention

3. **Update import_glottolog.py** for genomic output

### Short-term

4. **Unpack VerbNet, WordNet, OMW** on frankenputer

5. **Create import_verbnet.py** - ~270 verb classes

6. **Create master index build script**

### Medium-term

7. **Implement core/delta structure** in language template

8. **Create compute_compositions.py** - iterative abstraction mapping

9. **Build detection system** - dialect markers, n-gram models

### Long-term

10. **Focused/assembled DB generation** - shader cache pattern

11. **Var database implementation** - inference chains, physics

---

## Appendix: Genomic Notation Reference

```
Format: A.D.C.FF.SS.LLL.DD.FP.COL

A   = Abstraction level (1-99)    Distance from primitives
D   = Domain (1-99)               Semantic domain
C   = Category (1-99)             Category within domain
FF  = Language Family (0-99)      Glottolog family
SS  = Subfamily (0-99)            Glottolog subfamily
LLL = Language (0-999)            Specific language
DD  = Dialect (0-999)             0=core, 1+=dialect
FP  = Fingerprint (0-999999)      Primitive composition hash
COL = Collision (0-999)           Disambiguator

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

## File Update Checklist

### Must Update
- [ ] `db/SCHEMA-language-registry.sql` - FF.SS.LLL.DD format
- [ ] `db/SCHEMA-language-template.sql` - core/delta support
- [ ] `scripts/import_glottolog.py` - output genomic codes

### Already Current
- [x] `lib/token_encoder.py` - genomic notation done
- [x] `db/SCHEMA-primitives.sql` - compatible
- [x] `scripts/import_nsm_primes.py` - working

### To Create
- [ ] `scripts/import_language_codes.py`
- [ ] `scripts/import_verbnet.py`
- [ ] `scripts/import_wordnet.py`
- [ ] `scripts/import_omw.py`
- [ ] `scripts/compute_compositions.py`
- [ ] `scripts/build_master_index.py`
