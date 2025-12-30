# Concept Tokenizer Database Population Pipeline

> **Updated:** Full multilingual support with Kaikki.org + Glottolog/WALS/Grambank

## Architecture Overview

```
Typology Sources                          Language DB
┌─────────────────────────────────┐       ┌──────────────────────────────┐
│  reference/                     │       │  db/language.db              │
│  ├── glottolog/  (lang tree)    │──┐    │                              │
│  ├── wals/       (typology)     │──┼───▶│  LANGUAGE LAYER:             │
│  └── grambank/   (grammar)      │──┘    │  ├── languages               │
└─────────────────────────────────┘       │  ├── language_families       │
                                          │  ├── feature_definitions     │
Lexical Source                            │  └── language_features       │
┌─────────────────────────────────┐       │                              │
│  reference/kaikki/              │       │  LEXICAL LAYER:              │
│  ├── english.jsonl (~1GB)       │──────▶│  ├── synsets                 │
│  ├── german.jsonl               │       │  ├── concepts                │
│  ├── [downloaded as needed]     │       │  ├── surface_forms           │
│  └── ...                        │       │  ├── modifiers               │
└─────────────────────────────────┘       │  └── translations            │
        ▲                                 └──────────────────────────────┘
        │                                            │
        └── downloads driven by ──────────────────────┘
            languages in DB

                                          lib/concept_tokenizer.py
                                          (The DB IS the tokenizer)
```

## Data Sources

| Order | Source | Purpose | Size | Format |
|-------|--------|---------|------|--------|
| 1 | Glottolog | Language inventory, ISO codes, genealogy | ~50MB | CLDF CSV |
| 2 | WALS | Typological features (word order, morphology) | ~10MB | CLDF CSV |
| 3 | Grambank | 195 grammatical features per language | ~20MB | CLDF CSV |
| 4 | Kaikki.org | Words, senses, forms, translations | ~1GB/lang | JSONL |

**Glottolog** provides the map of all languages - what exists, how they're related.
**WALS + Grambank** tell us HOW each language structures thought (SVO vs SOV, case marking, etc.)
**Kaikki.org** gives us the actual vocabulary in each language.

## Directory Structure

```
research/tokenizers/
├── db/
│   ├── language.db               # Language DB (all tables)
│   └── SCHEMA-language-db.sql    # Schema definition
├── lib/
│   ├── __init__.py
│   └── concept_tokenizer.py      # Tokenizer interface
├── reference/
│   ├── glottolog/                # Glottolog CLDF
│   ├── wals/                     # WALS CLDF
│   ├── grambank/                 # Grambank CLDF
│   └── kaikki/                   # Kaikki JSONL per language
│       ├── english.jsonl
│       ├── german.jsonl
│       └── manifest.json
├── scripts/
│   ├── download_typology.py      # Download Glottolog/WALS/Grambank
│   ├── import_glottolog.py       # Import language tree
│   ├── import_wals.py            # Import WALS features
│   ├── import_grambank.py        # Import Grambank features
│   ├── download_kaikki.py        # Download lexical data (language-aware)
│   └── import_kaikki.py          # Import lexical data
└── NOTES-language-db-architecture.md
```

## Token ID Structure

```
Token ID Ranges:
  0-999:         Reserved/System tokens
  1000-2999:     Modifier tokens (grammar)
  3000000+:      Concept tokens (synset-based)

Synset token calculation:
  token_id = 3,000,000 + (synset_id × 128) + concept_offset

128 slots per synset allows additions without renumbering.
```

## Pipeline Steps

### 1. Download Typology Sources (once)

```bash
cd research/tokenizers
python scripts/download_typology.py           # Downloads Glottolog, WALS, Grambank
python scripts/download_typology.py --source glottolog  # Individual source
```

### 2. Import Typology Data (once)

```bash
python scripts/import_glottolog.py            # Languages + families
python scripts/import_wals.py                 # Typological features
python scripts/import_grambank.py             # Grammar features
```

### 3. Download Lexical Data (as needed)

```bash
python scripts/download_kaikki.py             # English only
python scripts/download_kaikki.py --from-db   # All languages in database
python scripts/download_kaikki.py --from-db --limit 20  # Top 20 by speakers
python scripts/download_kaikki.py --from-db --missing-only  # Only missing
```

### 4. Import Lexical Data

```bash
python scripts/import_kaikki.py               # Import English
python scripts/import_kaikki.py --input reference/kaikki/german.jsonl
```

### 5. Use the Tokenizer

```python
from lib import ConceptTokenizer, tokenize

# One-shot
tokens = tokenize("The cat is running")
# [CONCEPT:the, CONCEPT:cat, CONCEPT:be, CONCEPT:run, MODIFIER:PROG]

# With caching and language features
tokenizer = ConceptTokenizer()

# Query language features for processing
tokenizer.conn.execute('''
    SELECT value FROM language_features
    WHERE lang_code = 'deu' AND feature_id = '81A'
''')  # → SVO (German word order)

tokenizer.prefetch_synset(123)  # Warm cache
tokens = tokenizer.surface_to_tokens("running quickly")
text = tokenizer.tokens_to_surface(tokens)
tokenizer.close()
```

## Schema Tables

### Language Layer

| Table | Purpose | Est. Rows |
|-------|---------|-----------|
| languages | Language inventory (ISO codes, names, families) | ~8000 |
| language_families | Genealogical hierarchy | ~500 |
| feature_definitions | What each feature means (WALS/Grambank) | ~400 |
| language_features | Feature values per language | ~500k |

### Lexical Layer

| Table | Purpose | Est. Rows |
|-------|---------|-----------|
| synsets | Concept clusters with glosses | ~100-150k |
| concepts | Individual concepts in synsets | ~150-200k |
| surface_forms | All word variants | ~500k+ |
| modifiers | Grammar tokens (tense, case, etc.) | ~50 |
| translations | Cross-language links | ~1M+ |
| synset_relations | Hypernym, meronym, etc. | ~200k |

## Key Design Principles

1. **The DB IS the tokenizer** - every word is a query
2. **Language-aware** - typology data guides processing per language
3. **Synsets as token groups** - related concepts have nearby IDs
4. **128 slots per synset** - room for additions without renumbering
5. **Demand-driven downloads** - Kaikki data fetched when language identified
6. **Efficient for spinners** - synset batching = sequential reads

## Example: Multilingual Expression

To express a concept in all known languages (for weight draining):

```python
# Get concept
concept = tokenizer.lookup_concept("run")  # → (token_id, synset_id, gloss)

# Get all translations
translations = tokenizer.conn.execute('''
    SELECT t.target_lang, t.translation, l.name, lf.value
    FROM translations t
    JOIN languages l ON t.target_lang = l.lang_code
    LEFT JOIN language_features lf ON l.lang_code = lf.lang_code AND lf.feature_id = '81A'
    WHERE t.concept_id = ?
''', (concept[0],))

# For each language, structure according to its typology
for lang_code, word, lang_name, word_order in translations:
    # Use word_order to compose properly
    ...
```

## Hardware Target

- 16GB RAM
- HDD (spinner drives)
- Ubuntu headless

Design optimizes for:
- Sequential reads (synset batching)
- Working set in RAM (~10-20% of total)
- Cold fetch on demand
- Language-specific data loaded lazily
