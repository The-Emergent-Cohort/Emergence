# Concept Tokenizer Database Population Pipeline

> **Updated:** Simplified to use Kaikki.org as primary source (replaces WordNet, Wikidata Lexemes, UniMorph, MorphoLex)

## Architecture Overview

```
Kaikki.org (Wiktionary extraction)        Language DB (synset-based)
┌─────────────────────────────────┐       ┌──────────────────────────────┐
│  reference/kaikki/              │       │  db/language.db              │
│  ├── english.jsonl (~1GB)       │       │  ├── synsets                 │
│  ├── german.jsonl               │ ──────▶│  ├── concepts                │
│  ├── french.jsonl               │ import │  ├── surface_forms           │
│  └── ...                        │       │  ├── modifiers               │
└─────────────────────────────────┘       │  └── translations            │
                                          └──────────────────────────────┘
Weekly updates from kaikki.org                      │
                                                    ▼
                                          lib/concept_tokenizer.py
                                          (The DB IS the tokenizer)
```

## Data Source

| Source | URL | Format | Size | Content |
|--------|-----|--------|------|---------|
| Kaikki.org | kaikki.org/dictionary | JSONL | ~1GB/lang | Definitions, etymology, forms, translations |

Kaikki.org provides weekly Wiktionary extractions that include:
- **Definitions/glosses** (grouped by sense)
- **Etymology** (word origins)
- **Inflected forms** (conjugations, declensions)
- **Translations** (cross-language links)
- **Pronunciations** (IPA)

This single source replaces:
- ~~WordNet~~ → senses from Wiktionary definitions
- ~~Wikidata Lexemes~~ → multilingual data from Wiktionary
- ~~UniMorph~~ → inflection forms in Kaikki data
- ~~MorphoLex~~ → morpheme info in etymology/forms

## Directory Structure

```
research/tokenizers/
├── db/
│   ├── language.db               # Language DB (synset-based)
│   └── SCHEMA-language-db.sql    # Schema definition
├── lib/
│   ├── __init__.py
│   └── concept_tokenizer.py      # Tokenizer interface
├── reference/
│   └── kaikki/
│       ├── english.jsonl         # Downloaded Kaikki data
│       ├── german.jsonl
│       └── manifest.json
├── scripts/
│   ├── download_kaikki.py        # Download from kaikki.org
│   └── import_kaikki.py          # Stream JSONL → SQLite
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

This gives 128 slots per synset for related concepts,
with room for additions without renumbering.
```

## Pipeline Steps

### 1. Download

```bash
cd research/tokenizers
python scripts/download_kaikki.py                    # English only
python scripts/download_kaikki.py --lang German French Spanish
python scripts/download_kaikki.py --all              # All common languages
```

### 2. Import

```bash
python scripts/import_kaikki.py                      # Default: english.jsonl → language.db
python scripts/import_kaikki.py --input reference/kaikki/german.jsonl
python scripts/import_kaikki.py --limit 10000        # Test with subset
```

### 3. Use

```python
from lib import ConceptTokenizer, tokenize

# One-shot
tokens = tokenize("The cat is running")
print(tokens)
# [CONCEPT:the, CONCEPT:cat, CONCEPT:be, CONCEPT:run, MODIFIER:PROG]

# With caching
tokenizer = ConceptTokenizer()
tokenizer.prefetch_synset(123)  # Warm cache for related concepts
tokens = tokenizer.surface_to_tokens("running quickly")
text = tokenizer.tokens_to_surface(tokens)
tokenizer.close()
```

## Schema Tables

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
2. **Synsets as token groups** - related concepts have nearby IDs
3. **128 slots per synset** - room for additions without renumbering
4. **Efficient for spinners** - synset batching = sequential reads
5. **Lazy loading** - fetch on demand, cache hot concepts
6. **Modifiers separate** - grammar tokens in fixed range (1000-2999)

## Python Dependencies

```
# No external dependencies beyond stdlib for core pipeline
# Optional for progress bars:
tqdm>=4.64.0
```

## Hardware Target

- 16GB RAM
- HDD (spinner drives)
- Ubuntu headless

Design optimizes for:
- Sequential reads (synset batching)
- Working set in RAM (~10-20% of total)
- Cold fetch on demand
