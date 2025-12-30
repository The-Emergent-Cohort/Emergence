# Concept Tokenizer Database Population Pipeline

## Architecture Overview

```
Reference Tier (read-heavy, large)          Working DB (fast, subset)
┌─────────────────────────────────┐         ┌──────────────────────────┐
│  reference/                      │         │  tokenizer.db            │
│  ├── glottolog/                 │         │  ├── morphemes (73)      │
│  ├── unimorph/                  │ ───────▶│  ├── surface_forms       │
│  ├── morpholex/                 │ generate │  ├── language_families   │
│  ├── grambank/                  │         │  ├── grammar_rules       │
│  └── wals/                      │         │  └── ...                  │
└─────────────────────────────────┘         └──────────────────────────┘
                                                      │
                                            Live SPARQL queries
                                                      ▼
                                            Wikidata Lexemes
```

## Data Sources

| Order | Source | Location | Format | Size | Purpose |
|-------|--------|----------|--------|------|---------|
| 1 | Glottolog | Zenodo | CLDF CSV | ~50MB | Language IDs, family hierarchy |
| 2 | WALS | Zenodo | CLDF CSV | ~10MB | Typological defaults (word order) |
| 3 | Grambank | Zenodo | CLDF CSV | ~20MB | Grammar features (195 features) |
| 4 | UniMorph | GitHub | TSV | ~200MB | Inflection paradigms (169 langs) |
| 5 | MorphoLex-en | GitHub | Excel | ~10MB | English morpheme segmentation |
| 6 | Wikidata | SPARQL | JSON | Live | Multilingual glosses, translations |

## Directory Structure

```
research/tokenizers/
├── db/
│   ├── tokenizer.db              # Working DB
│   ├── SCHEMA-*.sql              # Schema definitions
│   └── *.py                      # Import scripts
├── reference/                    # Reference tier (large datasets)
│   ├── glottolog/
│   ├── unimorph/
│   ├── morpholex/
│   ├── grambank/
│   └── wals/
└── scripts/
    ├── setup.sh                  # Master setup script
    ├── download_sources.py       # Download all sources
    ├── import_glottolog.py
    ├── import_unimorph.py
    ├── import_morpholex.py
    ├── import_grambank.py
    ├── import_wals.py
    ├── query_wikidata.py
    └── generate_working_db.py
```

## Schema Mapping

### Glottolog → language_families
- `ID` → `glottolog_code`
- `Name` → `name`
- `ISO639P3code` → `iso_639_3`
- `Family_ID` → `parent_id` (lookup)
- `level` → `level` (family/language/dialect)

### WALS → language_families (updates)
- `81A` → `default_word_order` (SVO, SOV, etc.)
- `26A` → `default_morphology_type`

### Grambank → grammar_rules
- `Language_ID` → `family_id` (lookup via glottocode)
- `Parameter_ID` → `rule_type` + `rule_name`
- `Value` → `abstract_form`

### UniMorph → surface_forms
- Column 1 (lemma) → lookup morpheme_id
- Column 2 (surface) → `surface_form`
- Column 3 (features) → `pos_features`

### MorphoLex-en → surface_forms + morpheme mapping
- `Word` → `surface_form`
- `MorphoLexSegm` → morpheme breakdown
- `SUBTLEX_Frequency` → `frequency`

## Lazy Loading Pattern

Working DB grows on demand:
1. Detect input language
2. Check if in working DB
3. If not: query reference tier, inject tables
4. Process normally

## Python Dependencies

```
requests>=2.28.0
tqdm>=4.64.0
openpyxl>=3.1.0
csvw>=3.1.0
SPARQLWrapper>=2.0.0
```
