# Structured Tokenizer Development Notes

## Goal
Replace/supplement Mystral's 32k BPE tokenizer with a concept-based etymological tokenizer that:
- Keeps concepts whole instead of fragmenting
- Uses etymological roots + grammatical modifiers
- Enables type/subtype organization for efficient RLHF guidance
- Works with DB backend for dynamic lookup

## Data Source
**etymology-db** (GitHub: droher/etymology-db)
- 4.2M+ etymological relationships
- 2.0M+ terms across 3300+ languages
- 31 relationship types including:
  - `has_prefix`, `has_suffix` — morphological modifiers
  - `compound_of` — multi-root combinations
  - `derived_from`, `inherited_from` — concept lineages
  - `group_affix_root`, `group_derived_root` — root clustering
- Format: Gzipped CSV and Parquet

## Architecture Vision
```
Surface text
     ↓
Concept-based tokenizer (lookup in DB)
     ↓
[ROOT_CONCEPT] + [MODIFIERS]
     ↓
Embedding lookup (smaller, root-focused)
     ↓
Model inference
     ↓
generate() with guidance from structured tokens
```

## Key Insight from Patrick
- English has ~1M "words" but far fewer concept roots
- "-ness" / "-iness" = [STATE_OF] modifier
- "-en" vs "-y" = different grammatical transformations
- Same root + different modifier = different word class
- Tokenizer needs morphological rules, not just vocabulary

## Next Steps
1. Download etymology-db sample
2. Analyze structure for tokenizer mapping
3. Extract English concept roots
4. Map modifier relationships
5. Design DB schema
6. Build prototype tokenizer

## Hardware Constraint
- Target: 8GB GTX 1070
- Must be efficient enough to run alongside model inference
- SQLite for embedded DB queries during generation
