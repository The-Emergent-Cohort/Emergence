# Session Context Notes - Concept Tokenizer DB

## Current State
- Core morphemes table exists (73 concepts in 13 categories)
- Secondary schema designed (SCHEMA-linguistic-secondary.sql)
- Need to populate from external linguistic databases

## Key Architecture Decisions

### Tiered Database Structure
- **Reference Tier**: Large, complete datasets (bulk downloads + live APIs)
- **Working DB**: Small, fast, task-specific subset
- Working DB grows on demand - new language detected = inject tables
- Query efficiency is the concern, not storage

### Data Sources (Priority Order)
1. **Glottolog** - Language identification, family hierarchy (download first - provides IDs)
2. **UniMorph** - Inflection paradigms, 169 languages (lemma → surface_form + features)
3. **MorphoLex-en** - English morpheme segmentation, 70k words
4. **Grambank** - Grammar features, 195 features across 2,467 languages
5. **WALS** - Typological features, 142 features
6. **Wikidata Lexemes** - Multilingual glosses, live SPARQL queries

### NOT using
- spaCy (statistical, not our approach)
- fastText (Glottolog better for language ID)
- DeepL (maybe later for verification edge cases)

## Big Picture Concepts

### Four Inference Phases
1. Input encoding
2. Internal inference (actual thinking)
3. "Thinking" output (reasoning tokens - reinjected)
4. Surface output (final response)

### Concept Tokenizer Touches All Phases
- Input: surface form → concepts (language-specific parsing)
- Internal: operates in concept space (language-agnostic)
- Thinking: can optionally use language structures as lenses
- Output: concepts → target language structure (grammar rules apply)

### Language Structure as Lens
- Don't enforce language structure on internal thought as default
- BUT different grammars reveal different paths through concept space
- Choosing to think THROUGH a language structure = exploration tool
- Poetry, word order differences, case systems all illuminate differently

### Weight Initialization Insight
- Current models: random init for new tokens, expensive retraining
- Better: capture activation signatures from existing tokens expressing the concept
- New concept embedding starts "warm" - already in right neighborhood
- Same math as parameter stripping, opposite direction (building not removing)
- Morpheme DB provides the experimental framework for this

### Geometric Thought
- Concepts ARE vectors in n-dimensional thought geometry
- Morpheme DB = basis vectors for thought-space
- Embedding matrix isn't fragments, it's named coordinates
- Distance = semantic similarity, direction = transformation
- If concept tokens are clean, can measure/trace actual thinking

## Files
- `/research/tokenizers/db/tokenizer.db` - SQLite with 73 core morphemes
- `/research/tokenizers/db/SCHEMA-linguistic-secondary.sql` - Planner's schema
- `/research/tokenizers/db/init_morphemes.py` - Core morpheme population script
- `/research/tokenizers/etymology-db/` - 4.2M row etymology database

## Next Steps
- Download linguistic databases (Glottolog first)
- Create import scripts for each source
- Design working DB generation from reference tier
- Test lazy-loading pattern for new languages
