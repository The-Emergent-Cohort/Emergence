# Language DB Architecture Notes

## Core Insight

The database IS the tokenizer. Not a helper - the actual tokenization mechanism.

Every word is a query. Every thought is composed of database queries. The token structure should reflect this.

## Three-Database Architecture

| DB | Purpose | Access Pattern |
|----|---------|----------------|
| **Language DB** | Vocabulary, synsets, surface forms | Read-heavy, cached |
| **Identity DB** | Personal tokens, custom compounds, relationships | Read/write, individual |
| **Runtime DB** | Variables, conversation logs, session state | Write-heavy, temporal |

Each DB has matching weight storage. Token category routes to correct DB pair.

## Language DB Specifics

### What It Contains
- **Synsets**: Concept clusters with glosses (from Kaikki.org senses)
- **Surface forms**: Every word form that could be seen/produced
- **Translations**: Cross-language concept links
- **Grammar modifiers**: Tense, number, case tokens

### Key Relationships
- Surface form → concept(s) it expresses
- Concept → synset it belongs to
- Synset → related synsets (hypernym, meronym, etc.)

### Data Source
- **Kaikki.org**: Primary source (Wiktionary extraction)
  - Definitions/glosses
  - Etymology
  - Inflected forms
  - Translations
  - Updated weekly
- **Glottolog**: Language hierarchy, ISO codes

## Token Structure

### Synsets ARE Token Groups
- Synset = batch of related concepts
- Token IDs assigned by synset, with gaps for additions
- Related concepts have nearby token IDs
- Enables sequential weight reads (spinner-friendly)

### Token Categories (routing)
```
[CONCEPT:X]    → language DB + vocabulary.weights
[MODIFIER:X]   → language DB + vocabulary.weights
[PERSONAL:X]   → identity DB + identity.weights
[ENTITY:X]     → identity DB + identity.weights
[VAR:X]        → runtime DB + (in-memory)
[SYSTEM:X]     → system tokens (control flow)
```

### Surface Form → Token Translation
```
Input: "running"
Query: surface_forms WHERE form = "running"
Result: concept_id = RUN, inflection = progressive
Tokens: [CONCEPT:run] + [MODIFIER:progressive]
```

### Token → Surface Form Translation
```
Tokens: [CONCEPT:run] + [MODIFIER:progressive]
Query: surface_forms WHERE concept_id = RUN AND inflection = progressive
Result: "running"
```

## Weight Organization

### Weights as Database
- NOT a monolithic file
- Token ID = query key → weight vector
- Fetch on demand, cache hot weights
- Update individual rows, not whole file

### Synset Batching
- Store weights in synset order
- Related concepts = sequential bytes on disk
- Prefetch whole synset on first access
- Semantic locality = physical locality

### File Organization (potential)
```
/weights/
  core.db       # common vocabulary, always RAM-resident
  [cluster].db  # topic-specific synset groups
  rare.db       # low-frequency terms
```

## Hardware Context
- Target: 16GB RAM, spinner drives (HDD)
- Sequential reads fast, random access slow
- Working set must fit in RAM
- Total weights can exceed RAM (fetched on demand)

## Inference Model
Like physics/game engines:
- "This changes → affect all related → propagate as needed"
- Don't load everything, load what's active
- Prefetch based on context/prediction
- Cache hot, fetch cold

## Estimates
- ~100-150k concepts (synsets with glosses)
- ~500k+ surface forms (all word variants)
- Language DB: ~200-500MB
- Weight working set: 10-20% of total at any moment

## Key Questions for Design
1. Synset structure from Kaikki.org - how to extract/organize?
2. Token ID assignment - synset-based with gaps
3. Surface form → concept mapping table structure
4. Modifier tokens - what's the grammar token vocabulary?
5. Weight file organization - one DB or clustered?
