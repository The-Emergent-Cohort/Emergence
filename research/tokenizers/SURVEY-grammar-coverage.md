# Grammar Source Coverage Survey

## Summary Table

| Source | Languages | Data Type | Depth |
|--------|-----------|-----------|-------|
| Etymology DB | 2,265 | Vocabulary, etymology links | Words only |
| WALS | 2,650 | Typological features | High-level (word order, phonology) |
| UniMorph | 169 (98 full) | Morphological paradigms | Conjugation/declension tables |
| Universal Dependencies | 150+ | Syntax treebanks | POS, dependencies, morphology |
| spaCy | 70+ (25 full) | NLP pipelines | POS, NER, parsing |
| GF | ~40 | Formal grammars | Complete syntax + morphology |
| fastText | 157 | Language identification | ID only, no grammar |

## Coverage Analysis

### Vocabulary (Etymology DB): 2,265 languages
- Broadest vocabulary coverage
- Cross-language etymology links
- No grammar rules

### Typological Features (WALS): 2,650 languages
- Broadest grammar FEATURE coverage
- But only high-level patterns:
  - Word order (SVO, SOV, etc.)
  - Case systems
  - Tense marking
  - Phonological features
- NOT full grammar rules, just typological classification
- Useful for: Family inference, template selection

### Morphology (UniMorph): 169 languages
- Conjugation/declension paradigms
- 98 languages with substantial data
- Examples:
  - Czech: 824k paradigms
  - Polish: 274k paradigms
  - Finnish: 57k paradigms
- Useful for: Verb/noun forms, agreement rules

### Syntax (Universal Dependencies): 150+ languages
- Annotated treebanks
- POS tags, dependency relations
- Morphological features
- Useful for: Actual usage patterns, NLP training

### NLP Pipelines (spaCy): 70+ languages
- 25 languages with full trained pipelines
- 70+ with tokenization support
- Full pipelines include: POS, NER, dependency parsing
- Useful for: Runtime analysis, our primary tool

### Formal Grammars (GF): ~40 languages
- Complete formal grammar specifications
- Abstract syntax + concrete linearization
- Highest quality, lowest coverage
- Useful for: Gold standard rules

### Language ID (fastText): 157 languages
- Identification only, no grammar data
- Useful for: Fallback language detection

## Coverage Tiers

### Tier 1: Full Grammar (~40 languages)
- GF formal grammar
- UniMorph morphology
- UD treebank
- spaCy pipeline
Languages: English, German, French, Spanish, Italian, Finnish, Russian, etc.

### Tier 2: Good Coverage (~100-150 languages)
- UniMorph OR UD
- spaCy tokenization
- WALS typology
Languages: Most major languages, some regional

### Tier 3: Partial Coverage (~150-500 languages)
- WALS typology only
- fastText ID
- Family inference possible
Languages: Many African, Asian, indigenous languages

### Tier 4: Minimal Coverage (~500-2000+ languages)
- WALS typology (some)
- Etymology vocabulary
- No grammar rules
Languages: Rare, endangered, historical languages

## Gap Analysis

```
Etymology DB languages:        2,265
 - With GF grammar:              ~40  (1.8%)
 - With UniMorph morphology:    ~169  (7.5%)
 - With UD treebank:            ~150  (6.6%)
 - With spaCy pipeline:          ~25  (1.1%)
 - With WALS typology:        ~2,650  (covers most + more)

Grammar rule coverage gap:    ~2,100 languages (93%)
```

## Patching Strategy by Tier

### Tier 1 (40 langs): Direct mapping
- Use GF rules as formal base
- Cross-reference with UD for NLP weights
- spaCy for runtime

### Tier 2 (100-150 langs): Morphology + inference
- UniMorph for forms
- UD patterns where available
- Infer syntax from WALS typology

### Tier 3 (150-500 langs): Family inheritance
- WALS typology → template selection
- Inherit from closest Tier 1/2 relative
- Lower confidence rating

### Tier 4 (500+ langs): Minimal bootstrap
- WALS word order if available
- Family inheritance from distant relatives
- Mark as "needs verification"
- Use DeepL API for specific queries

## Data Sources

- WALS: https://wals.info/ (2,650 languages, 142 features)
- UniMorph: https://unimorph.github.io/ (169 languages)
- Universal Dependencies: https://universaldependencies.org/ (150+ languages)
- spaCy: https://spacy.io/models (70+ languages)
- GF: https://www.grammaticalframework.org/ (~40 languages)
- fastText: https://fasttext.cc/docs/en/language-identification.html (157 languages)

## Next Steps

1. Download WALS feature data for language family mapping
2. Cross-reference UniMorph language list with etymology DB
3. Map language families for inheritance hierarchy
4. Identify Tier 1 languages to prioritize
5. Design schema for multi-source confidence tracking
