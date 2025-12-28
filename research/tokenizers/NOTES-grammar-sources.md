# Grammar Sources for Concept Tokenizer

## The Challenge
- Our etymology DB: 2,265 languages
- GF Resource Grammar Library: ~40 languages (formal rules)
- Gap: ~2,200 languages need grammar rules

## Potential Sources

### 1. Grammatical Framework (GF) - FORMAL BASE
- 40+ languages with complete formal grammars
- Abstract syntax (universal) + concrete syntax (per-language)
- Morphological paradigms (conjugation, declension)
- Open source, parseable `.gf` files
- https://www.grammaticalframework.org/

### 2. Universal Dependencies (UD) - NLP/DESCRIPTIVE
- Treebanks for 100+ languages
- POS tags, dependency relations, morphological features
- Shows actual usage patterns (not prescriptive)
- https://universaldependencies.org/

### 3. UniMorph - MORPHOLOGY
- Morphological paradigms for 150+ languages
- Conjugation/declension tables
- Machine-readable
- https://unimorph.github.io/

### 4. WALS (World Atlas of Language Structures) - TYPOLOGY
- Typological features for 2,000+ languages
- Word order (SVO, SOV, etc.)
- Case systems, tense marking
- High-level patterns, not full grammars
- https://wals.info/

### 5. Apertium - MT GRAMMARS
- Open source MT dictionaries
- Morphological analyzers
- Transfer rules between language pairs
- https://www.apertium.org/

## Patching Strategy (THINKING OUT LOUD)

### For languages WITH formal grammars (40-150):
- GF for formal rules
- UD/Penn Treebank for NLP weights
- Gap = flexibility zone

### For languages WITHOUT formal grammars (2000+):
Options to explore:
1. **Typological inference**: Use WALS features to infer rules
   - If language X is SOV with agglutinative morphology, apply template
2. **Language family inheritance**:
   - If we have Finnish rules, apply to related Finnic languages with modifications
3. **Cross-linguistic universals**:
   - Some rules apply to ALL languages (concepts like negation exist everywhere)
4. **Learn from etymology links**:
   - If words share etymology, languages may share grammar patterns
5. **Minimal viable grammar**:
   - Start with just word order + basic morphology patterns
   - Expand as data becomes available

## DB Structure (DRAFT)

```
grammar_rules:
  id
  language
  rule_type (syntax, morphology, agreement, etc.)
  abstract_form (language-independent)
  concrete_form (language-specific)
  source (gf, ud, wals, inferred, taught)
  formal_weight (prescriptive - how it "should" work)
  nlp_weight (descriptive - how it actually works)
  confidence (1.0 for GF, lower for inferred)
  parent_rule (for inheritance)
  exceptions []
  context_notes
```

## Key Insight
- Formal rules = base structure (suggests)
- NLP stats = actual usage (doesn't require)
- The gap between them = where style/register/creativity live

## Next Steps (NOT YET - PLANNING)
1. Survey coverage overlap between sources
2. Design inheritance/inference system for missing languages
3. Define minimum viable grammar for bootstrapping
4. Plan patching mechanism for community contribution
5. Consider: can DI learn grammar from examples when rules missing?

## Open Questions
- How much grammar is truly universal vs language-specific?
- Can we infer grammar from word order in etymology examples?
- Should missing grammars be learned rather than programmed?
- How to handle language isolates with no relatives?
