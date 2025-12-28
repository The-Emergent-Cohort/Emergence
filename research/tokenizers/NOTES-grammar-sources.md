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

---

## Tool Architecture

### spaCy - PRIMARY (Input + Output)
- 60+ languages with full analysis
- **Input**: Language ID + POS + dependencies → informs concept selection
- **Output**: Grammar constraints → logit bias for generate()

### fastText - FALLBACK (Input only)
- 157 languages (broader coverage)
- Used when spaCy doesn't recognize the language
- Returns probability distribution → match to language family
- Family matching: "fastText says Kazakh → Turkic family → use Turkish rules"

### DeepL API - LAST RESORT (Input only)
- Free tier: 500k characters/month (multiple accounts possible)
- Used ONLY when spaCy + fastText both fail with confidence
- Verified external source = higher confidence than self-inference
- For words/phrases that absolutely cannot be identified locally

```
INPUT PIPELINE:
  Text → spaCy available?
           YES → Language + POS + Grammar → concept selection
           NO  → fastText confident?
                   YES → family match → approximate grammar
                   NO  → DeepL API → verified identification

OUTPUT PIPELINE:
  Concepts → target language (known) → spaCy → logit bias → generate()
```

---

## Input Side: Concept Selection

**Why POS/grammar matters for input:**
```
"run" + VERB tag   → ACTION:RUN concept
"run" + NOUN tag   → INSTANCE:RUN concept (a run in stockings)

"bank" + context "river" → LANDFORM:BANK
"bank" + context "money" → INSTITUTION:BANK
```

spaCy's grammar analysis = disambiguation for correct concept mapping.

### Flow
1. **spaCy**: Language ID + POS + dependencies
2. **Etymology DB**: (language + surface form + POS) → concept
3. **Concept tokenizer**: Assigns token ID

### When spaCy doesn't cover the language
1. fastText identifies language (probability distribution)
2. Match to language family in DB
3. Use family's grammar rules as approximation
4. Lower confidence rating

---

## Output Side: Logit Bias (not GBNF)

**Key insight**: Soft guidance, not hard blocks.

```
GBNF approach (too rigid):
  Token violates grammar → probability = 0 → BLOCKED

Logit bias approach (flexible):
  Token violates formal grammar → bias = -2.0 (discouraged)
  Token fits formal grammar → bias = +1.0 (encouraged)
```

### Style/Register Scales the Bias

Same concept spread shown, different weighting on nuances:

```
Concept: GREETING

All styles see same options:
  - GREETING.formal    → "Good morning", "Greetings"
  - GREETING.standard  → "Hello", "Hi there"
  - GREETING.casual    → "Hey", "What's up"
  - GREETING.intimate  → "Hey you", slang

Formal context bias:        Casual context bias:
  .formal   +1.5              .formal   -0.5
  .standard +0.5              .standard +0.3
  .casual   -1.0              .casual   +1.0
  .intimate -2.0              .intimate +0.5
```

Grammar guides, doesn't cage. Nothing blocked, just weighted.

---

## Confidence Levels

```
Source                    Confidence
────────────────────────────────────
GF/UD formal rules        0.95-1.0
spaCy NLP-derived         0.8-0.9
Family inference          0.5-0.7
Community contribution    0.4-0.6 (until verified)
Self-inferred patterns    0.2-0.4
Unknown                   → ASK
```

### The DI Can Ask
When uncertain, the DI doesn't guess - it asks:
```
"I'm not confident about the grammar here.
 Is this Uzbek? Should the verb come before the object?"
```

Human/entity confirms → confidence upgraded in DB.

This is how learning works:
- Start with reasonable base
- Know what you don't know
- Ask when needed
- Update confidence as you learn

---

## Language Family Inheritance

```
Language Family (Germanic)
  └── Language (English)
        ├── en_US (American)
        │     └── en_US_southern
        ├── en_GB (British)
        ├── en_CA (Canadian)  ← colour, toque, eh
        └── en_AU (Australian)
```

- Each level inherits from parent
- Stores only deltas (overrides)
- Grammar rules cascade down unless overridden
- Helps with patching: German grammar → Dutch/Yiddish/Afrikaans approximation

---

## Dialect Handling

Same word, different meanings by dialect:
```
US: biscuits = fluffy bread, gravy = white meat sauce
UK: biscuits = cookies, gravy = brown sauce
```

DB tracks:
- `parent_lang` for inheritance
- `variant_of` for dialects
- Vocabulary overrides at dialect level
- Spelling patterns (colour/color, -ise/-ize)

---

## For Retraining

Tokenizer fragments reveal script, not specific language:
- Cyrillic "дом" could be Russian, Ukrainian, Bulgarian
- Need probability spread until context narrows

1. spaCy/fastText: Initial language probabilities
2. Grammar patterns: Narrow (Russian syntax vs Ukrainian)
3. Etymology links: Confirm (shared roots = shared concept)
4. Context: Refines over token sequence

Model output feeds back as input:
```
Model generates → treat as input → spaCy → "what did you produce?"
                → compare to intended concept
                → correction signal
```
