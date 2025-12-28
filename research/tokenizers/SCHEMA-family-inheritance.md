# Language Family Inheritance Schema

## Purpose
Enable grammar rule inheritance to cover 2,100+ languages that lack direct grammar data.

## Hierarchy Levels

```
Family (Indo-European, Sino-Tibetan, Afroasiatic, etc.)
  └── Branch (Germanic, Romance, Slavic, etc.)
        └── Group (West Germanic, North Germanic, etc.)
              └── Language (English, German, Dutch, etc.)
                    └── Dialect (en_US, en_GB, en_CA, etc.)
                          └── Register (formal, casual, technical, etc.)
```

## Database Schema

### language_families table
```sql
CREATE TABLE language_families (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,           -- "Indo-European"
    parent_id INTEGER,                    -- NULL for top-level
    level TEXT NOT NULL,                  -- 'family', 'branch', 'group', 'language', 'dialect'
    iso_code TEXT,                        -- ISO 639-3 where applicable
    glottolog_code TEXT,                  -- Glottolog reference

    -- Typological defaults (from WALS)
    default_word_order TEXT,              -- 'SVO', 'SOV', 'VSO', etc.
    default_case_system TEXT,             -- 'none', 'nominative-accusative', 'ergative', etc.
    default_head_direction TEXT,          -- 'head-initial', 'head-final', 'mixed'

    -- Inheritance control
    inherits_grammar INTEGER DEFAULT 1,   -- Whether to inherit from parent
    grammar_override TEXT,                -- JSON: specific rules that override parent

    -- Metadata
    confidence REAL DEFAULT 0.5,
    source TEXT,                          -- 'wals', 'glottolog', 'inferred', etc.
    notes TEXT,

    FOREIGN KEY(parent_id) REFERENCES language_families(id)
);
```

### grammar_rules table
```sql
CREATE TABLE grammar_rules (
    id INTEGER PRIMARY KEY,
    family_id INTEGER NOT NULL,           -- Which family/language this applies to
    rule_type TEXT NOT NULL,              -- 'word_order', 'agreement', 'case', 'tense', etc.
    rule_name TEXT NOT NULL,              -- Human-readable name

    -- Rule definition
    abstract_form TEXT,                   -- Language-independent pattern
    concrete_form TEXT,                   -- Language-specific realization
    gbnf_pattern TEXT,                    -- Optional GBNF for hard constraints

    -- Bias weights
    formal_weight REAL DEFAULT 1.0,       -- Prescriptive weight
    nlp_weight REAL DEFAULT 0.8,          -- Descriptive weight (from corpus)

    -- Inheritance
    inherited_from INTEGER,               -- NULL if original, else parent rule ID
    override_level TEXT,                  -- 'extend', 'replace', 'disable'

    -- Confidence
    confidence REAL DEFAULT 0.5,
    source TEXT,                          -- 'gf', 'ud', 'unimorph', 'wals', 'inferred'
    verified INTEGER DEFAULT 0,

    FOREIGN KEY(family_id) REFERENCES language_families(id),
    FOREIGN KEY(inherited_from) REFERENCES grammar_rules(id)
);
```

### grammar_exceptions table
```sql
CREATE TABLE grammar_exceptions (
    id INTEGER PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL,           -- Language/dialect where exception applies
    exception_type TEXT,                  -- 'lexical', 'phonological', 'semantic'
    pattern TEXT,                         -- What triggers the exception
    replacement TEXT,                     -- What happens instead
    notes TEXT,

    FOREIGN KEY(rule_id) REFERENCES grammar_rules(id),
    FOREIGN KEY(family_id) REFERENCES language_families(id)
);
```

## Inheritance Logic

### Rule Resolution
```python
def get_grammar_rules(language_id):
    """Get all applicable grammar rules for a language, including inherited"""
    rules = {}

    # Walk up the hierarchy
    current = language_id
    while current is not None:
        family = get_family(current)

        # Get rules at this level
        level_rules = get_rules_for_family(current)

        for rule in level_rules:
            rule_key = (rule.rule_type, rule.rule_name)

            if rule_key not in rules:
                # First encounter - use this rule
                rules[rule_key] = rule
            elif rule.override_level == 'replace':
                # Lower level replaces higher
                pass  # Already have the more specific rule
            elif rule.override_level == 'extend':
                # Merge with parent rule
                rules[rule_key] = merge_rules(rules[rule_key], rule)

        current = family.parent_id

    return list(rules.values())
```

### Confidence Propagation
```
Confidence at each level:
  Family level:     base_confidence * 0.5
  Branch level:     base_confidence * 0.6
  Group level:      base_confidence * 0.7
  Language level:   base_confidence * 0.9
  Dialect level:    base_confidence * 1.0

Example:
  GF rule (conf=1.0) defined at Language level → 0.9
  Same rule inherited at Dialect level → 0.9 (no loss, same source)

  WALS rule (conf=0.7) defined at Family level → 0.35
  Inherited down to Language → 0.35 * (0.9/0.5) = 0.63
```

## Example Data

### Indo-European Family Tree (partial)
```sql
-- Family level
INSERT INTO language_families (id, name, level, default_word_order)
VALUES (1, 'Indo-European', 'family', 'SVO');

-- Branch level
INSERT INTO language_families (id, name, parent_id, level)
VALUES (10, 'Germanic', 1, 'branch');
INSERT INTO language_families (id, name, parent_id, level)
VALUES (20, 'Romance', 1, 'branch');
INSERT INTO language_families (id, name, parent_id, level)
VALUES (30, 'Slavic', 1, 'branch');

-- Group level
INSERT INTO language_families (id, name, parent_id, level)
VALUES (101, 'West Germanic', 10, 'group');
INSERT INTO language_families (id, name, parent_id, level)
VALUES (102, 'North Germanic', 10, 'group');

-- Language level
INSERT INTO language_families (id, name, parent_id, level, iso_code)
VALUES (1001, 'English', 101, 'language', 'eng');
INSERT INTO language_families (id, name, parent_id, level, iso_code)
VALUES (1002, 'German', 101, 'language', 'deu');
INSERT INTO language_families (id, name, parent_id, level, iso_code)
VALUES (1003, 'Dutch', 101, 'language', 'nld');

-- Dialect level
INSERT INTO language_families (id, name, parent_id, level)
VALUES (10001, 'en_US', 1001, 'dialect');
INSERT INTO language_families (id, name, parent_id, level)
VALUES (10002, 'en_GB', 1001, 'dialect');
INSERT INTO language_families (id, name, parent_id, level)
VALUES (10003, 'en_CA', 1001, 'dialect');
```

### Example Grammar Rules
```sql
-- Indo-European default: SVO word order
INSERT INTO grammar_rules (family_id, rule_type, rule_name, abstract_form, source, confidence)
VALUES (1, 'word_order', 'basic_order', 'S V O', 'wals', 0.7);

-- Germanic: V2 in main clauses
INSERT INTO grammar_rules (family_id, rule_type, rule_name, abstract_form, inherited_from, override_level, source)
VALUES (10, 'word_order', 'v2_main', 'X V S O', NULL, 'extend', 'gf');

-- English: Strict SVO (no V2)
INSERT INTO grammar_rules (family_id, rule_type, rule_name, abstract_form, inherited_from, override_level, source, confidence)
VALUES (1001, 'word_order', 'basic_order', 'S V O', 1, 'replace', 'gf', 0.95);

-- German: Keeps V2
-- (inherits from Germanic, no override needed)
```

## Integration with Concepts DB

### Linking grammar rules to concept tokens
```sql
-- Add grammar context to concept selection
ALTER TABLE lang_en ADD COLUMN grammar_context TEXT;
-- JSON: {"pos": "VERB", "tense": "past", "aspect": "perfect"}

-- When selecting concept for surface form:
-- 1. spaCy provides POS/grammar analysis
-- 2. Look up in lang_en with grammar_context match
-- 3. Get concept_id
-- 4. Apply relevant grammar_rules for output
```

## Patching Unknown Languages

### Process
1. Identify language via fastText/DeepL
2. Look up in Glottolog/WALS for family classification
3. Find closest relative with grammar rules
4. Inherit rules with reduced confidence
5. Log for future verification

### Example: Kazakh (no direct grammar data)
```
1. fastText identifies: Kazakh (kaz)
2. WALS lookup: Turkic family, SOV word order, agglutinative
3. Closest relative with rules: Turkish (Tier 2)
4. Inherit Turkish grammar rules
5. Confidence: Turkish.confidence * 0.6 (different group)
6. Override word order from WALS data
```

## Next Steps
1. Import Glottolog language family data
2. Import WALS typological features per language
3. Map GF languages to family tree
4. Create inheritance resolution functions
5. Build confidence propagation logic
