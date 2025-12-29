-- ============================================================================
-- UNIFIED CONCEPT TOKENIZER DATABASE SCHEMA
-- The DB IS the tokenizer. Each concept entry = one token ID.
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;  -- Better concurrent read performance

-- ============================================================================
-- TOKEN ID RANGE ARCHITECTURE
-- ============================================================================
-- ID Range            Purpose                        Examples
-- ─────────────────────────────────────────────────────────────────
-- 0 - 999             Reserved control tokens        <pad>, <unk>, <s>, </s>
-- 1,000 - 99,999      Thinking/reasoning tokens      <think>, </think>
-- 100,000 - 199,999   Tool control tokens            <tool_call>, </tool_call>
-- 200,000 - 299,999   Role markers                   <user>, <assistant>
-- 300,000 - 499,999   Modality routing tokens        <text:en>, <code:python>
-- 500,000 - 999,999   System reserved                Byte fallbacks, future use
-- 1,000,000 - 1,999,999  Personal/entity tokens      Navigator-specific concepts
-- 2,000,000 - 2,999,999  Universal modifiers         STATE_OF, NEGATION, etc.
-- 3,000,000+             Lexical concepts            Etymology-derived roots

-- ============================================================================
-- 1. CONCEPTS TABLE (Core Token Table)
-- ============================================================================
-- Each row represents a single token ID in the model's vocabulary.
-- This is THE tokenizer vocabulary.

CREATE TABLE concepts (
    -- Primary identity
    id INTEGER PRIMARY KEY,                    -- THE token ID used by model
    canonical TEXT NOT NULL,                   -- Primary representation

    -- Classification
    concept_type TEXT NOT NULL DEFAULT 'morpheme',
    -- Values: 'control', 'thinking', 'tool', 'role', 'byte_fallback',
    --         'personal', 'modifier', 'morpheme', 'word', 'phrase', 'deprecated'

    -- Semantic domain
    domain TEXT,                               -- 'abstract', 'physical', 'social', etc.
    subdomain TEXT,                            -- More specific category

    -- Output routing (THE modality field)
    modality TEXT DEFAULT 'text',
    -- Values: 'text', 'code', 'thinking', 'tool_call', 'tool_result',
    --         'physics', 'internal', 'display', 'file', 'shell'

    target_language TEXT,                      -- For text: 'en', 'de', 'fr', etc.
    target_format TEXT,                        -- For code: 'python', 'sql', etc.
    route_destination TEXT,                    -- For internal: fd path, pipe name

    -- Hierarchy
    parent_concept_id INTEGER,                 -- For inheritance (modifier chains)
    composable INTEGER DEFAULT 1,              -- Can this combine with others?

    -- Confidence tracking
    confidence REAL DEFAULT 0.8,
    source TEXT DEFAULT 'etymology',           -- 'gf', 'ud', 'etymology', 'taught', 'inferred'
    verified INTEGER DEFAULT 0,                -- Human verified?

    -- For deprecated tokens (V3-Tekken migration)
    is_deprecated INTEGER DEFAULT 0,
    deprecated_from_id INTEGER,                -- Original V3-Tekken ID if deprecated
    replacement_concept_id INTEGER,            -- New concept to use instead

    -- Metadata
    description TEXT,
    usage_notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,

    -- Constraints
    FOREIGN KEY(parent_concept_id) REFERENCES concepts(id),
    FOREIGN KEY(replacement_concept_id) REFERENCES concepts(id)
);

-- ============================================================================
-- 2. ETYMOLOGY TABLE (Surface Form → Concept Mapping)
-- ============================================================================
-- Maps (language, surface_form, POS) → concept_id
-- This is the INPUT side of tokenization

CREATE TABLE etymology (
    id INTEGER PRIMARY KEY,

    -- Surface form identity
    language TEXT NOT NULL,                    -- ISO code or full name
    surface_form TEXT NOT NULL,                -- How it appears as text
    pos_tag TEXT,                              -- Part of speech (spaCy tag)

    -- Mapping to concept
    concept_id INTEGER NOT NULL,

    -- Morphological info
    morpheme_type TEXT DEFAULT 'word',
    -- Values: 'root', 'prefix', 'suffix', 'infix', 'word', 'phrase'

    -- Etymology chain
    derived_from_id INTEGER,                   -- Parent etymology entry
    etymology_chain TEXT,                      -- Proto-language chain (JSON)

    -- Confidence
    confidence REAL DEFAULT 0.8,
    source TEXT DEFAULT 'wiktionary',          -- 'wiktionary', 'ud', 'spacy', 'taught'

    -- Usage frequency (for disambiguation)
    frequency INTEGER DEFAULT 0,
    register TEXT,                             -- 'formal', 'casual', 'technical', etc.

    -- Metadata
    wiktionary_id TEXT,                        -- Original source ID
    notes TEXT,

    -- Constraints
    FOREIGN KEY(concept_id) REFERENCES concepts(id),
    FOREIGN KEY(derived_from_id) REFERENCES etymology(id),
    UNIQUE(language, surface_form, pos_tag)    -- One concept per (lang, form, pos)
);

-- ============================================================================
-- 3. LANGUAGE FAMILIES TABLE
-- ============================================================================
-- Hierarchical structure for grammar rule inheritance

CREATE TABLE language_families (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,

    -- Hierarchy
    parent_id INTEGER,
    level TEXT NOT NULL,
    -- Values: 'family', 'branch', 'group', 'language', 'dialect', 'register'

    -- Standard codes
    iso_639_3 TEXT,                            -- ISO 639-3 code
    glottolog_code TEXT,                       -- Glottolog reference
    wals_code TEXT,                            -- WALS reference

    -- Typological defaults (from WALS)
    default_word_order TEXT,                   -- 'SVO', 'SOV', 'VSO', etc.
    default_head_direction TEXT,               -- 'head-initial', 'head-final', 'mixed'
    default_case_system TEXT,                  -- 'nominative-accusative', 'ergative', etc.
    default_morphology_type TEXT,              -- 'isolating', 'agglutinative', 'fusional'

    -- Inheritance control
    inherits_grammar INTEGER DEFAULT 1,        -- Whether to inherit from parent
    grammar_override TEXT,                     -- JSON: specific overrides

    -- Confidence
    confidence REAL DEFAULT 0.5,
    source TEXT,                               -- 'wals', 'glottolog', 'inferred'

    -- NLP tool support
    spacy_model TEXT,                          -- e.g., 'en_core_web_sm'
    fasttext_supported INTEGER DEFAULT 0,

    -- Metadata
    notes TEXT,

    FOREIGN KEY(parent_id) REFERENCES language_families(id)
);

-- ============================================================================
-- 4. GRAMMAR RULES TABLE
-- ============================================================================
-- Grammar rules with inheritance and dual weighting

CREATE TABLE grammar_rules (
    id INTEGER PRIMARY KEY,

    -- Scope
    family_id INTEGER NOT NULL,                -- Which family/language this applies to

    -- Rule identity
    rule_type TEXT NOT NULL,
    -- Values: 'word_order', 'agreement', 'case', 'tense', 'aspect', 'mood',
    --         'negation', 'question', 'relative', 'coordination', 'morphology'

    rule_name TEXT NOT NULL,                   -- Human-readable identifier

    -- Rule definition
    abstract_form TEXT,                        -- Language-independent pattern
    concrete_form TEXT,                        -- Language-specific realization
    gbnf_pattern TEXT,                         -- Optional GBNF for hard constraints
    regex_pattern TEXT,                        -- For validation

    -- Dual weights (prescriptive vs descriptive)
    formal_weight REAL DEFAULT 1.0,            -- Prescriptive (how it "should" work)
    nlp_weight REAL DEFAULT 0.8,               -- Descriptive (actual corpus usage)

    -- For logit bias calculation
    bias_when_violated REAL DEFAULT -2.0,      -- Negative bias if rule violated
    bias_when_satisfied REAL DEFAULT 0.5,      -- Positive bias if rule followed

    -- Inheritance
    inherited_from_id INTEGER,                 -- Parent rule if inherited
    override_level TEXT DEFAULT 'extend',      -- 'extend', 'replace', 'disable'

    -- Confidence
    confidence REAL DEFAULT 0.5,
    source TEXT,                               -- 'gf', 'ud', 'unimorph', 'wals', 'inferred'
    verified INTEGER DEFAULT 0,

    -- Metadata
    examples TEXT,                             -- JSON array of example sentences
    counter_examples TEXT,                     -- JSON array of violations
    notes TEXT,

    FOREIGN KEY(family_id) REFERENCES language_families(id),
    FOREIGN KEY(inherited_from_id) REFERENCES grammar_rules(id),
    UNIQUE(family_id, rule_type, rule_name)
);

-- ============================================================================
-- 5. GRAMMAR EXCEPTIONS TABLE
-- ============================================================================
-- Exceptions to grammar rules (lexical, phonological, semantic)

CREATE TABLE grammar_exceptions (
    id INTEGER PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    family_id INTEGER NOT NULL,                -- Language/dialect where exception applies

    exception_type TEXT NOT NULL,
    -- Values: 'lexical', 'phonological', 'semantic', 'register', 'historical'

    trigger_pattern TEXT NOT NULL,             -- What triggers the exception
    replacement_pattern TEXT,                  -- What happens instead
    affected_concepts TEXT,                    -- JSON array of concept IDs

    confidence REAL DEFAULT 0.7,
    notes TEXT,

    FOREIGN KEY(rule_id) REFERENCES grammar_rules(id),
    FOREIGN KEY(family_id) REFERENCES language_families(id)
);

-- ============================================================================
-- 6. DEPRECATED TOKENS TABLE
-- ============================================================================
-- Maps V3-Tekken tokens (0-131071) to new concepts

CREATE TABLE deprecated_tokens (
    id INTEGER PRIMARY KEY,

    -- Original V3-Tekken token
    old_token_id INTEGER NOT NULL UNIQUE,      -- V3-Tekken ID (0-131071)
    old_surface_form TEXT NOT NULL,            -- Original text representation
    old_token_bytes BLOB,                      -- Raw bytes if needed

    -- New concept mapping
    new_concept_id INTEGER,                    -- Our 2M+ range concept
    mapping_type TEXT DEFAULT 'direct',
    -- Values: 'direct', 'split', 'merged', 'obsolete'

    -- For split tokens (one old → multiple new)
    split_concept_ids TEXT,                    -- JSON array of concept IDs

    -- Confidence
    confidence REAL DEFAULT 0.9,
    mapping_source TEXT DEFAULT 'automatic',   -- 'automatic', 'manual', 'verified'

    -- Metadata
    deprecated_date TEXT DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,

    FOREIGN KEY(new_concept_id) REFERENCES concepts(id)
);

-- ============================================================================
-- 7. CONCEPT COMPOSITIONS TABLE
-- ============================================================================
-- How concepts combine (morpheme stacking)

CREATE TABLE concept_compositions (
    id INTEGER PRIMARY KEY,

    -- The composed result
    result_concept_id INTEGER NOT NULL,        -- The compound concept

    -- Components
    component_concept_id INTEGER NOT NULL,     -- A component concept
    position INTEGER NOT NULL,                 -- Order in composition (0-indexed)
    component_role TEXT,                       -- 'root', 'prefix', 'suffix', 'modifier'

    -- Metadata
    composition_type TEXT DEFAULT 'morphological',
    -- Values: 'morphological', 'compound', 'phrasal', 'semantic'

    FOREIGN KEY(result_concept_id) REFERENCES concepts(id),
    FOREIGN KEY(component_concept_id) REFERENCES concepts(id),
    UNIQUE(result_concept_id, position)
);

-- ============================================================================
-- 8. OUTPUT ROUTES TABLE
-- ============================================================================
-- Detailed routing configuration (extends concepts.modality)

CREATE TABLE output_routes (
    id INTEGER PRIMARY KEY,

    -- Pattern matching
    concept_id INTEGER,                        -- Specific concept trigger
    modality_pattern TEXT,                     -- Or match by modality
    context_pattern TEXT,                      -- Additional context matching

    -- Destination
    route_type TEXT NOT NULL,
    -- Values: 'fd', 'pipe', 'file', 'api', 'internal', 'display'

    destination TEXT NOT NULL,                 -- fd number, path, or API endpoint

    -- Permissions
    requires_permission TEXT,                  -- Permission level needed
    allowed_by_default INTEGER DEFAULT 1,

    -- Processing
    pre_process TEXT,                          -- JSON: transformations before routing
    post_process TEXT,                         -- JSON: transformations after routing

    -- Priority (for multiple matching routes)
    priority INTEGER DEFAULT 0,

    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

-- ============================================================================
-- 9. CONFIDENCE SOURCES TABLE
-- ============================================================================
-- Tracks where confidence values come from

CREATE TABLE confidence_sources (
    id INTEGER PRIMARY KEY,

    source_name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    -- Values: 'formal_grammar', 'corpus', 'nlp_tool', 'human', 'inferred'

    base_confidence REAL NOT NULL,             -- Default confidence for this source
    url TEXT,                                  -- Reference URL
    description TEXT
);

-- Insert standard confidence sources
INSERT INTO confidence_sources (source_name, source_type, base_confidence, description) VALUES
    ('gf', 'formal_grammar', 0.95, 'Grammatical Framework formal rules'),
    ('ud', 'corpus', 0.85, 'Universal Dependencies treebank'),
    ('unimorph', 'corpus', 0.85, 'UniMorph morphological paradigms'),
    ('wals', 'corpus', 0.70, 'World Atlas of Language Structures'),
    ('spacy', 'nlp_tool', 0.80, 'spaCy NLP analysis'),
    ('fasttext', 'nlp_tool', 0.60, 'fastText language identification'),
    ('deepl', 'nlp_tool', 0.75, 'DeepL API verification'),
    ('wiktionary', 'corpus', 0.80, 'Wiktionary etymology data'),
    ('taught', 'human', 0.90, 'Explicitly taught by human'),
    ('inferred', 'inferred', 0.40, 'Self-inferred from patterns'),
    ('family_inheritance', 'inferred', 0.60, 'Inherited from language family');

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Concepts table indexes (critical for tokenization)
CREATE INDEX idx_concepts_type ON concepts(concept_type);
CREATE INDEX idx_concepts_modality ON concepts(modality);
CREATE INDEX idx_concepts_domain ON concepts(domain);
CREATE INDEX idx_concepts_parent ON concepts(parent_concept_id);
CREATE INDEX idx_concepts_deprecated ON concepts(is_deprecated) WHERE is_deprecated = 1;
CREATE INDEX idx_concepts_canonical ON concepts(canonical);

-- Etymology indexes (THE critical tokenization lookup path)
CREATE INDEX idx_etymology_lookup ON etymology(language, surface_form);
CREATE INDEX idx_etymology_lookup_pos ON etymology(language, surface_form, pos_tag);
CREATE INDEX idx_etymology_concept ON etymology(concept_id);
CREATE INDEX idx_etymology_language ON etymology(language);
CREATE INDEX idx_etymology_morpheme ON etymology(morpheme_type);

-- Language families indexes
CREATE INDEX idx_families_parent ON language_families(parent_id);
CREATE INDEX idx_families_level ON language_families(level);
CREATE INDEX idx_families_iso ON language_families(iso_639_3);

-- Grammar rules indexes
CREATE INDEX idx_grammar_family ON grammar_rules(family_id);
CREATE INDEX idx_grammar_type ON grammar_rules(rule_type);
CREATE INDEX idx_grammar_inherited ON grammar_rules(inherited_from_id);

-- Deprecated tokens index
CREATE INDEX idx_deprecated_old ON deprecated_tokens(old_token_id);
CREATE INDEX idx_deprecated_surface ON deprecated_tokens(old_surface_form);

-- Compositions index
CREATE INDEX idx_compositions_result ON concept_compositions(result_concept_id);
CREATE INDEX idx_compositions_component ON concept_compositions(component_concept_id);

-- Output routes index
CREATE INDEX idx_routes_concept ON output_routes(concept_id);
CREATE INDEX idx_routes_modality ON output_routes(modality_pattern);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: All surface forms for a concept
CREATE VIEW v_concept_surface_forms AS
SELECT
    c.id AS concept_id,
    c.canonical,
    c.concept_type,
    c.modality,
    e.language,
    e.surface_form,
    e.pos_tag,
    e.frequency,
    e.confidence AS etymology_confidence,
    c.confidence AS concept_confidence
FROM concepts c
LEFT JOIN etymology e ON e.concept_id = c.id;

-- View: Grammar rules with inheritance chain
CREATE VIEW v_grammar_rules_resolved AS
WITH RECURSIVE rule_chain AS (
    SELECT
        gr.*,
        lf.name AS family_name,
        lf.level AS family_level,
        0 AS inheritance_depth
    FROM grammar_rules gr
    JOIN language_families lf ON gr.family_id = lf.id
    WHERE gr.inherited_from_id IS NULL

    UNION ALL

    SELECT
        gr.*,
        lf.name AS family_name,
        lf.level AS family_level,
        rc.inheritance_depth + 1
    FROM grammar_rules gr
    JOIN language_families lf ON gr.family_id = lf.id
    JOIN rule_chain rc ON gr.inherited_from_id = rc.id
)
SELECT * FROM rule_chain;

-- View: Deprecated token migration status
CREATE VIEW v_migration_status AS
SELECT
    dt.old_token_id,
    dt.old_surface_form,
    dt.new_concept_id,
    c.canonical AS new_canonical,
    dt.mapping_type,
    dt.confidence,
    CASE
        WHEN dt.new_concept_id IS NOT NULL THEN 'migrated'
        WHEN dt.mapping_type = 'obsolete' THEN 'obsolete'
        ELSE 'pending'
    END AS status
FROM deprecated_tokens dt
LEFT JOIN concepts c ON dt.new_concept_id = c.id;

-- ============================================================================
-- SEED DATA: System Primitives (0-999,999 range)
-- ============================================================================

-- Control tokens (0-999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, confidence, source) VALUES
    (0, '<pad>', 'control', 'internal', 'system', 1.0, 'system'),
    (1, '<unk>', 'control', 'internal', 'system', 1.0, 'system'),
    (2, '<s>', 'control', 'internal', 'system', 1.0, 'system'),
    (3, '</s>', 'control', 'internal', 'system', 1.0, 'system'),
    (4, '<mask>', 'control', 'internal', 'system', 1.0, 'system'),
    (5, '<escape>', 'control', 'internal', 'system', 1.0, 'system');

-- Thinking tokens (1,000-99,999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, confidence, source) VALUES
    (1000, '<think>', 'thinking', 'thinking', 'reasoning', 1.0, 'system'),
    (1001, '</think>', 'thinking', 'thinking', 'reasoning', 1.0, 'system'),
    (1002, '<scratchpad>', 'thinking', 'thinking', 'reasoning', 1.0, 'system'),
    (1003, '</scratchpad>', 'thinking', 'thinking', 'reasoning', 1.0, 'system'),
    (1004, '<reflect>', 'thinking', 'thinking', 'reasoning', 1.0, 'system'),
    (1005, '</reflect>', 'thinking', 'thinking', 'reasoning', 1.0, 'system');

-- Tool tokens (100,000-199,999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, confidence, source) VALUES
    (100000, '<tool_call>', 'tool', 'tool_call', 'system', 1.0, 'system'),
    (100001, '</tool_call>', 'tool', 'tool_call', 'system', 1.0, 'system'),
    (100002, '<tool_result>', 'tool', 'tool_result', 'system', 1.0, 'system'),
    (100003, '</tool_result>', 'tool', 'tool_result', 'system', 1.0, 'system'),
    (100004, '<tool_error>', 'tool', 'tool_result', 'system', 1.0, 'system'),
    (100005, '</tool_error>', 'tool', 'tool_result', 'system', 1.0, 'system');

-- Role markers (200,000-299,999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, confidence, source) VALUES
    (200000, '<user>', 'role', 'display', 'conversation', 1.0, 'system'),
    (200001, '</user>', 'role', 'display', 'conversation', 1.0, 'system'),
    (200002, '<assistant>', 'role', 'display', 'conversation', 1.0, 'system'),
    (200003, '</assistant>', 'role', 'display', 'conversation', 1.0, 'system'),
    (200004, '<system>', 'role', 'internal', 'conversation', 1.0, 'system'),
    (200005, '</system>', 'role', 'internal', 'conversation', 1.0, 'system');

-- Modality routing tokens (300,000-499,999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, target_language, target_format, confidence, source) VALUES
    (300000, '<text:en>', 'modality', 'text', 'language', 'en', NULL, 1.0, 'system'),
    (300001, '<text:de>', 'modality', 'text', 'language', 'de', NULL, 1.0, 'system'),
    (300002, '<text:fr>', 'modality', 'text', 'language', 'fr', NULL, 1.0, 'system'),
    (300100, '<code:python>', 'modality', 'code', 'programming', NULL, 'python', 1.0, 'system'),
    (300101, '<code:sql>', 'modality', 'code', 'programming', NULL, 'sql', 1.0, 'system'),
    (300102, '<code:javascript>', 'modality', 'code', 'programming', NULL, 'javascript', 1.0, 'system'),
    (300200, '<physics:body>', 'modality', 'physics', 'perception', NULL, 'body', 1.0, 'system'),
    (300201, '<physics:env>', 'modality', 'physics', 'perception', NULL, 'environment', 1.0, 'system'),
    (300202, '<physics:contact>', 'modality', 'physics', 'perception', NULL, 'contact', 1.0, 'system');

-- Universal modifiers (2,000,000-2,999,999)
INSERT INTO concepts (id, canonical, concept_type, modality, domain, subdomain, composable, confidence, source, description) VALUES
    (2000001, 'STATE_OF', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'Converts X to "state of being X" (-ness, -heit, -ость)'),
    (2000002, 'QUALITY_OF', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'Converts X to quality/property (-ful, -ous, -lich)'),
    (2000003, 'AGENT_OF', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'One who does X (-er, -or, -ist)'),
    (2000004, 'ACT_OF', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'The act of doing X (-ing, -ation, -ung)'),
    (2000005, 'NEGATION', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'Not X, opposite of X (un-, in-, не-)'),
    (2000006, 'REPETITION', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'Again, repeated (re-, wieder-, пере-)'),
    (2000007, 'CAUSATIVE', 'modifier', 'text', 'abstract', 'modifier', 1, 0.95, 'gf', 'To cause to be X (-ify, -ize, -en)'),
    (2000008, 'DIMINUTIVE', 'modifier', 'text', 'abstract', 'modifier', 1, 0.90, 'gf', 'Small version of X (-let, -chen, -ik)'),
    (2000009, 'AUGMENTATIVE', 'modifier', 'text', 'abstract', 'modifier', 1, 0.90, 'gf', 'Large version of X'),
    (2000010, 'COLLECTIVE', 'modifier', 'text', 'abstract', 'modifier', 1, 0.90, 'gf', 'Group of X (-hood, -schaft)');
