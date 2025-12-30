-- Language DB Schema: Synset-Based Token Organization
-- The DB IS the tokenizer. Every word is a query.
--
-- Token ID Ranges:
--   0-999:         Reserved/System tokens
--   1000-2999:     Modifier tokens (grammar: tense, number, case, etc.)
--   3000000+:      Concept tokens (synset-based, 128 slots per synset)
--
-- Token ID for concept in synset:
--   token_id = 3000000 + (synset_id * 128) + concept_offset
--   This allows 128 concepts per synset with room for additions

PRAGMA foreign_keys = ON;

-- =============================================================================
-- SYNSETS: Concept clusters (the core organizational unit)
-- =============================================================================
CREATE TABLE IF NOT EXISTS synsets (
    synset_id INTEGER PRIMARY KEY,           -- Internal ID, maps to token range
    gloss TEXT NOT NULL,                     -- Definition/meaning
    pos TEXT,                                -- Part of speech (noun, verb, adj, adv)
    domain TEXT,                             -- Semantic domain (physics, biology, etc.)
    source TEXT DEFAULT 'kaikki',            -- Data source
    source_id TEXT,                          -- Original ID from source (WordNet synset ID, etc.)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Synset relationships (hypernym, hyponym, meronym, etc.)
CREATE TABLE IF NOT EXISTS synset_relations (
    id INTEGER PRIMARY KEY,
    synset_id INTEGER NOT NULL,
    related_synset_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,             -- hypernym, hyponym, meronym, holonym, antonym, similar
    weight REAL DEFAULT 1.0,                 -- Relation strength
    FOREIGN KEY (synset_id) REFERENCES synsets(synset_id),
    FOREIGN KEY (related_synset_id) REFERENCES synsets(synset_id)
);
CREATE INDEX IF NOT EXISTS idx_synset_rel_synset ON synset_relations(synset_id);
CREATE INDEX IF NOT EXISTS idx_synset_rel_related ON synset_relations(related_synset_id);
CREATE INDEX IF NOT EXISTS idx_synset_rel_type ON synset_relations(relation_type);

-- =============================================================================
-- CONCEPTS: Individual meanings within synsets
-- =============================================================================
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,          -- Globally unique concept ID
    synset_id INTEGER NOT NULL,              -- Parent synset
    concept_offset INTEGER NOT NULL,         -- Position within synset (0-127)
    lemma TEXT NOT NULL,                     -- Canonical form (e.g., "run")
    gloss TEXT,                              -- Specific meaning if different from synset
    lang TEXT DEFAULT 'en',                  -- Language code (ISO 639-3)
    frequency INTEGER DEFAULT 0,             -- Usage frequency (for caching priority)
    FOREIGN KEY (synset_id) REFERENCES synsets(synset_id),
    UNIQUE(synset_id, concept_offset)
);
CREATE INDEX IF NOT EXISTS idx_concept_synset ON concepts(synset_id);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_lang ON concepts(lang);

-- Computed token ID view (for clarity, actual computation in code)
-- token_id = 3000000 + (synset_id * 128) + concept_offset
CREATE VIEW IF NOT EXISTS concept_tokens AS
SELECT
    concept_id,
    synset_id,
    concept_offset,
    lemma,
    3000000 + (synset_id * 128) + concept_offset AS token_id
FROM concepts;

-- =============================================================================
-- SURFACE FORMS: All word variants that map to concepts
-- =============================================================================
CREATE TABLE IF NOT EXISTS surface_forms (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,              -- Actual string (e.g., "running", "ran")
    concept_id INTEGER NOT NULL,             -- Links to concept
    lang TEXT DEFAULT 'en',                  -- Language code
    form_type TEXT,                          -- lemma, inflected, compound, derived
    pos_features TEXT,                       -- Morphological features (e.g., "V;PST;3;SG")
    frequency INTEGER DEFAULT 0,             -- Usage frequency
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);
CREATE INDEX IF NOT EXISTS idx_surface_form ON surface_forms(surface_form);
CREATE INDEX IF NOT EXISTS idx_surface_concept ON surface_forms(concept_id);
CREATE INDEX IF NOT EXISTS idx_surface_lang ON surface_forms(lang);
CREATE INDEX IF NOT EXISTS idx_surface_freq ON surface_forms(frequency DESC);

-- =============================================================================
-- MODIFIERS: Grammar tokens (occupy token range 1000-2999)
-- =============================================================================
CREATE TABLE IF NOT EXISTS modifiers (
    modifier_id INTEGER PRIMARY KEY,         -- Token ID (1000-2999)
    category TEXT NOT NULL,                  -- tense, aspect, number, case, gender, etc.
    name TEXT NOT NULL,                      -- present, past, singular, plural, etc.
    symbol TEXT,                             -- Short notation (e.g., "PST", "PL")
    lang TEXT,                               -- NULL = universal, otherwise language-specific
    description TEXT,
    UNIQUE(category, name, lang)
);
CREATE INDEX IF NOT EXISTS idx_modifier_category ON modifiers(category);
CREATE INDEX IF NOT EXISTS idx_modifier_symbol ON modifiers(symbol);

-- Seed universal grammar modifiers
INSERT OR IGNORE INTO modifiers (modifier_id, category, name, symbol, description) VALUES
    -- Tense (1000-1049)
    (1000, 'tense', 'present', 'PRS', 'Present tense'),
    (1001, 'tense', 'past', 'PST', 'Past tense'),
    (1002, 'tense', 'future', 'FUT', 'Future tense'),
    (1003, 'tense', 'perfect', 'PRF', 'Perfect aspect'),
    (1004, 'tense', 'imperfect', 'IPFV', 'Imperfective aspect'),
    (1005, 'tense', 'progressive', 'PROG', 'Progressive/continuous aspect'),

    -- Number (1050-1099)
    (1050, 'number', 'singular', 'SG', 'Singular number'),
    (1051, 'number', 'plural', 'PL', 'Plural number'),
    (1052, 'number', 'dual', 'DU', 'Dual number'),
    (1053, 'number', 'paucal', 'PAU', 'Paucal (few) number'),

    -- Person (1100-1149)
    (1100, 'person', 'first', '1', 'First person'),
    (1101, 'person', 'second', '2', 'Second person'),
    (1102, 'person', 'third', '3', 'Third person'),

    -- Case (1150-1199)
    (1150, 'case', 'nominative', 'NOM', 'Nominative case'),
    (1151, 'case', 'accusative', 'ACC', 'Accusative case'),
    (1152, 'case', 'dative', 'DAT', 'Dative case'),
    (1153, 'case', 'genitive', 'GEN', 'Genitive case'),
    (1154, 'case', 'locative', 'LOC', 'Locative case'),
    (1155, 'case', 'instrumental', 'INS', 'Instrumental case'),
    (1156, 'case', 'vocative', 'VOC', 'Vocative case'),

    -- Voice (1200-1249)
    (1200, 'voice', 'active', 'ACT', 'Active voice'),
    (1201, 'voice', 'passive', 'PASS', 'Passive voice'),
    (1202, 'voice', 'middle', 'MID', 'Middle voice'),

    -- Mood (1250-1299)
    (1250, 'mood', 'indicative', 'IND', 'Indicative mood'),
    (1251, 'mood', 'subjunctive', 'SBJV', 'Subjunctive mood'),
    (1252, 'mood', 'imperative', 'IMP', 'Imperative mood'),
    (1253, 'mood', 'conditional', 'COND', 'Conditional mood'),

    -- Gender (1300-1349)
    (1300, 'gender', 'masculine', 'MASC', 'Masculine gender'),
    (1301, 'gender', 'feminine', 'FEM', 'Feminine gender'),
    (1302, 'gender', 'neuter', 'NEUT', 'Neuter gender'),

    -- Definiteness (1350-1399)
    (1350, 'definiteness', 'definite', 'DEF', 'Definite article/marker'),
    (1351, 'definiteness', 'indefinite', 'INDEF', 'Indefinite article/marker'),

    -- Degree (1400-1449)
    (1400, 'degree', 'positive', 'POS', 'Positive degree'),
    (1401, 'degree', 'comparative', 'CMPR', 'Comparative degree'),
    (1402, 'degree', 'superlative', 'SPRL', 'Superlative degree'),

    -- Negation (1450-1499)
    (1450, 'polarity', 'affirmative', 'AFF', 'Affirmative polarity'),
    (1451, 'polarity', 'negative', 'NEG', 'Negative polarity');

-- =============================================================================
-- TRANSLATIONS: Cross-language concept links
-- =============================================================================
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    target_lang TEXT NOT NULL,               -- Target language code
    translation TEXT NOT NULL,               -- Translated form
    confidence REAL DEFAULT 1.0,             -- Translation confidence
    source TEXT,                             -- Source of translation
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);
CREATE INDEX IF NOT EXISTS idx_trans_concept ON translations(concept_id);
CREATE INDEX IF NOT EXISTS idx_trans_lang ON translations(target_lang);
CREATE INDEX IF NOT EXISTS idx_trans_text ON translations(translation);

-- =============================================================================
-- ETYMOLOGY: Word origin tracking (optional, for research)
-- =============================================================================
CREATE TABLE IF NOT EXISTS etymology (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    origin_lang TEXT,                        -- Proto-language or source language
    origin_form TEXT,                        -- Original form
    etymology_text TEXT,                     -- Full etymology description
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);
CREATE INDEX IF NOT EXISTS idx_etym_concept ON etymology(concept_id);

-- =============================================================================
-- METADATA: Import tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS import_metadata (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    import_date TEXT DEFAULT CURRENT_TIMESTAMP,
    file_hash TEXT,
    record_count INTEGER,
    notes TEXT
);

-- =============================================================================
-- HELPER FUNCTIONS (as views)
-- =============================================================================

-- Surface form to token lookup
-- Usage: SELECT * FROM surface_to_token WHERE surface_form = 'running';
CREATE VIEW IF NOT EXISTS surface_to_token AS
SELECT
    sf.surface_form,
    sf.pos_features,
    c.lemma,
    c.concept_id,
    c.synset_id,
    3000000 + (c.synset_id * 128) + c.concept_offset AS token_id,
    s.gloss
FROM surface_forms sf
JOIN concepts c ON sf.concept_id = c.concept_id
JOIN synsets s ON c.synset_id = s.synset_id;

-- Token to surface form lookup
-- Usage: SELECT * FROM token_to_surface WHERE token_id = 3000128;
CREATE VIEW IF NOT EXISTS token_to_surface AS
SELECT
    3000000 + (c.synset_id * 128) + c.concept_offset AS token_id,
    c.lemma,
    c.concept_id,
    c.synset_id,
    sf.surface_form,
    sf.form_type,
    sf.pos_features
FROM concepts c
JOIN surface_forms sf ON c.concept_id = sf.concept_id;

-- Synset neighborhood (for prefetching related concepts)
CREATE VIEW IF NOT EXISTS synset_neighborhood AS
SELECT
    s1.synset_id AS source_synset,
    s1.gloss AS source_gloss,
    sr.relation_type,
    s2.synset_id AS related_synset,
    s2.gloss AS related_gloss
FROM synsets s1
JOIN synset_relations sr ON s1.synset_id = sr.synset_id
JOIN synsets s2 ON sr.related_synset_id = s2.synset_id;
