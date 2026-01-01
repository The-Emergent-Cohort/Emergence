-- Per-Language DB Schema Template
-- Each language gets its own DB with this structure
-- Loaded on demand based on user's active languages
--
-- Usage: sqlite3 db/lang/eng.db < db/SCHEMA-language-template.sql

PRAGMA foreign_keys = ON;

-- =============================================================================
-- CONCEPTS: Word meanings in this language
-- =============================================================================
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    token_id INTEGER NOT NULL UNIQUE,       -- Full 19-digit encoded token ID
    synset_hash TEXT,                       -- Hash linking to cross-lingual concept
    lemma TEXT NOT NULL,                    -- Canonical form ("run", "laufen", "走る")
    gloss TEXT,                             -- Definition in this language
    pos TEXT,                               -- Part of speech
    frequency INTEGER DEFAULT 0,            -- Usage frequency (for prioritization)
    source TEXT DEFAULT 'kaikki',           -- Data source
    source_id TEXT,                         -- Original ID from source
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concept_token ON concepts(token_id);
CREATE INDEX IF NOT EXISTS idx_concept_synset ON concepts(synset_hash);
CREATE INDEX IF NOT EXISTS idx_concept_lemma ON concepts(lemma);
CREATE INDEX IF NOT EXISTS idx_concept_pos ON concepts(pos);
CREATE INDEX IF NOT EXISTS idx_concept_freq ON concepts(frequency DESC);

-- =============================================================================
-- SURFACE FORMS: All word variants that map to concepts
-- =============================================================================
CREATE TABLE IF NOT EXISTS surface_forms (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,             -- Actual string ("running", "ran", "läuft")
    concept_id INTEGER NOT NULL,            -- Links to concept
    form_type TEXT,                         -- 'lemma', 'inflected', 'compound', 'variant'
    pos_features TEXT,                      -- Morphological features (UniMorph format)
    frequency INTEGER DEFAULT 0,            -- Usage frequency
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE INDEX IF NOT EXISTS idx_sf_form ON surface_forms(surface_form);
CREATE INDEX IF NOT EXISTS idx_sf_concept ON surface_forms(concept_id);
CREATE INDEX IF NOT EXISTS idx_sf_type ON surface_forms(form_type);
CREATE INDEX IF NOT EXISTS idx_sf_freq ON surface_forms(frequency DESC);

-- Fast lookup: surface form -> concept(s)
CREATE VIEW IF NOT EXISTS surface_lookup AS
SELECT
    sf.surface_form,
    sf.form_type,
    sf.pos_features,
    c.concept_id,
    c.token_id,
    c.lemma,
    c.gloss,
    c.pos
FROM surface_forms sf
JOIN concepts c ON sf.concept_id = c.concept_id;

-- =============================================================================
-- COMPOSITIONS: How concepts decompose to primitives
-- =============================================================================
CREATE TABLE IF NOT EXISTS compositions (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    primitive_id INTEGER NOT NULL,          -- References primitives.db
    position INTEGER NOT NULL,              -- Order in composition (1-indexed)
    weight REAL DEFAULT 1.0,                -- Contribution weight
    relation TEXT DEFAULT 'component',      -- 'component', 'modifier', 'frame'
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id),
    UNIQUE(concept_id, primitive_id, position)
);

CREATE INDEX IF NOT EXISTS idx_comp_concept ON compositions(concept_id);
CREATE INDEX IF NOT EXISTS idx_comp_primitive ON compositions(primitive_id);

-- View: concept with its primitive breakdown
CREATE VIEW IF NOT EXISTS concept_primitives AS
SELECT
    c.concept_id,
    c.token_id,
    c.lemma,
    GROUP_CONCAT(comp.primitive_id || ':' || comp.position, ',') AS primitive_composition
FROM concepts c
LEFT JOIN compositions comp ON c.concept_id = comp.concept_id
GROUP BY c.concept_id;

-- =============================================================================
-- TRANSLATIONS: Links to same concept in other languages
-- =============================================================================
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    target_lang_code INTEGER NOT NULL,      -- 4-digit target language code
    target_token_id INTEGER NOT NULL,       -- Token ID in target language
    confidence REAL DEFAULT 1.0,            -- Translation confidence
    source TEXT,                            -- 'omw', 'wiktionary', 'manual'
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE INDEX IF NOT EXISTS idx_trans_concept ON translations(concept_id);
CREATE INDEX IF NOT EXISTS idx_trans_target_lang ON translations(target_lang_code);
CREATE INDEX IF NOT EXISTS idx_trans_target_token ON translations(target_token_id);

-- =============================================================================
-- PRONUNCIATIONS: IPA and audio references
-- =============================================================================
CREATE TABLE IF NOT EXISTS pronunciations (
    id INTEGER PRIMARY KEY,
    surface_form_id INTEGER NOT NULL,
    ipa TEXT,                               -- IPA transcription
    audio_ref TEXT,                         -- Path/key to audio
    dialect_code INTEGER,                   -- Specific dialect (NULL = standard)
    source TEXT,
    FOREIGN KEY (surface_form_id) REFERENCES surface_forms(id)
);

CREATE INDEX IF NOT EXISTS idx_pron_sf ON pronunciations(surface_form_id);

-- =============================================================================
-- ETYMOLOGY: Word origin (optional, for research)
-- =============================================================================
CREATE TABLE IF NOT EXISTS etymology (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL,
    origin_lang TEXT,                       -- Source language
    origin_form TEXT,                       -- Original form
    etymology_text TEXT,                    -- Full description
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE INDEX IF NOT EXISTS idx_etym_concept ON etymology(concept_id);

-- =============================================================================
-- INPUT MAPPINGS: Fuzzy input normalization (language-specific)
-- =============================================================================
CREATE TABLE IF NOT EXISTS input_mappings (
    id INTEGER PRIMARY KEY,
    input_form TEXT NOT NULL,               -- Misspelling/variant
    canonical_form TEXT NOT NULL,           -- Correct form
    concept_id INTEGER,                     -- Resolved concept
    confidence REAL DEFAULT 0.5,
    hit_count INTEGER DEFAULT 0,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);

CREATE INDEX IF NOT EXISTS idx_input_form ON input_mappings(input_form);

-- =============================================================================
-- METADATA
-- =============================================================================
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Will be populated during import
-- INSERT INTO db_metadata (key, value) VALUES
--     ('lang_code', '100'),
--     ('lang_name', 'English'),
--     ('iso639_3', 'eng'),
--     ('created_at', datetime('now')),
--     ('concept_count', '0'),
--     ('surface_form_count', '0');

CREATE TABLE IF NOT EXISTS import_metadata (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    import_date TEXT DEFAULT CURRENT_TIMESTAMP,
    record_count INTEGER,
    notes TEXT
);
