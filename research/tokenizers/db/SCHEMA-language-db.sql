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
-- LANGUAGES: Language inventory and metadata (from Glottolog)
-- =============================================================================
CREATE TABLE IF NOT EXISTS languages (
    lang_code TEXT PRIMARY KEY,              -- ISO 639-3 code (e.g., "eng", "deu", "jpn")
    glottocode TEXT UNIQUE,                  -- Glottolog code (e.g., "stan1293")
    name TEXT NOT NULL,                      -- Language name
    family_id TEXT,                          -- Parent family glottocode
    level TEXT,                              -- family, language, dialect
    macroarea TEXT,                          -- Africa, Eurasia, etc.
    latitude REAL,                           -- Geographic center
    longitude REAL,
    status TEXT,                             -- living, extinct, ancient, constructed
    speaker_count INTEGER,                   -- Estimated speakers (for prioritization)
    FOREIGN KEY (family_id) REFERENCES language_families(glottocode)
);
CREATE INDEX IF NOT EXISTS idx_lang_glottocode ON languages(glottocode);
CREATE INDEX IF NOT EXISTS idx_lang_family ON languages(family_id);
CREATE INDEX IF NOT EXISTS idx_lang_level ON languages(level);

-- =============================================================================
-- LANGUAGE FAMILIES: Genealogical hierarchy (from Glottolog)
-- =============================================================================
CREATE TABLE IF NOT EXISTS language_families (
    glottocode TEXT PRIMARY KEY,             -- Glottolog family code
    name TEXT NOT NULL,                      -- Family name (e.g., "Indo-European")
    parent_id TEXT,                          -- Parent family (for hierarchy)
    level INTEGER DEFAULT 0,                 -- Depth in tree (0 = top-level)
    FOREIGN KEY (parent_id) REFERENCES language_families(glottocode)
);
CREATE INDEX IF NOT EXISTS idx_family_parent ON language_families(parent_id);

-- =============================================================================
-- FEATURE DEFINITIONS: What each typological feature means (WALS + Grambank)
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_definitions (
    feature_id TEXT PRIMARY KEY,             -- e.g., "81A" (WALS) or "GB020" (Grambank)
    name TEXT NOT NULL,                      -- e.g., "Order of Subject, Object and Verb"
    description TEXT,                        -- Full description
    domain TEXT,                             -- word_order, morphology, phonology, etc.
    source TEXT NOT NULL,                    -- wals, grambank
    possible_values TEXT                     -- JSON array of valid values
);
CREATE INDEX IF NOT EXISTS idx_feature_domain ON feature_definitions(domain);
CREATE INDEX IF NOT EXISTS idx_feature_source ON feature_definitions(source);

-- =============================================================================
-- LANGUAGE FEATURES: Typological values per language (WALS + Grambank)
-- =============================================================================
CREATE TABLE IF NOT EXISTS language_features (
    id INTEGER PRIMARY KEY,
    lang_code TEXT NOT NULL,                 -- ISO 639-3 or glottocode
    feature_id TEXT NOT NULL,                -- Links to feature_definitions
    value TEXT NOT NULL,                     -- The actual value (e.g., "SVO", "1", "0")
    value_name TEXT,                         -- Human-readable value (e.g., "Subject-Verb-Object")
    source TEXT NOT NULL,                    -- wals, grambank
    confidence REAL DEFAULT 1.0,             -- Data quality
    FOREIGN KEY (lang_code) REFERENCES languages(lang_code),
    FOREIGN KEY (feature_id) REFERENCES feature_definitions(feature_id),
    UNIQUE(lang_code, feature_id, source)
);
CREATE INDEX IF NOT EXISTS idx_langfeat_lang ON language_features(lang_code);
CREATE INDEX IF NOT EXISTS idx_langfeat_feature ON language_features(feature_id);

-- Seed key WALS feature definitions (most relevant for tokenization)
INSERT OR IGNORE INTO feature_definitions (feature_id, name, domain, source, possible_values) VALUES
    -- Word Order features
    ('81A', 'Order of Subject, Object and Verb', 'word_order', 'wals',
     '["SOV","SVO","VSO","VOS","OVS","OSV","No dominant order"]'),
    ('82A', 'Order of Subject and Verb', 'word_order', 'wals',
     '["SV","VS","No dominant order"]'),
    ('83A', 'Order of Object and Verb', 'word_order', 'wals',
     '["OV","VO","No dominant order"]'),
    ('85A', 'Order of Adposition and Noun Phrase', 'word_order', 'wals',
     '["Postpositions","Prepositions","Inpositions","No dominant order"]'),
    ('86A', 'Order of Genitive and Noun', 'word_order', 'wals',
     '["Genitive-Noun","Noun-Genitive","No dominant order"]'),
    ('87A', 'Order of Adjective and Noun', 'word_order', 'wals',
     '["Adjective-Noun","Noun-Adjective","No dominant order","Only internally-headed relative clauses"]'),

    -- Morphology features
    ('20A', 'Fusion of Selected Inflectional Formatives', 'morphology', 'wals',
     '["Exclusively concatenative","Predominantly concatenative","Predominantly isolating","Exclusively isolating"]'),
    ('21A', 'Exponence of Selected Inflectional Formatives', 'morphology', 'wals',
     '["Monoexponential case","Case+number","Case+referentiality","No case"]'),
    ('22A', 'Inflectional Synthesis of the Verb', 'morphology', 'wals',
     '["0-1 categories","2-3 categories","4-5 categories","6-7 categories","8-9 categories","10-11 categories","12-13 categories"]'),
    ('26A', 'Prefixing vs. Suffixing in Inflectional Morphology', 'morphology', 'wals',
     '["Strongly suffixing","Weakly suffixing","Equal prefixing and suffixing","Weakly prefixing","Strongly prefixing","Little affixation"]'),

    -- Case and Agreement
    ('28A', 'Case Syncretism', 'case', 'wals',
     '["No case marking","Core cases only","Core and non-core","Non-core only"]'),
    ('49A', 'Number of Cases', 'case', 'wals',
     '["No morphological case-marking","2 cases","3 cases","4 cases","5 cases","6-7 cases","8-9 cases","10 or more cases"]'),
    ('51A', 'Position of Case Affixes', 'case', 'wals',
     '["Case suffixes","Case prefixes","Case tone","Mixed"]'),

    -- Nominal features
    ('30A', 'Number of Genders', 'nominal', 'wals',
     '["None","Two","Three","Four","Five or more"]'),
    ('33A', 'Coding of Nominal Plurality', 'nominal', 'wals',
     '["Plural prefix","Plural suffix","Plural stem change","Plural clitic","Plural word","Mixed morphological plural","No plural"]'),
    ('34A', 'Occurrence of Nominal Plurality', 'nominal', 'wals',
     '["All nouns, always optional","All nouns, always obligatory","All nouns, optionality unclear","Obligatory in some, optional in others"]'),

    -- Verbal features
    ('65A', 'Perfective/Imperfective Aspect', 'verbal', 'wals',
     '["Grammatical marking","No grammatical marking"]'),
    ('66A', 'The Past Tense', 'verbal', 'wals',
     '["Past vs. non-past","No past tense","No tense marking","Present vs. non-present"]'),
    ('69A', 'Position of Tense-Aspect Affixes', 'verbal', 'wals',
     '["Tense-aspect suffixes","Tense-aspect prefixes","Tense-aspect tone","Mixed"]'),
    ('70A', 'The Morphological Imperative', 'verbal', 'wals',
     '["Second singular and second plural","Second person only","Second singular","Second and third persons","All persons"]'),

    -- Negation
    ('112A', 'Negative Morphemes', 'negation', 'wals',
     '["Negative affix","Negative particle","Negative auxiliary verb","Negative word, unclear if verb or particle","Variation between negative word and affix"]'),
    ('143A', 'Order of Negative Morpheme and Verb', 'negation', 'wals',
     '["NegV","VNeg","[Neg-V]","[V-Neg]","Negative tone","Mixed"]');

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

-- Language profile (all features for a language)
-- Usage: SELECT * FROM language_profile WHERE lang_code = 'deu';
CREATE VIEW IF NOT EXISTS language_profile AS
SELECT
    l.lang_code,
    l.name AS language_name,
    l.family_id,
    f.name AS family_name,
    fd.feature_id,
    fd.name AS feature_name,
    fd.domain,
    lf.value,
    lf.value_name
FROM languages l
LEFT JOIN language_families f ON l.family_id = f.glottocode
LEFT JOIN language_features lf ON l.lang_code = lf.lang_code
LEFT JOIN feature_definitions fd ON lf.feature_id = fd.feature_id;

-- Language family tree (recursive traversal helper)
-- Usage: WITH RECURSIVE ... to walk the tree
CREATE VIEW IF NOT EXISTS family_tree AS
SELECT
    lf.glottocode,
    lf.name,
    lf.parent_id,
    lf.level,
    p.name AS parent_name
FROM language_families lf
LEFT JOIN language_families p ON lf.parent_id = p.glottocode;

-- =============================================================================
-- TOKEN INDEX: Bootstrap table for fast token → synset lookups
-- This is the "first query" that provides keys for all subsequent queries
-- =============================================================================
CREATE TABLE IF NOT EXISTS token_index (
    token_id INTEGER PRIMARY KEY,            -- The handle
    synset_id INTEGER NOT NULL,              -- Family key for weight queries
    ref_table TEXT NOT NULL,                 -- Source table: 'concepts', 'entities', 'context'
    ref_id INTEGER NOT NULL,                 -- Row ID in that table
    created_by TEXT DEFAULT 'import',        -- 'import', 'di', 'user'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (synset_id) REFERENCES synsets(synset_id)
);
CREATE INDEX IF NOT EXISTS idx_token_synset ON token_index(synset_id);
CREATE INDEX IF NOT EXISTS idx_token_ref ON token_index(ref_table, ref_id);

-- =============================================================================
-- INPUT NORMALIZATION: Fuzzy matching, misspellings, user patterns
-- Input-only mappings: resolve noisy input to clean concepts
-- Never used for output generation (one-way valve)
-- =============================================================================
CREATE TABLE IF NOT EXISTS input_mappings (
    id INTEGER PRIMARY KEY,
    input_form TEXT NOT NULL,                -- The misspelling/variant (e.g., "teh", "definately")
    canonical_form TEXT NOT NULL,            -- Correct form (e.g., "the", "definitely")
    concept_id INTEGER,                      -- Resolved concept (if known)
    confidence REAL DEFAULT 0.5,             -- Confidence in mapping (0-1)
    source TEXT DEFAULT 'detected',          -- 'common', 'user', 'detected', 'di'
    user_id TEXT,                            -- NULL = global, otherwise user-specific
    hit_count INTEGER DEFAULT 0,             -- Times this mapping was used
    last_hit TEXT,                           -- Last usage timestamp
    confirmed INTEGER DEFAULT 0,             -- 1 = confirmed by context/user
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id)
);
CREATE INDEX IF NOT EXISTS idx_input_form ON input_mappings(input_form);
CREATE INDEX IF NOT EXISTS idx_input_user ON input_mappings(user_id);
CREATE INDEX IF NOT EXISTS idx_input_confidence ON input_mappings(confidence DESC);

-- Ambiguous input candidates (when multiple interpretations possible)
CREATE TABLE IF NOT EXISTS input_candidates (
    id INTEGER PRIMARY KEY,
    input_form TEXT NOT NULL,                -- The ambiguous input
    candidate_concept_id INTEGER NOT NULL,   -- Possible interpretation
    probability REAL DEFAULT 0.5,            -- P(this interpretation | input)
    context_hint TEXT,                       -- What context favors this interpretation
    FOREIGN KEY (candidate_concept_id) REFERENCES concepts(concept_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_form ON input_candidates(input_form);

-- =============================================================================
-- CONTEXT TOKENS: DI-created refinements and learned associations
-- These are tokens/weight-clouds that DI adds through learning
-- =============================================================================
CREATE TABLE IF NOT EXISTS context_tokens (
    context_token_id INTEGER PRIMARY KEY,    -- Unique ID for this context token
    name TEXT NOT NULL,                      -- Human-readable name
    description TEXT,                        -- What this context represents
    parent_synset_id INTEGER,                -- Base synset this refines (if any)
    token_type TEXT DEFAULT 'refinement',    -- 'refinement', 'composite', 'learned', 'user_defined'
    created_by TEXT DEFAULT 'di',            -- 'di', 'user', 'system'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_activated TEXT,                     -- Last time this was used
    activation_count INTEGER DEFAULT 0,      -- Usage frequency
    active INTEGER DEFAULT 1,                -- 0 = deprecated/disabled
    FOREIGN KEY (parent_synset_id) REFERENCES synsets(synset_id)
);
CREATE INDEX IF NOT EXISTS idx_context_parent ON context_tokens(parent_synset_id);
CREATE INDEX IF NOT EXISTS idx_context_type ON context_tokens(token_type);
CREATE INDEX IF NOT EXISTS idx_context_active ON context_tokens(active);

-- Context token composition (for composite tokens built from multiple concepts)
CREATE TABLE IF NOT EXISTS context_composition (
    id INTEGER PRIMARY KEY,
    context_token_id INTEGER NOT NULL,       -- The composite token
    component_synset_id INTEGER NOT NULL,    -- A component synset
    weight REAL DEFAULT 1.0,                 -- Weight of this component
    relation TEXT DEFAULT 'part_of',         -- 'part_of', 'modifies', 'contrasts', etc.
    FOREIGN KEY (context_token_id) REFERENCES context_tokens(context_token_id),
    FOREIGN KEY (component_synset_id) REFERENCES synsets(synset_id)
);
CREATE INDEX IF NOT EXISTS idx_composition_token ON context_composition(context_token_id);
CREATE INDEX IF NOT EXISTS idx_composition_component ON context_composition(component_synset_id);

-- =============================================================================
-- WEIGHT REFERENCES: Pointers to weight storage (organized by synset)
-- The weights themselves are in a separate DB, this tracks what exists
-- =============================================================================
CREATE TABLE IF NOT EXISTS weight_refs (
    synset_id INTEGER PRIMARY KEY,           -- Which synset these weights are for
    weight_db TEXT NOT NULL,                 -- Which weight database file
    layer_count INTEGER,                     -- Number of layers stored
    total_size INTEGER,                      -- Total bytes for this synset's weights
    last_updated TEXT,                       -- When weights were last modified
    checksum TEXT,                           -- For integrity verification
    FOREIGN KEY (synset_id) REFERENCES synsets(synset_id)
);

-- Language interpretation weights (combinatorial, by feature family)
CREATE TABLE IF NOT EXISTS lang_weight_refs (
    id INTEGER PRIMARY KEY,
    feature_family TEXT NOT NULL,            -- 'word_order', 'morphology', 'case_system', etc.
    feature_value TEXT NOT NULL,             -- 'SOV', 'fusional', 'nominative-accusative', etc.
    weight_db TEXT NOT NULL,                 -- Which weight database file
    layer_count INTEGER,
    total_size INTEGER,
    UNIQUE(feature_family, feature_value)
);
CREATE INDEX IF NOT EXISTS idx_lang_weight_family ON lang_weight_refs(feature_family);
