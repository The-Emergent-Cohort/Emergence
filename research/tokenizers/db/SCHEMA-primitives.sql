-- Primitives DB Schema
-- Core semantic primitives that form the foundation layer
-- This DB is always loaded - it's the hub for all concept compositions
--
-- Size target: ~5MB
-- Contents: ~425 primitives × ~30 language forms = ~13K rows

PRAGMA foreign_keys = ON;

-- =============================================================================
-- PRIMITIVES: The atomic semantic units
-- =============================================================================
CREATE TABLE IF NOT EXISTS primitives (
    primitive_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,           -- "KNOW", "MOVE", "CONTAINER"
    source TEXT NOT NULL,                   -- 'nsm', 'image_schema', 'verbnet'
    domain INTEGER NOT NULL,                -- Matches token encoding domain (01-99)
    category INTEGER NOT NULL,              -- Matches token encoding category (01-99)
    description TEXT,
    examples TEXT,                          -- JSON array of example usages
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_name, source)
);

CREATE INDEX IF NOT EXISTS idx_primitive_source ON primitives(source);
CREATE INDEX IF NOT EXISTS idx_primitive_domain ON primitives(domain);
CREATE INDEX IF NOT EXISTS idx_primitive_domain_cat ON primitives(domain, category);

-- =============================================================================
-- PRIMITIVE FORMS: Cross-linguistic surface realizations
-- =============================================================================
CREATE TABLE IF NOT EXISTS primitive_forms (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    lang_code INTEGER NOT NULL,             -- 4-digit language code
    surface_form TEXT NOT NULL,             -- "know", "savoir", "wissen", "知る"
    confidence REAL DEFAULT 1.0,            -- How confident is this mapping
    source TEXT,                            -- 'nsm_research', 'omw', 'manual'
    notes TEXT,                             -- Any caveats about this form
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    UNIQUE(primitive_id, lang_code, surface_form)
);

CREATE INDEX IF NOT EXISTS idx_pform_primitive ON primitive_forms(primitive_id);
CREATE INDEX IF NOT EXISTS idx_pform_lang ON primitive_forms(lang_code);
CREATE INDEX IF NOT EXISTS idx_pform_surface ON primitive_forms(surface_form);

-- =============================================================================
-- PRIMITIVE RELATIONS: How primitives relate to each other
-- =============================================================================
CREATE TABLE IF NOT EXISTS primitive_relations (
    id INTEGER PRIMARY KEY,
    primitive_id INTEGER NOT NULL,
    related_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,            -- 'entails', 'contrasts', 'part_of', 'similar'
    weight REAL DEFAULT 1.0,                -- Strength of relation
    bidirectional INTEGER DEFAULT 0,        -- 1 = relation goes both ways
    FOREIGN KEY (primitive_id) REFERENCES primitives(primitive_id),
    FOREIGN KEY (related_id) REFERENCES primitives(primitive_id),
    UNIQUE(primitive_id, related_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_prel_primitive ON primitive_relations(primitive_id);
CREATE INDEX IF NOT EXISTS idx_prel_related ON primitive_relations(related_id);
CREATE INDEX IF NOT EXISTS idx_prel_type ON primitive_relations(relation_type);

-- =============================================================================
-- DOMAIN DEFINITIONS: What each domain code means
-- =============================================================================
CREATE TABLE IF NOT EXISTS domains (
    domain_id INTEGER PRIMARY KEY,          -- 01-99
    name TEXT NOT NULL,                     -- "physical", "mental", "social"
    description TEXT,
    parent_domain INTEGER,                  -- For sub-domains
    FOREIGN KEY (parent_domain) REFERENCES domains(domain_id)
);

-- Seed core domains (matching token_encoder.py)
INSERT OR IGNORE INTO domains (domain_id, name, description) VALUES
    (1, 'physical', 'Matter, space, motion, physical properties'),
    (2, 'temporal', 'Time, sequence, duration, change'),
    (3, 'mental', 'Cognition, perception, emotion, volition'),
    (4, 'social', 'Relations, communication, groups, roles'),
    (5, 'abstract', 'Logic, mathematics, categories, relations'),
    (6, 'biological', 'Life, organisms, health, body'),
    (7, 'artifact', 'Made things, tools, technology'),
    (8, 'natural', 'Nature, environment, elements'),
    (9, 'evaluative', 'Good/bad, values, judgments');

-- =============================================================================
-- CATEGORY DEFINITIONS: What each category code means within domains
-- =============================================================================
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,           -- 01-99 within domain
    name TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (domain_id) REFERENCES domains(domain_id),
    UNIQUE(domain_id, category_id)
);

-- Seed categories (matching token_encoder.py)
INSERT OR IGNORE INTO categories (domain_id, category_id, name) VALUES
    -- Physical domain (01)
    (1, 1, 'motion'),
    (1, 2, 'location'),
    (1, 3, 'contact'),
    (1, 4, 'change'),
    (1, 5, 'state'),
    (1, 6, 'size'),
    (1, 7, 'shape'),
    (1, 8, 'substance'),

    -- Temporal domain (02)
    (2, 1, 'sequence'),
    (2, 2, 'duration'),
    (2, 3, 'frequency'),
    (2, 4, 'tense'),

    -- Mental domain (03)
    (3, 1, 'perception'),
    (3, 2, 'cognition'),
    (3, 3, 'emotion'),
    (3, 4, 'volition'),
    (3, 5, 'memory'),
    (3, 6, 'attention'),
    (3, 7, 'understanding'),
    (3, 8, 'belief'),

    -- Social domain (04)
    (4, 1, 'communication'),
    (4, 2, 'relation'),
    (4, 3, 'group'),
    (4, 4, 'possession'),
    (4, 5, 'exchange'),
    (4, 6, 'conflict'),
    (4, 7, 'cooperation'),
    (4, 8, 'authority'),

    -- Abstract domain (05)
    (5, 1, 'quantity'),
    (5, 2, 'quality'),
    (5, 3, 'comparison'),
    (5, 4, 'logic'),
    (5, 5, 'category'),
    (5, 6, 'relation'),

    -- Biological domain (06)
    (6, 1, 'life'),
    (6, 2, 'body'),
    (6, 3, 'health'),
    (6, 4, 'growth'),

    -- Evaluative domain (09)
    (9, 1, 'value'),
    (9, 2, 'morality'),
    (9, 3, 'aesthetics');

-- =============================================================================
-- VIEWS: Helpful queries
-- =============================================================================

-- Full primitive with all forms
CREATE VIEW IF NOT EXISTS primitive_full AS
SELECT
    p.primitive_id,
    p.canonical_name,
    p.source,
    d.name AS domain_name,
    c.name AS category_name,
    p.domain,
    p.category,
    p.description,
    GROUP_CONCAT(pf.surface_form || ':' || pf.lang_code, ', ') AS forms
FROM primitives p
LEFT JOIN domains d ON p.domain = d.domain_id
LEFT JOIN categories c ON p.domain = c.domain_id AND p.category = c.category_id
LEFT JOIN primitive_forms pf ON p.primitive_id = pf.primitive_id
GROUP BY p.primitive_id;

-- Primitive lookup by surface form in any language
CREATE VIEW IF NOT EXISTS surface_to_primitive AS
SELECT
    pf.surface_form,
    pf.lang_code,
    p.primitive_id,
    p.canonical_name,
    p.domain,
    p.category
FROM primitive_forms pf
JOIN primitives p ON pf.primitive_id = p.primitive_id;

-- =============================================================================
-- METADATA
-- =============================================================================
CREATE TABLE IF NOT EXISTS import_metadata (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    import_date TEXT DEFAULT CURRENT_TIMESTAMP,
    record_count INTEGER,
    notes TEXT
);
