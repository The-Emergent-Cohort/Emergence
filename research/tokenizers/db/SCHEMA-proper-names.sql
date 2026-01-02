-- Proper Names Extension Schema (Genomic Notation)
--
-- Proper nouns use the same A.D.C.FF.SS.LLL.DD.FP.COL structure
-- but with:
--   A = 99 (maximum abstraction - they're compound all the way down)
--   D = 10 (entity domain)
--   C = category (person=1, place=10, org=20, etc.)
--
-- Example: "Einstein" in English
--   99.10.2.1.8.127.0.{fingerprint}.0
--   ^  ^  ^ ^ ^ ^   ^ ^             ^
--   |  |  | | | |   | +-- fingerprint (derived from name properties)
--   |  |  | | | |   +-- dialect (standard)
--   |  |  | | | +-- language (127 = English)
--   |  |  | | +-- subfamily (8 = Germanic)
--   |  |  | +-- family (1 = Indo-European)
--   |  |  +-- category (2 = person_historical)
--   |  +-- domain (10 = entity)
--   +-- abstraction (99 = maximum compound)
--
-- This keeps proper names in the same coordinate space as all other tokens
-- but naturally isolated at the far end of the abstraction spectrum.

PRAGMA foreign_keys = ON;

-- =============================================================================
-- PROPER NAME CATEGORIES: Maps to entity domain categories
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_name_categories (
    category_id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,               -- 'person', 'place_city', etc.
    domain_category INTEGER NOT NULL,        -- Maps to CATEGORIES[(10, code)]
    name TEXT NOT NULL,                      -- Human-readable name
    description TEXT,
    parent_category_id INTEGER,              -- For subcategories
    FOREIGN KEY (parent_category_id) REFERENCES proper_name_categories(category_id)
);

-- Seed categories: person, place, thing (classic noun breakdown)
-- Top-level: 1=person, 2=place, 3=thing
-- Subcategories derived from parent
INSERT OR IGNORE INTO proper_name_categories (category_id, code, domain_category, name, description, parent_category_id) VALUES
    -- Person (1.x)
    (1, 'person', 1, 'Person', 'Named individuals', NULL),
    (11, 'person_historical', 1, 'Historical Figure', 'Historical persons', 1),
    (12, 'person_fictional', 1, 'Fictional Character', 'Characters from fiction', 1),
    (13, 'person_mythological', 1, 'Mythological Figure', 'Gods, heroes, mythological beings', 1),
    (14, 'person_contemporary', 1, 'Contemporary Person', 'Living or recent persons', 1),

    -- Place (2.x)
    (2, 'place', 2, 'Place', 'Geographic locations', NULL),
    (21, 'place_country', 2, 'Country', 'Nations and countries', 2),
    (22, 'place_city', 2, 'City', 'Cities and towns', 2),
    (23, 'place_region', 2, 'Region', 'States, provinces, regions', 2),
    (24, 'place_geographic', 2, 'Geographic Feature', 'Mountains, rivers, oceans', 2),
    (25, 'place_celestial', 2, 'Celestial Body', 'Planets, stars, astronomical objects', 2),

    -- Thing (3.x) - organizations, works, events, etc.
    (3, 'thing', 3, 'Thing', 'Named things', NULL),
    (31, 'thing_organization', 3, 'Organization', 'Companies, governments, institutions', 3),
    (32, 'thing_work', 3, 'Work/Product', 'Creative works, products, brands', 3),
    (33, 'thing_event', 3, 'Event', 'Historical events, named occurrences', 3);

-- =============================================================================
-- PROPER NAMES: The core entity table
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_names (
    name_id INTEGER PRIMARY KEY,

    -- Genomic token ID: 99.10.C.FF.SS.LLL.DD.FP.COL
    token_genomic TEXT UNIQUE,               -- Full genomic notation string

    canonical_name TEXT NOT NULL,            -- Primary name form
    category_id INTEGER NOT NULL,            -- Category/subcategory

    -- Language coordinates (for primary form)
    lang_family INTEGER DEFAULT 0,           -- FF
    lang_subfamily INTEGER DEFAULT 0,        -- SS
    lang_code INTEGER DEFAULT 0,             -- LLL
    lang_dialect INTEGER DEFAULT 0,          -- DD

    fingerprint INTEGER DEFAULT 0,           -- FP - derived from name properties
    collision INTEGER DEFAULT 0,             -- COL - disambiguator

    -- Semantic content (proper names can have meaning)
    etymology TEXT,                          -- Origin/meaning of name if known
    meaning_concept_id INTEGER,              -- If name has lexical meaning (e.g., "Dawn")

    -- Metadata
    wikidata_id TEXT,                        -- External reference
    source TEXT DEFAULT 'kaikki',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (category_id) REFERENCES proper_name_categories(category_id)
);
CREATE INDEX IF NOT EXISTS idx_propname_canonical ON proper_names(canonical_name);
CREATE INDEX IF NOT EXISTS idx_propname_category ON proper_names(category_id);
CREATE INDEX IF NOT EXISTS idx_propname_token ON proper_names(token_genomic);
CREATE INDEX IF NOT EXISTS idx_propname_wikidata ON proper_names(wikidata_id);
CREATE INDEX IF NOT EXISTS idx_propname_lang ON proper_names(lang_family, lang_subfamily, lang_code);

-- =============================================================================
-- PROPER NAME VARIANTS: All forms of a name across languages
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_name_variants (
    id INTEGER PRIMARY KEY,
    name_id INTEGER NOT NULL,                -- Parent proper name
    variant TEXT NOT NULL,                   -- The variant form

    -- Language coordinates for this variant
    lang_family INTEGER NOT NULL,
    lang_subfamily INTEGER NOT NULL,
    lang_code INTEGER NOT NULL,
    lang_dialect INTEGER DEFAULT 0,

    variant_type TEXT DEFAULT 'translation', -- 'canonical', 'translation', 'nickname', 'romanization'
    script TEXT,                             -- Writing system (latin, cyrillic, arabic, etc.)
    is_primary INTEGER DEFAULT 0,            -- 1 = primary form for this language
    frequency INTEGER DEFAULT 0,             -- Usage frequency if known

    FOREIGN KEY (name_id) REFERENCES proper_names(name_id)
);
CREATE INDEX IF NOT EXISTS idx_propvar_name ON proper_name_variants(name_id);
CREATE INDEX IF NOT EXISTS idx_propvar_variant ON proper_name_variants(variant);
CREATE INDEX IF NOT EXISTS idx_propvar_lang ON proper_name_variants(lang_family, lang_subfamily, lang_code);

-- =============================================================================
-- PROPER NAME RELATIONS: Relationships between named entities
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_name_relations (
    id INTEGER PRIMARY KEY,
    name_id INTEGER NOT NULL,
    related_name_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,             -- 'born_in', 'founded', 'member_of', 'located_in', etc.

    FOREIGN KEY (name_id) REFERENCES proper_names(name_id),
    FOREIGN KEY (related_name_id) REFERENCES proper_names(name_id)
);
CREATE INDEX IF NOT EXISTS idx_proprel_name ON proper_name_relations(name_id);
CREATE INDEX IF NOT EXISTS idx_proprel_related ON proper_name_relations(related_name_id);

-- =============================================================================
-- VIEWS: Convenient access patterns
-- =============================================================================

-- Full proper name lookup with genomic coordinates
CREATE VIEW IF NOT EXISTS proper_name_full AS
SELECT
    pn.name_id,
    pn.token_genomic,
    pn.canonical_name,
    pnc.code AS category_code,
    pnc.domain_category,
    pnc.name AS category_name,
    pn.lang_family || '.' || pn.lang_subfamily || '.' ||
        pn.lang_code || '.' || pn.lang_dialect AS lang_genomic,
    pn.fingerprint,
    pn.etymology,
    pn.wikidata_id
FROM proper_names pn
JOIN proper_name_categories pnc ON pn.category_id = pnc.category_id;

-- Search proper names by any variant
CREATE VIEW IF NOT EXISTS proper_name_search AS
SELECT
    pn.name_id,
    pn.token_genomic,
    pn.canonical_name,
    pnv.variant AS search_form,
    pnv.lang_family || '.' || pnv.lang_subfamily || '.' ||
        pnv.lang_code || '.' || pnv.lang_dialect AS variant_lang,
    pnv.variant_type,
    pnc.code AS category_code
FROM proper_names pn
JOIN proper_name_variants pnv ON pn.name_id = pnv.name_id
JOIN proper_name_categories pnc ON pn.category_id = pnc.category_id;

-- =============================================================================
-- HELPER FUNCTION: Generate genomic token from components
-- Called from Python: build_genomic_token(category_id, lang_coords, fingerprint, collision)
-- =============================================================================
-- Note: SQLite doesn't have stored functions, so this is done in Python:
--
-- def build_proper_name_genomic(category_id: int, lang_genomic: str,
--                                fingerprint: int, collision: int) -> str:
--     """Build genomic token for a proper name.
--
--     99.10.{category}.{lang_genomic}.{fingerprint}.{collision}
--     """
--     # Get domain_category from proper_name_categories
--     return f"99.10.{domain_category}.{lang_genomic}.{fingerprint}.{collision}"
