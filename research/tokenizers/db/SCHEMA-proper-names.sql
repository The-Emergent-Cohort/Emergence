-- Proper Names Extension Schema
-- Proper nouns are labels AND concepts - special category for weight handling
--
-- Token ID Range: 2000000 - 2999999 (between modifiers and concepts)
-- This keeps them separate from general semantic weights while maintaining
-- the ability to reference them as concepts.
--
-- Categories:
--   2000000-2099999: Persons (historical, fictional, living)
--   2100000-2199999: Places (countries, cities, geographic features)
--   2200000-2299999: Organizations (companies, governments, institutions)
--   2300000-2399999: Products/Works (brands, titles, artworks)
--   2400000-2499999: Events (wars, treaties, discoveries)
--   2500000-2599999: Other named entities
--   2600000-2999999: Reserved for expansion

PRAGMA foreign_keys = ON;

-- =============================================================================
-- PROPER NAME CATEGORIES: Type taxonomy for named entities
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_name_categories (
    category_id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,               -- 'PERSON', 'PLACE', 'ORG', etc.
    name TEXT NOT NULL,                      -- Human-readable name
    token_base INTEGER NOT NULL,             -- Starting token ID for this category
    token_limit INTEGER NOT NULL,            -- Ending token ID for this category
    description TEXT,
    parent_category_id INTEGER,              -- For subcategories
    FOREIGN KEY (parent_category_id) REFERENCES proper_name_categories(category_id)
);

-- Seed categories
INSERT OR IGNORE INTO proper_name_categories (category_id, code, name, token_base, token_limit, description) VALUES
    (1, 'PERSON', 'Person', 2000000, 2099999, 'Named individuals'),
    (2, 'PLACE', 'Place', 2100000, 2199999, 'Geographic locations'),
    (3, 'ORG', 'Organization', 2200000, 2299999, 'Companies, governments, institutions'),
    (4, 'WORK', 'Work/Product', 2300000, 2399999, 'Creative works, products, brands'),
    (5, 'EVENT', 'Event', 2400000, 2499999, 'Historical events, named occurrences'),
    (6, 'OTHER', 'Other Entity', 2500000, 2599999, 'Other named entities');

-- Subcategories for PERSON
INSERT OR IGNORE INTO proper_name_categories (category_id, code, name, token_base, token_limit, description, parent_category_id) VALUES
    (101, 'PERSON.HISTORICAL', 'Historical Figure', 2000000, 2019999, 'Historical persons', 1),
    (102, 'PERSON.FICTIONAL', 'Fictional Character', 2020000, 2039999, 'Characters from fiction', 1),
    (103, 'PERSON.MYTHOLOGICAL', 'Mythological Figure', 2040000, 2059999, 'Gods, heroes, mythological beings', 1),
    (104, 'PERSON.CONTEMPORARY', 'Contemporary Person', 2060000, 2079999, 'Living or recent persons', 1);

-- Subcategories for PLACE
INSERT OR IGNORE INTO proper_name_categories (category_id, code, name, token_base, token_limit, description, parent_category_id) VALUES
    (201, 'PLACE.COUNTRY', 'Country', 2100000, 2109999, 'Nations and countries', 2),
    (202, 'PLACE.CITY', 'City', 2110000, 2129999, 'Cities and towns', 2),
    (203, 'PLACE.REGION', 'Region', 2130000, 2139999, 'States, provinces, regions', 2),
    (204, 'PLACE.GEOGRAPHIC', 'Geographic Feature', 2140000, 2159999, 'Mountains, rivers, oceans', 2),
    (205, 'PLACE.CELESTIAL', 'Celestial Body', 2160000, 2169999, 'Planets, stars, astronomical objects', 2);

-- =============================================================================
-- PROPER NAMES: The core entity table
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_names (
    name_id INTEGER PRIMARY KEY,             -- Internal ID
    token_id INTEGER UNIQUE,                 -- Token in 2000000-2999999 range
    canonical_name TEXT NOT NULL,            -- Primary name form
    category_id INTEGER NOT NULL,            -- Category/subcategory
    lang TEXT DEFAULT 'mul',                 -- Primary language ('mul' = multilingual)

    -- Semantic content (proper names can have meaning)
    etymology TEXT,                          -- Origin/meaning of name if known
    meaning_synset_id INTEGER,               -- If name has lexical meaning (e.g., "Dawn")

    -- Metadata
    birth_date TEXT,                         -- For persons
    death_date TEXT,                         -- For persons
    founding_date TEXT,                      -- For organizations
    location_parent TEXT,                    -- For places (e.g., country for city)
    wikidata_id TEXT,                        -- External reference

    -- Import tracking
    source TEXT DEFAULT 'kaikki',
    source_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (category_id) REFERENCES proper_name_categories(category_id),
    FOREIGN KEY (meaning_synset_id) REFERENCES synsets(synset_id)
);
CREATE INDEX IF NOT EXISTS idx_propname_canonical ON proper_names(canonical_name);
CREATE INDEX IF NOT EXISTS idx_propname_category ON proper_names(category_id);
CREATE INDEX IF NOT EXISTS idx_propname_token ON proper_names(token_id);
CREATE INDEX IF NOT EXISTS idx_propname_wikidata ON proper_names(wikidata_id);

-- =============================================================================
-- PROPER NAME VARIANTS: All forms of a name across languages
-- =============================================================================
CREATE TABLE IF NOT EXISTS proper_name_variants (
    id INTEGER PRIMARY KEY,
    name_id INTEGER NOT NULL,                -- Parent proper name
    variant TEXT NOT NULL,                   -- The variant form
    lang TEXT NOT NULL,                      -- Language of this variant
    variant_type TEXT DEFAULT 'translation', -- 'canonical', 'translation', 'nickname', 'historical', 'romanization'
    script TEXT,                             -- Writing system (latin, cyrillic, arabic, etc.)
    is_primary INTEGER DEFAULT 0,            -- 1 = primary form for this language
    frequency INTEGER DEFAULT 0,             -- Usage frequency if known

    FOREIGN KEY (name_id) REFERENCES proper_names(name_id)
);
CREATE INDEX IF NOT EXISTS idx_propvar_name ON proper_name_variants(name_id);
CREATE INDEX IF NOT EXISTS idx_propvar_variant ON proper_name_variants(variant);
CREATE INDEX IF NOT EXISTS idx_propvar_lang ON proper_name_variants(lang);

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
CREATE INDEX IF NOT EXISTS idx_proprel_type ON proper_name_relations(relation_type);

-- =============================================================================
-- VIEWS: Convenient access patterns
-- =============================================================================

-- Full proper name lookup with category
CREATE VIEW IF NOT EXISTS proper_name_full AS
SELECT
    pn.name_id,
    pn.token_id,
    pn.canonical_name,
    pnc.code AS category_code,
    pnc.name AS category_name,
    pn.etymology,
    pn.wikidata_id,
    pn.source
FROM proper_names pn
JOIN proper_name_categories pnc ON pn.category_id = pnc.category_id;

-- Search proper names by any variant
CREATE VIEW IF NOT EXISTS proper_name_search AS
SELECT
    pn.name_id,
    pn.token_id,
    pn.canonical_name,
    pnv.variant AS search_form,
    pnv.lang,
    pnv.variant_type,
    pnc.code AS category_code
FROM proper_names pn
JOIN proper_name_variants pnv ON pn.name_id = pnv.name_id
JOIN proper_name_categories pnc ON pn.category_id = pnc.category_id;

-- Token allocation tracking (for assigning new tokens)
CREATE VIEW IF NOT EXISTS proper_name_token_allocation AS
SELECT
    pnc.code,
    pnc.name,
    pnc.token_base,
    pnc.token_limit,
    COUNT(pn.name_id) AS allocated,
    pnc.token_limit - pnc.token_base - COUNT(pn.name_id) AS remaining
FROM proper_name_categories pnc
LEFT JOIN proper_names pn ON pn.category_id = pnc.category_id
    OR pn.category_id IN (SELECT category_id FROM proper_name_categories WHERE parent_category_id = pnc.category_id)
WHERE pnc.parent_category_id IS NULL
GROUP BY pnc.category_id;

-- =============================================================================
-- TRIGGER: Auto-assign token IDs for new proper names
-- =============================================================================
CREATE TRIGGER IF NOT EXISTS proper_name_assign_token
AFTER INSERT ON proper_names
WHEN NEW.token_id IS NULL
BEGIN
    UPDATE proper_names
    SET token_id = (
        SELECT COALESCE(MAX(token_id), pnc.token_base - 1) + 1
        FROM proper_names pn2
        JOIN proper_name_categories pnc2 ON pn2.category_id = pnc2.category_id
        JOIN proper_name_categories pnc ON NEW.category_id = pnc.category_id
        WHERE pnc2.token_base = pnc.token_base
    )
    WHERE name_id = NEW.name_id;
END;
