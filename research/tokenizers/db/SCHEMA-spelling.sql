-- Spelling Database Schema
-- Per-language spelling correction and input normalization
--
-- Handles:
--   1. Typos (edit distance corrections)
--   2. Homonyms (phonetically similar words)
--   3. Keyboard errors (adjacent key substitutions)
--   4. Common misspellings (learned patterns)

PRAGMA foreign_keys = ON;

-- =============================================================================
-- VALID_FORMS: Known correct spellings from Kaikki imports
-- =============================================================================
CREATE TABLE IF NOT EXISTS valid_forms (
    id INTEGER PRIMARY KEY,
    form TEXT NOT NULL UNIQUE,           -- The valid spelling
    form_lower TEXT NOT NULL,            -- Lowercase for lookup
    frequency INTEGER DEFAULT 0,         -- Usage frequency (higher = more common)
    pos TEXT,                            -- Part of speech if known
    concept_id INTEGER,                  -- Link to concept in lang DB
    phonetic_code TEXT,                  -- Soundex/Metaphone for homonym matching
    keyboard_signature TEXT              -- Key positions for adjacency matching
);

CREATE INDEX IF NOT EXISTS idx_valid_lower ON valid_forms(form_lower);
CREATE INDEX IF NOT EXISTS idx_valid_phonetic ON valid_forms(phonetic_code);
CREATE INDEX IF NOT EXISTS idx_valid_freq ON valid_forms(frequency DESC);

-- =============================================================================
-- MISSPELLINGS: Known misspelling -> correction mappings
-- =============================================================================
CREATE TABLE IF NOT EXISTS misspellings (
    id INTEGER PRIMARY KEY,
    misspelling TEXT NOT NULL,           -- The wrong form
    misspelling_lower TEXT NOT NULL,     -- Lowercase for lookup
    correction TEXT NOT NULL,            -- The correct form
    error_type TEXT DEFAULT 'unknown',   -- 'typo', 'homonym', 'keyboard', 'phonetic', 'learned'
    confidence REAL DEFAULT 0.8,         -- How confident in this correction
    frequency INTEGER DEFAULT 0,         -- How often this error is seen
    source TEXT DEFAULT 'imported',      -- 'imported', 'pyspellchecker', 'user', 'learned'

    UNIQUE(misspelling_lower, correction)
);

CREATE INDEX IF NOT EXISTS idx_misspell_lower ON misspellings(misspelling_lower);
CREATE INDEX IF NOT EXISTS idx_misspell_type ON misspellings(error_type);

-- =============================================================================
-- HOMONYM_GROUPS: Words that sound alike but differ in meaning
-- =============================================================================
CREATE TABLE IF NOT EXISTS homonym_groups (
    id INTEGER PRIMARY KEY,
    phonetic_code TEXT NOT NULL,         -- Shared sound signature
    group_name TEXT                      -- Optional label like "there-group"
);

CREATE TABLE IF NOT EXISTS homonym_members (
    id INTEGER PRIMARY KEY,
    group_id INTEGER NOT NULL,
    form TEXT NOT NULL,                  -- The word
    meaning_hint TEXT,                   -- Brief disambiguation hint
    usage_context TEXT,                  -- Typical context patterns
    concept_id INTEGER,                  -- Link to concept DB

    FOREIGN KEY (group_id) REFERENCES homonym_groups(id)
);

CREATE INDEX IF NOT EXISTS idx_homonym_group ON homonym_members(group_id);
CREATE INDEX IF NOT EXISTS idx_homonym_form ON homonym_members(form);

-- =============================================================================
-- KEYBOARD_ADJACENCY: For detecting fat-finger errors
-- =============================================================================
CREATE TABLE IF NOT EXISTS keyboard_layouts (
    id INTEGER PRIMARY KEY,
    layout_name TEXT NOT NULL UNIQUE,    -- 'qwerty', 'azerty', 'qwertz', etc.
    description TEXT
);

CREATE TABLE IF NOT EXISTS key_adjacency (
    id INTEGER PRIMARY KEY,
    layout_id INTEGER NOT NULL,
    key_char TEXT NOT NULL,              -- The key
    adjacent_chars TEXT NOT NULL,        -- Comma-separated adjacent keys
    row_pos INTEGER,                     -- Row on keyboard (0=number, 1=top, 2=home, 3=bottom)
    col_pos INTEGER,                     -- Column position

    FOREIGN KEY (layout_id) REFERENCES keyboard_layouts(id),
    UNIQUE(layout_id, key_char)
);

-- Seed QWERTY layout
INSERT OR IGNORE INTO keyboard_layouts (id, layout_name, description) VALUES
    (1, 'qwerty', 'Standard US QWERTY layout');

INSERT OR IGNORE INTO key_adjacency (layout_id, key_char, adjacent_chars, row_pos, col_pos) VALUES
    -- Number row
    (1, '1', '2,q', 0, 0), (1, '2', '1,3,q,w', 0, 1), (1, '3', '2,4,w,e', 0, 2),
    (1, '4', '3,5,e,r', 0, 3), (1, '5', '4,6,r,t', 0, 4), (1, '6', '5,7,t,y', 0, 5),
    (1, '7', '6,8,y,u', 0, 6), (1, '8', '7,9,u,i', 0, 7), (1, '9', '8,0,i,o', 0, 8),
    (1, '0', '9,o,p', 0, 9),
    -- Top row
    (1, 'q', '1,2,w,a', 1, 0), (1, 'w', 'q,e,2,3,a,s', 1, 1), (1, 'e', 'w,r,3,4,s,d', 1, 2),
    (1, 'r', 'e,t,4,5,d,f', 1, 3), (1, 't', 'r,y,5,6,f,g', 1, 4), (1, 'y', 't,u,6,7,g,h', 1, 5),
    (1, 'u', 'y,i,7,8,h,j', 1, 6), (1, 'i', 'u,o,8,9,j,k', 1, 7), (1, 'o', 'i,p,9,0,k,l', 1, 8),
    (1, 'p', 'o,0,l', 1, 9),
    -- Home row
    (1, 'a', 'q,w,s,z', 2, 0), (1, 's', 'a,d,w,e,z,x', 2, 1), (1, 'd', 's,f,e,r,x,c', 2, 2),
    (1, 'f', 'd,g,r,t,c,v', 2, 3), (1, 'g', 'f,h,t,y,v,b', 2, 4), (1, 'h', 'g,j,y,u,b,n', 2, 5),
    (1, 'j', 'h,k,u,i,n,m', 2, 6), (1, 'k', 'j,l,i,o,m', 2, 7), (1, 'l', 'k,o,p', 2, 8),
    -- Bottom row
    (1, 'z', 'a,s,x', 3, 0), (1, 'x', 'z,c,s,d', 3, 1), (1, 'c', 'x,v,d,f', 3, 2),
    (1, 'v', 'c,b,f,g', 3, 3), (1, 'b', 'v,n,g,h', 3, 4), (1, 'n', 'b,m,h,j', 3, 5),
    (1, 'm', 'n,j,k', 3, 6);

-- =============================================================================
-- ORTHOGRAPHIC_RULES: Language-specific spelling patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS orthographic_rules (
    id INTEGER PRIMARY KEY,
    rule_name TEXT NOT NULL,             -- e.g., 'i_before_e'
    pattern TEXT NOT NULL,               -- Regex or pattern to match
    exception_pattern TEXT,              -- Exceptions to the rule
    description TEXT,
    confidence REAL DEFAULT 0.9
);

-- =============================================================================
-- CORRECTION_LOG: Track corrections for learning
-- =============================================================================
CREATE TABLE IF NOT EXISTS correction_log (
    id INTEGER PRIMARY KEY,
    original TEXT NOT NULL,
    corrected TEXT NOT NULL,
    accepted INTEGER DEFAULT 1,          -- 1 = user accepted, 0 = rejected
    context TEXT,                        -- Surrounding words for context learning
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_correction_original ON correction_log(original);

-- =============================================================================
-- VIEWS: Convenient lookups
-- =============================================================================

-- Quick correction lookup
CREATE VIEW IF NOT EXISTS quick_correct AS
SELECT
    m.misspelling_lower AS input,
    m.correction,
    m.error_type,
    m.confidence,
    v.frequency AS correction_frequency
FROM misspellings m
LEFT JOIN valid_forms v ON m.correction = v.form_lower
ORDER BY m.confidence DESC, v.frequency DESC;

-- Homonym disambiguation
CREATE VIEW IF NOT EXISTS homonym_lookup AS
SELECT
    hm.form,
    hg.phonetic_code,
    hm.meaning_hint,
    hm.usage_context,
    GROUP_CONCAT(hm2.form, ', ') AS alternatives
FROM homonym_members hm
JOIN homonym_groups hg ON hm.group_id = hg.id
JOIN homonym_members hm2 ON hm2.group_id = hg.id AND hm2.form != hm.form
GROUP BY hm.form;
