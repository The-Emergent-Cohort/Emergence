-- Morphemes table - the hinge point
-- token IDs assigned later via concept_id
-- morpheme is unique key

CREATE TABLE IF NOT EXISTS morphemes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    morpheme TEXT NOT NULL UNIQUE,
    meaning TEXT,
    morpheme_type TEXT,  -- suffix, prefix, root
    origin TEXT,         -- Greek, Latin, or NULL
    pos_tendency TEXT    -- noun, verb, adjective, adverb
);

CREATE INDEX IF NOT EXISTS idx_morphemes_type ON morphemes(morpheme_type);
CREATE INDEX IF NOT EXISTS idx_morphemes_pos ON morphemes(pos_tendency);
CREATE INDEX IF NOT EXISTS idx_morphemes_origin ON morphemes(origin);
