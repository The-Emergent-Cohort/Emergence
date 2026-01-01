-- Language Registry DB Schema
-- Maps 4-digit language codes to metadata
-- This DB is always loaded - needed for token encoding/decoding
--
-- Size target: ~1MB
-- Contents: ~10K language codes (languages + dialects)

PRAGMA foreign_keys = ON;

-- =============================================================================
-- LANGUAGE CODES: The 4-digit code assignments
-- =============================================================================
CREATE TABLE IF NOT EXISTS language_codes (
    code INTEGER PRIMARY KEY,               -- 0000-9999
    iso639_3 TEXT,                          -- 'eng', 'deu', 'jpn' (3-letter)
    iso639_1 TEXT,                          -- 'en', 'de', 'ja' (2-letter, if exists)
    bcp47 TEXT,                             -- 'en-US', 'zh-Hans' (full tag)
    glottocode TEXT,                        -- Glottolog code
    name TEXT NOT NULL,                     -- "English", "German", "Japanese"
    native_name TEXT,                       -- "English", "Deutsch", "日本語"

    -- Hierarchy
    parent_code INTEGER,                    -- For dialects: en-US -> en (0102 -> 0100)
    family_code INTEGER,                    -- Top-level family code
    level TEXT NOT NULL,                    -- 'family', 'language', 'dialect', 'variant'

    -- Metadata
    db_file TEXT,                           -- Which DB file contains this language
    speaker_count INTEGER,                  -- Estimated speakers
    status TEXT DEFAULT 'living',           -- 'living', 'extinct', 'ancient', 'constructed'

    FOREIGN KEY (parent_code) REFERENCES language_codes(code),
    FOREIGN KEY (family_code) REFERENCES language_codes(code)
);

CREATE INDEX IF NOT EXISTS idx_lc_iso3 ON language_codes(iso639_3);
CREATE INDEX IF NOT EXISTS idx_lc_iso1 ON language_codes(iso639_1);
CREATE INDEX IF NOT EXISTS idx_lc_bcp47 ON language_codes(bcp47);
CREATE INDEX IF NOT EXISTS idx_lc_glotto ON language_codes(glottocode);
CREATE INDEX IF NOT EXISTS idx_lc_parent ON language_codes(parent_code);
CREATE INDEX IF NOT EXISTS idx_lc_family ON language_codes(family_code);
CREATE INDEX IF NOT EXISTS idx_lc_level ON language_codes(level);
CREATE INDEX IF NOT EXISTS idx_lc_db ON language_codes(db_file);

-- =============================================================================
-- CODE RANGES: Reserved ranges for families and major languages
-- =============================================================================
CREATE TABLE IF NOT EXISTS code_ranges (
    id INTEGER PRIMARY KEY,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    family_code INTEGER,                    -- Which family this belongs to
    description TEXT NOT NULL,
    reserved_for TEXT,                      -- 'family', 'language', 'dialect'
    UNIQUE(range_start, range_end)
);

-- Seed the allocation plan
INSERT OR IGNORE INTO code_ranges (range_start, range_end, description, reserved_for) VALUES
    -- Universal
    (0, 0, 'Universal/cross-linguistic', 'special'),

    -- Language families (0001-0099)
    (1, 99, 'Language families', 'family'),

    -- Major languages with dialect ranges (0100-0999)
    (100, 149, 'English and variants', 'language'),
    (150, 199, 'Chinese and variants', 'language'),
    (200, 249, 'Spanish and variants', 'language'),
    (250, 299, 'Arabic and variants', 'language'),
    (300, 349, 'Hindi-Urdu and variants', 'language'),
    (350, 399, 'Japanese and variants', 'language'),
    (400, 449, 'German and variants', 'language'),
    (450, 499, 'French and variants', 'language'),
    (500, 549, 'Portuguese and variants', 'language'),
    (550, 599, 'Russian and variants', 'language'),
    (600, 649, 'Korean and variants', 'language'),
    (650, 699, 'Italian and variants', 'language'),
    (700, 749, 'Dutch and variants', 'language'),
    (750, 799, 'Polish and variants', 'language'),
    (800, 849, 'Turkish and variants', 'language'),
    (850, 899, 'Vietnamese and variants', 'language'),
    (900, 949, 'Thai and variants', 'language'),
    (950, 999, 'Reserved major languages', 'language'),

    -- Minor languages by family (1000-9999)
    (1000, 1999, 'Indo-European minor languages', 'minor'),
    (2000, 2999, 'Sino-Tibetan minor languages', 'minor'),
    (3000, 3999, 'Afro-Asiatic minor languages', 'minor'),
    (4000, 4999, 'Niger-Congo languages', 'minor'),
    (5000, 5999, 'Austronesian languages', 'minor'),
    (6000, 6999, 'Dravidian languages', 'minor'),
    (7000, 7499, 'Uralic languages', 'minor'),
    (7500, 7999, 'Turkic languages', 'minor'),
    (8000, 8499, 'Japonic and Koreanic', 'minor'),
    (8500, 8999, 'Austroasiatic languages', 'minor'),
    (9000, 9499, 'Other language families', 'minor'),
    (9500, 9899, 'Isolates and unclassified', 'minor'),
    (9900, 9999, 'Constructed languages', 'constructed');

-- =============================================================================
-- FAMILY METADATA: Additional info about language families
-- =============================================================================
CREATE TABLE IF NOT EXISTS family_metadata (
    family_code INTEGER PRIMARY KEY,
    proto_lang TEXT,                        -- "Proto-Indo-European"
    geographic_origin TEXT,                 -- "Pontic steppe"
    approx_age INTEGER,                     -- Years BP
    writing_systems TEXT,                   -- JSON array
    FOREIGN KEY (family_code) REFERENCES language_codes(code)
);

-- =============================================================================
-- SEED: Core language codes
-- =============================================================================

-- Universal
INSERT OR IGNORE INTO language_codes (code, name, level, status) VALUES
    (0, 'Universal', 'special', 'special');

-- Language families
INSERT OR IGNORE INTO language_codes (code, name, level, glottocode) VALUES
    (1, 'Indo-European', 'family', 'indo1319'),
    (2, 'Sino-Tibetan', 'family', 'sino1245'),
    (3, 'Afro-Asiatic', 'family', 'afro1255'),
    (4, 'Niger-Congo', 'family', 'atla1278'),
    (5, 'Austronesian', 'family', 'aust1307'),
    (6, 'Dravidian', 'family', 'drav1251'),
    (7, 'Uralic', 'family', 'ural1272'),
    (8, 'Turkic', 'family', 'turk1311'),
    (9, 'Japonic', 'family', 'japo1237'),
    (10, 'Koreanic', 'family', 'kore1284'),
    (11, 'Austroasiatic', 'family', 'aust1305'),
    (12, 'Tai-Kadai', 'family', 'taik1256'),
    (13, 'Nilo-Saharan', 'family', 'nilo1247');

-- English and variants
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (100, 'eng', 'en', 'English', NULL, 1, 'language', 'eng.db', 1500000000),
    (101, 'eng', 'en', 'British English', 100, 1, 'dialect', 'eng.db', 67000000),
    (102, 'eng', 'en', 'American English', 100, 1, 'dialect', 'eng.db', 330000000),
    (103, 'eng', 'en', 'Australian English', 100, 1, 'dialect', 'eng.db', 25000000),
    (104, 'eng', 'en', 'Indian English', 100, 1, 'dialect', 'eng.db', 125000000),
    (105, 'eng', 'en', 'South African English', 100, 1, 'dialect', 'eng.db', 5000000),
    (106, 'eng', 'en', 'Irish English', 100, 1, 'dialect', 'eng.db', 5000000),
    (107, 'eng', 'en', 'Scottish English', 100, 1, 'dialect', 'eng.db', 5000000),
    (108, 'eng', 'en', 'Canadian English', 100, 1, 'dialect', 'eng.db', 30000000),
    (109, 'eng', 'en', 'New Zealand English', 100, 1, 'dialect', 'eng.db', 5000000);

-- Chinese and variants
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (150, 'zho', 'zh', 'Chinese', NULL, 2, 'language', 'zho.db', 1300000000),
    (151, 'cmn', 'zh', 'Mandarin Chinese', 150, 2, 'dialect', 'zho.db', 920000000),
    (152, 'yue', NULL, 'Cantonese', 150, 2, 'dialect', 'zho.db', 85000000),
    (153, 'wuu', NULL, 'Wu Chinese', 150, 2, 'dialect', 'zho.db', 80000000),
    (154, 'nan', NULL, 'Min Nan', 150, 2, 'dialect', 'zho.db', 50000000),
    (155, 'hak', NULL, 'Hakka', 150, 2, 'dialect', 'zho.db', 45000000);

-- Spanish
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (200, 'spa', 'es', 'Spanish', NULL, 1, 'language', 'spa.db', 550000000),
    (201, 'spa', 'es', 'Castilian Spanish', 200, 1, 'dialect', 'spa.db', 47000000),
    (202, 'spa', 'es', 'Mexican Spanish', 200, 1, 'dialect', 'spa.db', 130000000),
    (203, 'spa', 'es', 'Argentine Spanish', 200, 1, 'dialect', 'spa.db', 45000000);

-- Arabic
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (250, 'ara', 'ar', 'Arabic', NULL, 3, 'language', 'ara.db', 420000000),
    (251, 'arb', 'ar', 'Modern Standard Arabic', 250, 3, 'variant', 'ara.db', 270000000),
    (252, 'arz', NULL, 'Egyptian Arabic', 250, 3, 'dialect', 'ara.db', 100000000),
    (253, 'apc', NULL, 'Levantine Arabic', 250, 3, 'dialect', 'ara.db', 40000000);

-- Japanese
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (350, 'jpn', 'ja', 'Japanese', NULL, 9, 'language', 'jpn.db', 125000000);

-- German
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (400, 'deu', 'de', 'German', NULL, 1, 'language', 'deu.db', 100000000),
    (401, 'deu', 'de', 'Standard German', 400, 1, 'variant', 'deu.db', 80000000),
    (402, 'deu', 'de', 'Austrian German', 400, 1, 'dialect', 'deu.db', 9000000),
    (403, 'deu', 'de', 'Swiss German', 400, 1, 'dialect', 'deu.db', 5000000);

-- French
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (450, 'fra', 'fr', 'French', NULL, 1, 'language', 'fra.db', 280000000),
    (451, 'fra', 'fr', 'Metropolitan French', 450, 1, 'dialect', 'fra.db', 67000000),
    (452, 'fra', 'fr', 'Canadian French', 450, 1, 'dialect', 'fra.db', 7000000),
    (453, 'fra', 'fr', 'Belgian French', 450, 1, 'dialect', 'fra.db', 4000000);

-- Portuguese
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (500, 'por', 'pt', 'Portuguese', NULL, 1, 'language', 'por.db', 260000000),
    (501, 'por', 'pt', 'European Portuguese', 500, 1, 'dialect', 'por.db', 10000000),
    (502, 'por', 'pt', 'Brazilian Portuguese', 500, 1, 'dialect', 'por.db', 210000000);

-- Russian
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (550, 'rus', 'ru', 'Russian', NULL, 1, 'language', 'rus.db', 255000000);

-- Korean
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (600, 'kor', 'ko', 'Korean', NULL, 10, 'language', 'kor.db', 80000000);

-- Italian
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (650, 'ita', 'it', 'Italian', NULL, 1, 'language', 'ita.db', 67000000);

-- Dutch
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (700, 'nld', 'nl', 'Dutch', NULL, 1, 'language', 'nld.db', 25000000);

-- Hindi
INSERT OR IGNORE INTO language_codes
    (code, iso639_3, iso639_1, name, parent_code, family_code, level, db_file, speaker_count) VALUES
    (300, 'hin', 'hi', 'Hindi', NULL, 1, 'language', 'hin.db', 600000000);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Full language info with family
CREATE VIEW IF NOT EXISTS language_full AS
SELECT
    lc.code,
    lc.iso639_3,
    lc.iso639_1,
    lc.name,
    lc.native_name,
    lc.level,
    p.name AS parent_name,
    p.code AS parent_code,
    f.name AS family_name,
    f.code AS family_code,
    lc.db_file,
    lc.speaker_count,
    lc.status
FROM language_codes lc
LEFT JOIN language_codes p ON lc.parent_code = p.code
LEFT JOIN language_codes f ON lc.family_code = f.code;

-- ISO code to our code lookup
CREATE VIEW IF NOT EXISTS iso_to_code AS
SELECT
    iso639_3,
    iso639_1,
    code,
    name,
    level
FROM language_codes
WHERE iso639_3 IS NOT NULL OR iso639_1 IS NOT NULL;

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
