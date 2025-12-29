# Standard Query Templates for Unified Tokenizer DB

## The 5 Core Operations

### 1. TOKENIZE INPUT (text + language → concept IDs)

The primary tokenization lookup. This is THE critical path.

```sql
SELECT e.concept_id, c.canonical, c.concept_type, c.modality, e.confidence
FROM etymology e
JOIN concepts c ON e.concept_id = c.id
WHERE e.language = :language
  AND e.surface_form = :surface_form
  AND (e.pos_tag = :pos_tag OR e.pos_tag IS NULL)
ORDER BY
    CASE WHEN e.pos_tag = :pos_tag THEN 0 ELSE 1 END,
    e.frequency DESC,
    e.confidence DESC
LIMIT 1;
```

**Prepared statement version (for batch tokenization):**
```sql
PREPARE tokenize_stmt AS
  SELECT concept_id FROM etymology
  WHERE language = ? AND surface_form = ?
  ORDER BY frequency DESC LIMIT 1;
```

### 2. DETOKENIZE OUTPUT (concept ID → surface form for language)

Reverse lookup for text generation.

```sql
SELECT e.surface_form, e.pos_tag, e.register, e.frequency
FROM etymology e
WHERE e.concept_id = :concept_id
  AND e.language = :target_language
ORDER BY e.frequency DESC, e.confidence DESC
LIMIT 1;
```

**With fallback to canonical form:**
```sql
SELECT COALESCE(
    (SELECT surface_form FROM etymology
     WHERE concept_id = :concept_id AND language = :target_language
     ORDER BY frequency DESC LIMIT 1),
    (SELECT canonical FROM concepts WHERE id = :concept_id)
) AS surface_form;
```

### 3. GRAMMAR LOOKUP (language → applicable rules with inheritance)

Resolves full inheritance chain from language up to family.

```sql
WITH RECURSIVE family_chain AS (
    -- Start with the specific language
    SELECT id, parent_id, name, 1 AS depth
    FROM language_families
    WHERE id = :language_family_id

    UNION ALL

    -- Walk up to parent families
    SELECT lf.id, lf.parent_id, lf.name, fc.depth + 1
    FROM language_families lf
    JOIN family_chain fc ON lf.id = fc.parent_id
)
SELECT gr.*, fc.depth AS inheritance_depth
FROM grammar_rules gr
JOIN family_chain fc ON gr.family_id = fc.id
WHERE gr.override_level != 'disable'
ORDER BY fc.depth ASC, gr.rule_type, gr.rule_name;
```

**Get effective rule for specific type (respects overrides):**
```sql
WITH RECURSIVE family_chain AS (
    SELECT id, parent_id, 1 AS depth FROM language_families WHERE id = :family_id
    UNION ALL
    SELECT lf.id, lf.parent_id, fc.depth + 1
    FROM language_families lf JOIN family_chain fc ON lf.id = fc.parent_id
)
SELECT gr.*
FROM grammar_rules gr
JOIN family_chain fc ON gr.family_id = fc.id
WHERE gr.rule_type = :rule_type
ORDER BY fc.depth ASC
LIMIT 1;
```

### 4. ROUTE OUTPUT (concept ID → modality → destination)

Determines where output goes.

```sql
SELECT
    c.modality,
    c.target_language,
    c.target_format,
    c.route_destination,
    COALESCE(r.destination, c.route_destination) AS final_destination,
    r.route_type,
    r.pre_process,
    r.post_process
FROM concepts c
LEFT JOIN output_routes r ON r.concept_id = c.id OR r.modality_pattern = c.modality
WHERE c.id = :concept_id
ORDER BY r.priority DESC
LIMIT 1;
```

**Batch route lookup:**
```sql
SELECT c.id, c.modality, c.route_destination
FROM concepts c
WHERE c.id IN (:concept_ids)
ORDER BY c.id;
```

### 5. DEPRECATION LOOKUP (old V3-Tekken ID → new concept)

For backwards compatibility during migration.

```sql
SELECT
    dt.new_concept_id,
    dt.mapping_type,
    dt.split_concept_ids,
    c.canonical,
    c.concept_type,
    c.modality
FROM deprecated_tokens dt
LEFT JOIN concepts c ON dt.new_concept_id = c.id
WHERE dt.old_token_id = :old_token_id;
```

---

## Utility Queries

### Find all surface forms for a concept
```sql
SELECT language, surface_form, pos_tag, frequency, register
FROM etymology
WHERE concept_id = :concept_id
ORDER BY language, frequency DESC;
```

### Find languages that have a word for a concept
```sql
SELECT DISTINCT language, COUNT(*) as form_count
FROM etymology
WHERE concept_id = :concept_id
GROUP BY language
ORDER BY form_count DESC;
```

### Check migration status
```sql
SELECT
    mapping_type,
    COUNT(*) as count,
    ROUND(AVG(confidence), 2) as avg_confidence
FROM deprecated_tokens
GROUP BY mapping_type;
```

### Find unmapped V3-Tekken tokens
```sql
SELECT old_token_id, old_surface_form
FROM deprecated_tokens
WHERE new_concept_id IS NULL
  AND mapping_type != 'obsolete'
ORDER BY old_token_id;
```

### Logit bias calculation for grammar
```sql
-- Get bias values for a language's grammar rules
SELECT
    rule_type,
    rule_name,
    bias_when_violated,
    bias_when_satisfied,
    formal_weight,
    nlp_weight
FROM grammar_rules gr
JOIN language_families lf ON gr.family_id = lf.id
WHERE lf.name = :language
  AND gr.override_level != 'disable';
```

---

## Import Queries

### Stage etymology CSV
```sql
CREATE TABLE etymology_staging (
    lang TEXT,
    term TEXT,
    term_id TEXT,
    reltype TEXT,
    related_term TEXT,
    related_lang TEXT,
    position INTEGER
);

-- Import via: .mode csv / .import etymology.csv etymology_staging
```

### Create concepts from etymology clusters
```sql
INSERT INTO concepts (id, canonical, concept_type, modality, domain, confidence, source)
WITH root_clusters AS (
    SELECT DISTINCT
        CASE
            WHEN related_term LIKE '*%' THEN related_term
            WHEN related_lang LIKE 'Proto-%' THEN related_term
            ELSE term
        END AS root,
        COUNT(DISTINCT lang) AS lang_count
    FROM etymology_staging
    WHERE reltype IN ('derived_from', 'inherited_from', 'has_root')
    GROUP BY root
    HAVING lang_count > 1
)
SELECT
    3000000 + ROW_NUMBER() OVER (ORDER BY root) AS id,
    root AS canonical,
    'morpheme' AS concept_type,
    'text' AS modality,
    'lexical' AS domain,
    CASE WHEN root LIKE '*%' THEN 0.95 ELSE 0.80 END AS confidence,
    'wiktionary' AS source
FROM root_clusters;
```

### Register languages from etymology
```sql
INSERT INTO language_families (name, level, source)
SELECT DISTINCT language, 'language', 'wiktionary'
FROM etymology
WHERE language NOT IN (SELECT name FROM language_families);
```

---

## Performance Notes

Expected query times with proper indexes (2M+ entries):
- Tokenization lookup: < 1ms per token
- Detokenization: < 1ms per token
- Grammar rule resolution: < 10ms (cached after first call)
- Route lookup: < 1ms
- Deprecation lookup: < 1ms

For batch operations, use transactions of 10,000 rows.

Critical indexes:
- `idx_etymology_lookup_pos` - THE primary tokenization path
- `idx_concepts_canonical` - For reverse lookups
- `idx_deprecated_old` - For V3-Tekken compatibility
