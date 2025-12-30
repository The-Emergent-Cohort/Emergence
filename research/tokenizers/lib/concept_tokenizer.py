#!/usr/bin/env python3
"""
Concept Tokenizer: The DB IS the Tokenizer

This module provides the core tokenization interface:
- surface_to_tokens: Convert text to concept + modifier tokens
- tokens_to_surface: Convert tokens back to text
- Token caching and prefetching for efficiency

Every word is a query. Every thought is composed of database queries.

Token ID Ranges:
    0-999:         Reserved/System tokens
    1000-2999:     Modifier tokens (grammar: tense, number, case)
    3000000+:      Concept tokens (synset-based, 128 slots per synset)

Token Format:
    [CONCEPT:lemma]     -> concept token from language DB
    [MODIFIER:name]     -> grammar modifier token
    [PERSONAL:name]     -> personal token from identity DB
    [ENTITY:name]       -> entity token from identity DB
    [VAR:name]          -> runtime variable from runtime DB
    [SYSTEM:name]       -> system control token
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from functools import lru_cache

# Token ID constants
SYSTEM_RANGE = (0, 999)
MODIFIER_RANGE = (1000, 2999)
CONCEPT_BASE = 3_000_000
SYNSET_SLOTS = 128

# Default database path
DEFAULT_DB = Path(__file__).parent.parent / "db" / "language.db"


@dataclass
class Token:
    """Represents a single token with its metadata."""
    token_id: int
    category: str           # CONCEPT, MODIFIER, PERSONAL, ENTITY, VAR, SYSTEM
    name: str               # Human-readable name/lemma
    surface_form: str = ""  # Original surface form if applicable
    features: str = ""      # Morphological features if applicable

    def __repr__(self):
        if self.features:
            return f"[{self.category}:{self.name}+{self.features}]"
        return f"[{self.category}:{self.name}]"

    def to_tuple(self) -> Tuple[str, int]:
        return (self.category, self.token_id)


class ConceptTokenizer:
    """
    Database-backed concept tokenizer.

    The DB IS the tokenizer. Every tokenization is a query.
    """

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self._conn = None
        self._modifier_cache: Dict[str, int] = {}
        self._concept_cache: Dict[str, List[Tuple[int, str]]] = {}
        self._token_cache: Dict[int, Tuple[str, str]] = {}

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._load_modifier_cache()
        return self._conn

    def _load_modifier_cache(self):
        """Load all modifiers into memory (small set)."""
        cursor = self.conn.execute(
            "SELECT modifier_id, category, name, symbol FROM modifiers"
        )
        for row in cursor:
            key = f"{row['category']}:{row['name']}"
            self._modifier_cache[key] = row['modifier_id']
            if row['symbol']:
                self._modifier_cache[row['symbol']] = row['modifier_id']

    def token_id_to_synset(self, token_id: int) -> int:
        """Extract synset ID from a concept token ID."""
        if token_id < CONCEPT_BASE:
            return -1
        return (token_id - CONCEPT_BASE) // SYNSET_SLOTS

    def synset_to_token_range(self, synset_id: int) -> Tuple[int, int]:
        """Get token ID range for a synset."""
        base = CONCEPT_BASE + (synset_id * SYNSET_SLOTS)
        return (base, base + SYNSET_SLOTS - 1)

    def surface_to_tokens(self, text: str) -> List[Token]:
        """
        Convert surface text to tokens.

        Input: "running"
        Query: surface_forms WHERE form = "running"
        Result: concept_id = RUN, features = progressive
        Tokens: [CONCEPT:run] + [MODIFIER:progressive]
        """
        tokens = []

        # Simple word tokenization (can be enhanced)
        words = re.findall(r'\w+', text.lower())

        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        return tokens

    def _tokenize_word(self, word: str) -> List[Token]:
        """Tokenize a single word."""
        # Check concept cache first
        if word in self._concept_cache:
            cached = self._concept_cache[word]
            return [Token(
                token_id=c[0],
                category="CONCEPT",
                name=c[1],
                surface_form=word
            ) for c in cached[:1]]  # Return first match

        # Query database
        cursor = self.conn.execute("""
            SELECT
                sf.surface_form,
                sf.pos_features,
                c.lemma,
                c.concept_id,
                c.synset_id,
                c.concept_offset
            FROM surface_forms sf
            JOIN concepts c ON sf.concept_id = c.concept_id
            WHERE sf.surface_form = ?
            LIMIT 5
        """, (word,))

        results = cursor.fetchall()

        if not results:
            # Unknown word - return as-is with unknown token
            return [Token(
                token_id=0,
                category="UNKNOWN",
                name=word,
                surface_form=word
            )]

        tokens = []
        row = results[0]  # Use first match

        # Calculate token ID
        token_id = CONCEPT_BASE + (row['synset_id'] * SYNSET_SLOTS) + row['concept_offset']

        # Main concept token
        tokens.append(Token(
            token_id=token_id,
            category="CONCEPT",
            name=row['lemma'],
            surface_form=word
        ))

        # Add modifier tokens for morphological features
        if row['pos_features']:
            for feature in row['pos_features'].split(';'):
                if feature in self._modifier_cache:
                    mod_id = self._modifier_cache[feature]
                    tokens.append(Token(
                        token_id=mod_id,
                        category="MODIFIER",
                        name=feature,
                        features=feature
                    ))

        # Cache the result
        self._concept_cache[word] = [(token_id, row['lemma'])]

        return tokens

    def tokens_to_surface(
        self,
        tokens: List[Token],
        target_lang: str = "en"
    ) -> str:
        """
        Convert tokens back to surface text.

        Tokens: [CONCEPT:run] + [MODIFIER:progressive]
        Query: surface_forms WHERE concept_id = ? AND features LIKE '%PROG%'
        Result: "running"
        """
        words = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token.category == "UNKNOWN":
                words.append(token.name)
                i += 1
                continue

            if token.category != "CONCEPT":
                i += 1
                continue

            # Collect any following modifiers
            modifiers = []
            j = i + 1
            while j < len(tokens) and tokens[j].category == "MODIFIER":
                modifiers.append(tokens[j].name)
                j += 1

            # Find appropriate surface form
            surface = self._find_surface_form(token.token_id, modifiers)
            words.append(surface)

            i = j

        return " ".join(words)

    def _find_surface_form(self, token_id: int, modifiers: List[str]) -> str:
        """Find best surface form for concept + modifiers."""
        # Extract synset and offset from token ID
        if token_id < CONCEPT_BASE:
            return f"<{token_id}>"

        synset_id = (token_id - CONCEPT_BASE) // SYNSET_SLOTS
        offset = (token_id - CONCEPT_BASE) % SYNSET_SLOTS

        # First get the lemma
        cursor = self.conn.execute("""
            SELECT lemma FROM concepts
            WHERE synset_id = ? AND concept_offset = ?
        """, (synset_id, offset))

        row = cursor.fetchone()
        if not row:
            return f"<{token_id}>"

        lemma = row['lemma']

        # If no modifiers, return lemma
        if not modifiers:
            return lemma

        # Find surface form with matching features
        feature_pattern = "%".join(modifiers)
        cursor = self.conn.execute("""
            SELECT surface_form FROM surface_forms sf
            JOIN concepts c ON sf.concept_id = c.concept_id
            WHERE c.synset_id = ?
              AND c.concept_offset = ?
              AND sf.pos_features LIKE ?
            LIMIT 1
        """, (synset_id, offset, f"%{feature_pattern}%"))

        row = cursor.fetchone()
        if row:
            return row['surface_form']

        return lemma

    def prefetch_synset(self, synset_id: int):
        """
        Prefetch all concepts in a synset for cache warming.
        Related concepts = sequential reads = efficient on spinners.
        """
        cursor = self.conn.execute("""
            SELECT
                c.concept_id,
                c.synset_id,
                c.concept_offset,
                c.lemma,
                sf.surface_form
            FROM concepts c
            LEFT JOIN surface_forms sf ON c.concept_id = sf.concept_id
                                       AND sf.form_type = 'lemma'
            WHERE c.synset_id = ?
        """, (synset_id,))

        for row in cursor:
            token_id = CONCEPT_BASE + (row['synset_id'] * SYNSET_SLOTS) + row['concept_offset']
            lemma = row['lemma']
            surface = row['surface_form'] or lemma

            self._concept_cache[surface] = [(token_id, lemma)]
            self._token_cache[token_id] = (lemma, surface)

    def prefetch_related(self, synset_id: int, depth: int = 1):
        """
        Prefetch synset and its related synsets.
        Like physics engine propagation - load what's connected.
        """
        seen: Set[int] = set()
        to_fetch = [synset_id]

        for _ in range(depth):
            next_fetch = []
            for sid in to_fetch:
                if sid in seen:
                    continue
                seen.add(sid)

                self.prefetch_synset(sid)

                # Get related synsets
                cursor = self.conn.execute("""
                    SELECT related_synset_id FROM synset_relations
                    WHERE synset_id = ?
                """, (sid,))

                for row in cursor:
                    if row['related_synset_id'] not in seen:
                        next_fetch.append(row['related_synset_id'])

            to_fetch = next_fetch

    def get_modifier(self, category: str, name: str) -> Optional[int]:
        """Get modifier token ID by category and name."""
        key = f"{category}:{name}"
        return self._modifier_cache.get(key)

    def lookup_concept(self, lemma: str) -> Optional[Tuple[int, int, str]]:
        """
        Look up concept by lemma.
        Returns (token_id, synset_id, gloss) or None.
        """
        cursor = self.conn.execute("""
            SELECT
                c.concept_id,
                c.synset_id,
                c.concept_offset,
                s.gloss
            FROM concepts c
            JOIN synsets s ON c.synset_id = s.synset_id
            WHERE c.lemma = ?
            LIMIT 1
        """, (lemma,))

        row = cursor.fetchone()
        if not row:
            return None

        token_id = CONCEPT_BASE + (row['synset_id'] * SYNSET_SLOTS) + row['concept_offset']
        return (token_id, row['synset_id'], row['gloss'])

    def stats(self) -> dict:
        """Get database statistics."""
        stats = {}

        for table in ['synsets', 'concepts', 'surface_forms', 'modifiers', 'translations']:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        stats['concept_cache_size'] = len(self._concept_cache)
        stats['token_cache_size'] = len(self._token_cache)

        return stats

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Convenience functions
def tokenize(text: str, db_path: Path = DEFAULT_DB) -> List[Token]:
    """One-shot tokenization."""
    tokenizer = ConceptTokenizer(db_path)
    try:
        return tokenizer.surface_to_tokens(text)
    finally:
        tokenizer.close()


def detokenize(tokens: List[Token], db_path: Path = DEFAULT_DB) -> str:
    """One-shot detokenization."""
    tokenizer = ConceptTokenizer(db_path)
    try:
        return tokenizer.tokens_to_surface(tokens)
    finally:
        tokenizer.close()


if __name__ == "__main__":
    # Demo usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python concept_tokenizer.py <text>")
        print("\nExample: python concept_tokenizer.py 'The cat is running'")
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    print(f"Input: {text}")
    print()

    try:
        tokenizer = ConceptTokenizer()

        print("Database stats:")
        for key, value in tokenizer.stats().items():
            print(f"  {key}: {value:,}")
        print()

        tokens = tokenizer.surface_to_tokens(text)
        print("Tokens:")
        for t in tokens:
            print(f"  {t} (id={t.token_id})")
        print()

        reconstructed = tokenizer.tokens_to_surface(tokens)
        print(f"Reconstructed: {reconstructed}")

        tokenizer.close()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to run import_kaikki.py first to populate the database.")
        sys.exit(1)
