#!/usr/bin/env python3
"""
Token ID Encoder: Derived semantic fingerprints with genomic notation

Token IDs are not assigned, they're derived. Like Dewey Decimal,
the token value encodes the semantic coordinates of a concept.

Genomic Notation (dotted strings as primary format):
    A.D.C.FF.SS.LLL.DD.FP.COL

    A   = Abstraction level (1-99) - distance from primitives
    D   = Domain (1-99) - semantic domain
    C   = Category (1-99) - category within domain
    FF  = Family (0-99) - language family
    SS  = Subfamily (0-99) - subfamily within family
    LLL = Language (0-999) - specific language
    DD  = Dialect (0-999) - dialect/variant
    FP  = Fingerprint (0-999999) - weighted primitive composition
    COL = Collision (0-999) - disambiguator

Example:
    "comprehend" = 2.3.7.1.8.127.0.248.0
    - Abstraction 2: One layer from primitives
    - Domain 3: Mental/cognitive
    - Category 7: Understanding
    - Family 1: Indo-European
    - Subfamily 8: Germanic
    - Language 127: English
    - Dialect 0: Standard
    - Fingerprint 248: composition of [KNOW, INSIDE, COMPLETE]
    - Collision 0: First concept at this coordinate

The dotted notation:
- Provides clean separators for variable-width components
- Reduces storage bits (no padding zeros)
- Encodes relationships in structure (genomic)
- Is the primary format (integers derived when needed)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math


# Domain codes (1-99)
DOMAINS = {
    "physical": 1,      # Matter, space, motion
    "temporal": 2,      # Time, sequence, duration
    "mental": 3,        # Cognition, perception, emotion
    "social": 4,        # Relations, communication, groups
    "abstract": 5,      # Logic, mathematics, categories
    "biological": 6,    # Life, organisms, health
    "artifact": 7,      # Made things, tools, technology
    "natural": 8,       # Nature, environment, elements
    "evaluative": 9,    # Good/bad, values, judgments
    "entity": 10,       # Named entities (proper names) - labels that reference
}

# Category codes within domains (1-99)
CATEGORIES = {
    # Mental domain (3)
    (3, "perception"): 1,
    (3, "cognition"): 2,
    (3, "emotion"): 3,
    (3, "volition"): 4,
    (3, "memory"): 5,
    (3, "attention"): 6,
    (3, "understanding"): 7,
    (3, "belief"): 8,

    # Physical domain (1)
    (1, "motion"): 1,
    (1, "location"): 2,
    (1, "contact"): 3,
    (1, "change"): 4,
    (1, "state"): 5,
    (1, "size"): 6,
    (1, "shape"): 7,
    (1, "substance"): 8,

    # Social domain (4)
    (4, "communication"): 1,
    (4, "relation"): 2,
    (4, "group"): 3,
    (4, "possession"): 4,
    (4, "exchange"): 5,
    (4, "conflict"): 6,
    (4, "cooperation"): 7,
    (4, "authority"): 8,

    # Temporal domain (2)
    (2, "sequence"): 1,
    (2, "duration"): 2,
    (2, "frequency"): 3,
    (2, "tense"): 4,

    # Abstract domain (5)
    (5, "quantity"): 1,
    (5, "degree"): 2,
    (5, "similarity"): 3,
    (5, "logic"): 4,
    (5, "category"): 5,
    (5, "part_whole"): 6,

    # Entity domain (10) - proper names: person, place, thing
    (10, "person"): 1,           # Named individuals
    (10, "place"): 2,            # Geographic locations
    (10, "thing"): 3,            # Named things (orgs, works, events)
}

# Language family codes (0-99)
LANG_FAMILIES = {
    "universal": 0,         # Cross-linguistic primitives
    "indo-european": 1,
    "sino-tibetan": 2,
    "afro-asiatic": 3,
    "niger-congo": 4,
    "austronesian": 5,
    "dravidian": 6,
    "uralic": 7,
    "turkic": 8,
    "japonic": 9,
    "koreanic": 10,
    "austroasiatic": 11,
    "tai-kadai": 12,
    "nilo-saharan": 13,
}

# Subfamily codes within families (0-99)
LANG_SUBFAMILIES = {
    # Indo-European (1)
    (1, "germanic"): 8,
    (1, "romance"): 12,
    (1, "slavic"): 15,
    (1, "indo-iranian"): 20,
    (1, "celtic"): 25,
    (1, "baltic"): 28,
    (1, "hellenic"): 30,
    (1, "albanian"): 32,
    (1, "armenian"): 34,

    # Sino-Tibetan (2)
    (2, "sinitic"): 1,
    (2, "tibeto-burman"): 5,

    # Afro-Asiatic (3)
    (3, "semitic"): 1,
    (3, "berber"): 5,
    (3, "cushitic"): 10,
    (3, "chadic"): 15,

    # Niger-Congo (4)
    (4, "atlantic-congo"): 1,
    (4, "mande"): 10,
    (4, "kordofanian"): 20,
    (4, "bantu"): 8,  # Under atlantic-congo, but major enough for own code
}

# Language codes within subfamilies (0-999)
# Maps (family, subfamily, iso639-3) -> language number
LANG_CODES = {
    # Germanic languages
    (1, 8, "eng"): 127,     # English
    (1, 8, "deu"): 200,     # German
    (1, 8, "nld"): 210,     # Dutch
    (1, 8, "swe"): 220,     # Swedish
    (1, 8, "dan"): 225,     # Danish
    (1, 8, "nor"): 230,     # Norwegian

    # Romance languages
    (1, 12, "fra"): 100,    # French
    (1, 12, "spa"): 150,    # Spanish
    (1, 12, "por"): 160,    # Portuguese
    (1, 12, "ita"): 170,    # Italian
    (1, 12, "ron"): 180,    # Romanian

    # Slavic languages
    (1, 15, "rus"): 100,    # Russian
    (1, 15, "pol"): 150,    # Polish
    (1, 15, "ces"): 160,    # Czech
    (1, 15, "ukr"): 170,    # Ukrainian

    # Indo-Iranian
    (1, 20, "hin"): 100,    # Hindi
    (1, 20, "urd"): 105,    # Urdu
    (1, 20, "ben"): 150,    # Bengali
    (1, 20, "fas"): 200,    # Persian

    # Sinitic
    (2, 1, "cmn"): 100,     # Mandarin
    (2, 1, "yue"): 150,     # Cantonese
    (2, 1, "wuu"): 160,     # Wu
    (2, 1, "nan"): 170,     # Min Nan

    # Japonic
    (9, 0, "jpn"): 100,     # Japanese

    # Koreanic
    (10, 0, "kor"): 100,    # Korean

    # Semitic
    (3, 1, "ara"): 100,     # Arabic
    (3, 1, "heb"): 150,     # Hebrew

    # Turkic
    (8, 0, "tur"): 100,     # Turkish
}

# Dialect codes (0-999)
DIALECT_CODES = {
    # English dialects
    (1, 8, 127, "standard"): 0,
    (1, 8, 127, "gb"): 1,       # British
    (1, 8, 127, "us"): 2,       # American
    (1, 8, 127, "au"): 3,       # Australian
    (1, 8, 127, "in"): 4,       # Indian
    (1, 8, 127, "za"): 5,       # South African

    # Spanish dialects
    (1, 12, 150, "standard"): 0,
    (1, 12, 150, "es"): 1,      # Castilian
    (1, 12, 150, "mx"): 2,      # Mexican
    (1, 12, 150, "ar"): 3,      # Argentine

    # Chinese dialects (different from Mandarin/Cantonese distinction)
    (2, 1, 100, "standard"): 0,
    (2, 1, 100, "tw"): 1,       # Taiwan Mandarin
    (2, 1, 100, "sg"): 2,       # Singapore Mandarin
}


@dataclass
class LanguageCoord:
    """Language coordinates in genomic notation."""
    family: int = 0         # Family code (0-99)
    subfamily: int = 0      # Subfamily code (0-99)
    language: int = 0       # Language code (0-999)
    dialect: int = 0        # Dialect code (0-999)

    def to_string(self) -> str:
        """Convert to dotted notation: FF.SS.LLL.DD"""
        return f"{self.family}.{self.subfamily}.{self.language}.{self.dialect}"

    @classmethod
    def from_string(cls, s: str) -> "LanguageCoord":
        """Parse from dotted notation."""
        parts = s.split(".")
        return cls(
            family=int(parts[0]) if len(parts) > 0 else 0,
            subfamily=int(parts[1]) if len(parts) > 1 else 0,
            language=int(parts[2]) if len(parts) > 2 else 0,
            dialect=int(parts[3]) if len(parts) > 3 else 0
        )

    @classmethod
    def from_iso(cls, iso639_3: str, dialect: str = "standard") -> "LanguageCoord":
        """Look up language coordinates from ISO code."""
        # Find in LANG_CODES
        for (fam, sub, iso), lang_num in LANG_CODES.items():
            if iso == iso639_3:
                # Found - now check dialect
                dial_num = DIALECT_CODES.get((fam, sub, lang_num, dialect), 0)
                return cls(family=fam, subfamily=sub, language=lang_num, dialect=dial_num)

        # Not found - return unknown
        return cls(family=99, subfamily=99, language=999, dialect=0)

    def to_flat(self) -> int:
        """Convert to flat integer for legacy compatibility."""
        # FF.SS.LLL.DD -> FFSSLLLDD (9 digits max)
        return (
            self.family * 10_000_000 +
            self.subfamily * 100_000 +
            self.language * 100 +
            self.dialect
        )

    @classmethod
    def from_flat(cls, flat: int) -> "LanguageCoord":
        """Parse from flat integer."""
        dialect = flat % 100
        flat //= 100
        language = flat % 1000
        flat //= 1000
        subfamily = flat % 100
        flat //= 100
        family = flat
        return cls(family=family, subfamily=subfamily, language=language, dialect=dialect)


@dataclass
class PrimitiveComponent:
    """A primitive that contributes to a concept's composition."""
    primitive_id: int       # ID of the primitive concept
    position: int           # Order in composition (1-indexed)
    relation: str = "part"  # How it contributes: part, modifier, frame


@dataclass
class TokenCoordinates:
    """
    Semantic coordinates for a token in genomic notation.

    Primary format is the dotted string. Integer form is derived.
    """
    abstraction: int = 1            # Layers from primitives (1-99)
    domain: int = 1                 # Domain code (1-99)
    category: int = 1               # Category within domain (1-99)
    lang: LanguageCoord = field(default_factory=LanguageCoord)
    fingerprint: int = 0            # Primitive composition fingerprint (0-999999)
    collision: int = 0              # Collision counter (0-999)

    def to_string(self) -> str:
        """
        Primary format: genomic notation.

        A.D.C.FF.SS.LLL.DD.FP.COL
        """
        return (
            f"{self.abstraction}."
            f"{self.domain}."
            f"{self.category}."
            f"{self.lang.family}."
            f"{self.lang.subfamily}."
            f"{self.lang.language}."
            f"{self.lang.dialect}."
            f"{self.fingerprint}."
            f"{self.collision}"
        )

    @classmethod
    def from_string(cls, s: str) -> "TokenCoordinates":
        """Parse from genomic notation."""
        parts = s.split(".")
        return cls(
            abstraction=int(parts[0]) if len(parts) > 0 else 1,
            domain=int(parts[1]) if len(parts) > 1 else 1,
            category=int(parts[2]) if len(parts) > 2 else 1,
            lang=LanguageCoord(
                family=int(parts[3]) if len(parts) > 3 else 0,
                subfamily=int(parts[4]) if len(parts) > 4 else 0,
                language=int(parts[5]) if len(parts) > 5 else 0,
                dialect=int(parts[6]) if len(parts) > 6 else 0
            ),
            fingerprint=int(parts[7]) if len(parts) > 7 else 0,
            collision=int(parts[8]) if len(parts) > 8 else 0
        )

    def to_token_id(self) -> int:
        """
        Derive integer token ID (for legacy/compatibility).

        Format: AA DD CC FFSSLLLDD FFFFFF CCC
        Not recommended as primary - use string form.
        """
        lang_flat = self.lang.to_flat()
        # Total: 2 + 2 + 2 + 9 + 6 + 3 = 24 digits max
        # Still fits in 64-bit with room to spare
        return (
            self.abstraction * 10**22 +
            self.domain * 10**20 +
            self.category * 10**18 +
            lang_flat * 10**9 +
            self.fingerprint * 10**3 +
            self.collision
        )

    @classmethod
    def from_token_id(cls, token_id: int) -> "TokenCoordinates":
        """Decode from integer token ID."""
        collision = token_id % 1000
        token_id //= 1000
        fingerprint = token_id % 1_000_000
        token_id //= 1_000_000
        lang_flat = token_id % 1_000_000_000
        token_id //= 1_000_000_000
        category = token_id % 100
        token_id //= 100
        domain = token_id % 100
        token_id //= 100
        abstraction = token_id

        return cls(
            abstraction=abstraction,
            domain=domain,
            category=category,
            lang=LanguageCoord.from_flat(lang_flat),
            fingerprint=fingerprint,
            collision=collision
        )

    def __str__(self) -> str:
        """String representation is genomic notation."""
        return self.to_string()

    def __repr__(self) -> str:
        return f"TokenCoordinates({self.to_string()})"


def compute_fingerprint(primitives: List[PrimitiveComponent]) -> int:
    """
    Compute the fingerprint from primitive composition.

    Fingerprint = Σ(primitive_id × position_weight) mod 1000000

    Position weights ensure order matters:
    - Position 1: weight 1
    - Position 2: weight 2
    - etc.

    This creates fingerprints where:
    - Different primitives = usually different fingerprint
    - Same primitives, different order = different fingerprint
    - Collisions rare but handled by collision counter

    6 digits allows for highly abstract compositions like
    "antidisestablishmentarianism" with many morpheme primitives.
    """
    if not primitives:
        return 0

    total = 0
    for p in primitives:
        total += p.primitive_id * p.position

    # Keep within 6 digits (0-999999)
    return total % 1_000_000


def compute_abstraction_level(primitives: List[PrimitiveComponent]) -> int:
    """
    Compute abstraction level from primitive composition.

    Level 1: Is itself a primitive
    Level 2: Direct combination of primitives
    Level 3+: Combinations of combinations

    Simple heuristic based on count.
    Could be refined with actual composition depth.
    """
    if not primitives:
        return 1  # Assume primitive if no composition

    count = len(primitives)
    if count == 1:
        return 1
    elif count <= 3:
        return 2
    elif count <= 6:
        return 3
    elif count <= 10:
        return 4
    else:
        return min(5 + (count - 10) // 5, 99)


class TokenEncoder:
    """
    Encodes concepts into semantic token coordinates.

    Uses genomic notation (dotted strings) as primary format.
    Integer form available for compatibility.
    """

    def __init__(self):
        self.domain_map = DOMAINS
        self.category_map = CATEGORIES
        self.family_map = LANG_FAMILIES
        self.subfamily_map = LANG_SUBFAMILIES

        # Cache for primitive lookups (would come from DB)
        self._primitive_cache: Dict[str, List[PrimitiveComponent]] = {}

        # Collision tracking: genomic_prefix -> next collision number
        # (prefix = everything except collision counter)
        self._collision_counter: Dict[str, int] = {}

    def encode(
        self,
        lemma: str,
        domain: str,
        category: str,
        lang_iso: str = "eng",
        dialect: str = "standard",
        primitives: List[PrimitiveComponent] = None
    ) -> TokenCoordinates:
        """
        Encode a concept to its semantic coordinates.

        Args:
            lemma: The word/concept
            domain: Semantic domain name
            category: Category within domain
            lang_iso: ISO 639-3 language code
            dialect: Dialect identifier
            primitives: Primitive composition (if known)

        Returns:
            TokenCoordinates with derived values
        """
        # Look up codes
        domain_code = self.domain_map.get(domain, 99)
        cat_key = (domain_code, category)
        category_code = self.category_map.get(cat_key, 99)

        # Get language coordinates
        lang_coord = LanguageCoord.from_iso(lang_iso, dialect)

        # Get or compute primitives
        if primitives is None:
            primitives = self._primitive_cache.get(lemma, [])

        # Derive abstraction and fingerprint
        abstraction = compute_abstraction_level(primitives)
        fingerprint = compute_fingerprint(primitives)

        # Build coordinate prefix for collision tracking
        prefix = (
            f"{abstraction}.{domain_code}.{category_code}."
            f"{lang_coord.family}.{lang_coord.subfamily}."
            f"{lang_coord.language}.{lang_coord.dialect}."
            f"{fingerprint}"
        )

        # Get collision number for this coordinate slot
        collision = self._collision_counter.get(prefix, 0)
        self._collision_counter[prefix] = collision + 1

        return TokenCoordinates(
            abstraction=abstraction,
            domain=domain_code,
            category=category_code,
            lang=lang_coord,
            fingerprint=fingerprint,
            collision=collision
        )

    def similarity(self, coord1: TokenCoordinates, coord2: TokenCoordinates) -> float:
        """
        Compute semantic similarity from coordinates.

        Tokens in same domain/category are more similar.
        Tokens with similar fingerprints share primitive composition.
        """
        score = 0.0

        # Same domain: +0.3
        if coord1.domain == coord2.domain:
            score += 0.3
            # Same category within domain: +0.2
            if coord1.category == coord2.category:
                score += 0.2

        # Similar abstraction level: +0.1
        if abs(coord1.abstraction - coord2.abstraction) <= 1:
            score += 0.1

        # Fingerprint similarity (shared primitives)
        fp_diff = abs(coord1.fingerprint - coord2.fingerprint)
        if fp_diff < 100:
            score += 0.3
        elif fp_diff < 1000:
            score += 0.2
        elif fp_diff < 10000:
            score += 0.1

        # Same language family: small bonus
        if coord1.lang.family == coord2.lang.family:
            score += 0.05
            if coord1.lang.subfamily == coord2.lang.subfamily:
                score += 0.03

        return min(score, 1.0)

    def similarity_from_strings(self, s1: str, s2: str) -> float:
        """Compute similarity from genomic notation strings."""
        return self.similarity(
            TokenCoordinates.from_string(s1),
            TokenCoordinates.from_string(s2)
        )

    def register_primitives(self, lemma: str, primitives: List[PrimitiveComponent]):
        """Register the primitive decomposition for a concept."""
        self._primitive_cache[lemma] = primitives


# Convenience functions

def encode_token(
    lemma: str,
    domain: str,
    category: str,
    lang_iso: str = "eng",
    dialect: str = "standard",
    primitives: List[Tuple[int, int]] = None,
    collision: int = None
) -> str:
    """
    Quick encoding of a concept to genomic notation.

    Args:
        lemma: Word to encode
        domain: Semantic domain
        category: Category within domain
        lang_iso: ISO 639-3 language code
        dialect: Dialect identifier
        primitives: List of (primitive_id, position) tuples
        collision: Collision counter if known (usually from DB)

    Returns:
        Genomic notation string (e.g., "2.3.7.1.8.127.0.248.0")
    """
    encoder = TokenEncoder()

    if primitives:
        prims = [PrimitiveComponent(p[0], p[1]) for p in primitives]
    else:
        prims = None

    coords = encoder.encode(lemma, domain, category, lang_iso, dialect, prims)

    # Override collision if explicitly provided
    if collision is not None:
        coords.collision = collision

    return coords.to_string()


def decode_token(token_str: str) -> TokenCoordinates:
    """Parse genomic notation to coordinates."""
    return TokenCoordinates.from_string(token_str)


def extract_fingerprint(token_str: str) -> int:
    """Extract just the fingerprint from genomic notation."""
    coords = TokenCoordinates.from_string(token_str)
    return coords.fingerprint


def extract_language(token_str: str) -> LanguageCoord:
    """Extract language coordinates from genomic notation."""
    coords = TokenCoordinates.from_string(token_str)
    return coords.lang


if __name__ == "__main__":
    # Demo
    print("Token Encoder Demo - Genomic Notation")
    print("=" * 60)

    # Example: "comprehend" = KNOW + INSIDE + COMPLETE
    primitives = [
        PrimitiveComponent(primitive_id=12, position=1),  # KNOW
        PrimitiveComponent(primitive_id=34, position=2),  # INSIDE
        PrimitiveComponent(primitive_id=56, position=3),  # COMPLETE
    ]

    encoder = TokenEncoder()
    coords = encoder.encode(
        lemma="comprehend",
        domain="mental",
        category="understanding",
        lang_iso="eng",
        dialect="standard",
        primitives=primitives
    )

    print(f"\n'comprehend':")
    print(f"  Genomic notation: {coords}")
    print(f"  ├── Abstraction:  {coords.abstraction} (layers from primitives)")
    print(f"  ├── Domain:       {coords.domain} (mental)")
    print(f"  ├── Category:     {coords.category} (understanding)")
    print(f"  ├── Language:     {coords.lang.to_string()}")
    print(f"  │   ├── Family:     {coords.lang.family} (Indo-European)")
    print(f"  │   ├── Subfamily:  {coords.lang.subfamily} (Germanic)")
    print(f"  │   ├── Language:   {coords.lang.language} (English)")
    print(f"  │   └── Dialect:    {coords.lang.dialect} (Standard)")
    print(f"  ├── Fingerprint:  {coords.fingerprint} (primitive composition)")
    print(f"  └── Collision:    {coords.collision} (disambiguator)")

    # Legacy integer form (if needed)
    print(f"\n  Integer form (legacy): {coords.to_token_id()}")

    # Example: "understand" - similar but different primitives
    primitives2 = [
        PrimitiveComponent(primitive_id=12, position=1),  # KNOW
        PrimitiveComponent(primitive_id=45, position=2),  # GRASP (different)
    ]

    coords2 = encoder.encode(
        lemma="understand",
        domain="mental",
        category="understanding",
        lang_iso="eng",
        primitives=primitives2
    )

    print(f"\n'understand':")
    print(f"  Genomic notation: {coords2}")
    print(f"  Fingerprint: {coords2.fingerprint}")

    # Compare similarity
    sim = encoder.similarity(coords, coords2)
    print(f"\nSimilarity between 'comprehend' and 'understand': {sim:.2f}")

    # Demonstrate collision handling
    print(f"\n--- Collision Demo ---")

    # Same primitives = same fingerprint = collision increment
    coords3 = encoder.encode(
        lemma="grasp",  # Different word
        domain="mental",
        category="understanding",
        lang_iso="eng",
        primitives=primitives  # Same primitives as "comprehend"
    )
    print(f"'grasp' with same primitives as 'comprehend':")
    print(f"  Genomic notation: {coords3}")
    print(f"  Fingerprint: {coords3.fingerprint} (same as comprehend)")
    print(f"  Collision: {coords3.collision} (incremented!)")

    # Cross-language example
    print(f"\n--- Cross-Language Demo ---")

    coords_fr = encoder.encode(
        lemma="comprendre",
        domain="mental",
        category="understanding",
        lang_iso="fra",
        primitives=primitives
    )
    print(f"'comprendre' (French):")
    print(f"  Genomic notation: {coords_fr}")
    print(f"  Language: {coords_fr.lang.to_string()} (Romance)")

    coords_de = encoder.encode(
        lemma="verstehen",
        domain="mental",
        category="understanding",
        lang_iso="deu",
        primitives=primitives
    )
    print(f"'verstehen' (German):")
    print(f"  Genomic notation: {coords_de}")
    print(f"  Language: {coords_de.lang.to_string()} (Germanic)")

    # Same fingerprint, different language families
    print(f"\nAll three share fingerprint {coords.fingerprint}:")
    print(f"  - comprehend (eng): {coords}")
    print(f"  - comprendre (fra): {coords_fr}")
    print(f"  - verstehen  (deu): {coords_de}")

    # Parsing demo
    print(f"\n--- Parsing Demo ---")
    test_str = "2.3.7.1.8.127.0.248.0"
    parsed = decode_token(test_str)
    print(f"Parsed '{test_str}':")
    print(f"  Domain: {parsed.domain}, Category: {parsed.category}")
    print(f"  Language: {parsed.lang.to_string()}")
    print(f"  Fingerprint: {parsed.fingerprint}")
