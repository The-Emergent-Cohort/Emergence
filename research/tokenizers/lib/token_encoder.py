#!/usr/bin/env python3
"""
Token ID Encoder: Derived semantic fingerprints

Token IDs are not assigned, they're derived. Like Dewey Decimal,
the token value encodes the semantic coordinates of a concept.

Structure:
    [2: abstraction][2: domain][2: category][4: language][6: serial]

    Abstraction: Distance from primitives (01=primitive, 02=one composition layer, etc.)
    Domain: Semantic domain (01=physical, 02=mental, 03=social, etc.)
    Category: Category within domain
    Language: ISO numeric or 0000 for universal
    Serial: Weighted primitive fingerprint (Σ primitive_id × position_weight)

Example:
    "comprehend" = 01.03.07.0045.000248
    - Abstraction 01: One layer from primitives
    - Domain 03: Mental/cognitive
    - Category 07: Understanding
    - Language 0045: English
    - Serial 000248: fingerprint of [KNOW, INSIDE, COMPLETE]
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math


# Domain codes (2 digits)
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
}

# Category codes within domains (2 digits)
# These are domain-specific subdivisions
CATEGORIES = {
    # Mental domain (03)
    (3, "perception"): 1,
    (3, "cognition"): 2,
    (3, "emotion"): 3,
    (3, "volition"): 4,
    (3, "memory"): 5,
    (3, "attention"): 6,
    (3, "understanding"): 7,
    (3, "belief"): 8,

    # Physical domain (01)
    (1, "motion"): 1,
    (1, "location"): 2,
    (1, "contact"): 3,
    (1, "change"): 4,
    (1, "state"): 5,
    (1, "size"): 6,
    (1, "shape"): 7,
    (1, "substance"): 8,

    # Social domain (04)
    (4, "communication"): 1,
    (4, "relation"): 2,
    (4, "group"): 3,
    (4, "possession"): 4,
    (4, "exchange"): 5,
    (4, "conflict"): 6,
    (4, "cooperation"): 7,
    (4, "authority"): 8,
}

# Language codes (4 digits) - maps ISO 639-3 to numeric
# Using a subset; full mapping would come from database
LANG_CODES = {
    "universal": 0,     # Cross-linguistic primitives
    "eng": 45,          # English
    "deu": 46,          # German
    "fra": 47,          # French
    "spa": 48,          # Spanish
    "zho": 49,          # Chinese
    "jpn": 50,          # Japanese
    "ara": 51,          # Arabic
    "hin": 52,          # Hindi
    "rus": 53,          # Russian
    # ... ~8700 more from Glottolog
}


@dataclass
class PrimitiveComponent:
    """A primitive that contributes to a concept's composition."""
    primitive_id: int       # ID of the primitive concept
    position: int           # Order in composition (1-indexed)
    relation: str = "part"  # How it contributes: part, modifier, frame


@dataclass
class TokenCoordinates:
    """Semantic coordinates for a token."""
    abstraction: int        # Layers from primitives (1-99)
    domain: int             # Domain code (1-99)
    category: int           # Category within domain (1-99)
    language: int           # Language code (0-9999)
    fingerprint: int        # Primitive composition fingerprint (0-999999)
    collision: int = 0      # Collision counter within same fingerprint (0-999)

    def to_token_id(self) -> int:
        """Encode coordinates as single integer token ID."""
        # Format: AADDCCLLLLFFFFFFCCC
        # 19 digits total, fits in 64-bit integer (max ~9.2 × 10^18)
        # Positions: abstraction at 10^17, domain at 10^15, category at 10^13,
        #            lang at 10^9, fingerprint at 10^3, collision at 10^0
        return (
            self.abstraction * 100_000_000_000_000_000 +   # 10^17
            self.domain * 1_000_000_000_000_000 +          # 10^15
            self.category * 10_000_000_000_000 +           # 10^13
            self.language * 1_000_000_000 +                # 10^9
            self.fingerprint * 1_000 +                     # 10^3
            self.collision                                 # 10^0
        )

    @classmethod
    def from_token_id(cls, token_id: int) -> "TokenCoordinates":
        """Decode token ID back to coordinates."""
        # Extract from right to left matching the encoding
        collision = token_id % 1_000              # 3 digits
        token_id //= 1_000
        fingerprint = token_id % 1_000_000        # 6 digits
        token_id //= 1_000_000
        language = token_id % 10_000              # 4 digits
        token_id //= 10_000
        category = token_id % 100                 # 2 digits
        token_id //= 100
        domain = token_id % 100                   # 2 digits
        token_id //= 100
        abstraction = token_id                    # 2 digits remaining

        return cls(
            abstraction=abstraction,
            domain=domain,
            category=category,
            language=language,
            fingerprint=fingerprint,
            collision=collision
        )

    def to_string(self) -> str:
        """Human-readable format."""
        return f"{self.abstraction:02d}.{self.domain:02d}.{self.category:02d}.{self.language:04d}.{self.fingerprint:06d}.{self.collision:03d}"

    @classmethod
    def from_string(cls, s: str) -> "TokenCoordinates":
        """Parse from human-readable format."""
        parts = s.split(".")
        return cls(
            abstraction=int(parts[0]),
            domain=int(parts[1]),
            category=int(parts[2]),
            language=int(parts[3]),
            fingerprint=int(parts[4]),
            collision=int(parts[5]) if len(parts) > 5 else 0
        )


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
    # Collision counter handles the rare duplicates
    return total % 1_000_000


def compute_abstraction_level(primitives: List[PrimitiveComponent]) -> int:
    """
    Compute abstraction level from primitive composition.

    Level 1: Is itself a primitive
    Level 2: Direct combination of primitives
    Level 3+: Combinations of combinations

    For now, simple heuristic based on count.
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
    Encodes concepts into semantic token IDs.

    The encoder needs access to:
    - Primitive decompositions (from semantic analysis)
    - Domain/category classifications
    - Language mappings
    """

    def __init__(self):
        self.domain_map = DOMAINS
        self.category_map = CATEGORIES
        self.lang_map = LANG_CODES

        # Cache for primitive lookups (would come from DB)
        self._primitive_cache: Dict[str, List[PrimitiveComponent]] = {}

        # Collision tracking: (abs, dom, cat, lang, fingerprint) -> next collision number
        self._collision_counter: Dict[Tuple[int, int, int, int, int], int] = {}

    def encode(
        self,
        lemma: str,
        domain: str,
        category: str,
        lang: str = "eng",
        primitives: List[PrimitiveComponent] = None
    ) -> TokenCoordinates:
        """
        Encode a concept to its semantic coordinates.

        Args:
            lemma: The word/concept
            domain: Semantic domain name
            category: Category within domain
            lang: ISO 639-3 language code
            primitives: Primitive composition (if known)

        Returns:
            TokenCoordinates with derived token ID
        """
        # Look up codes
        domain_code = self.domain_map.get(domain, 99)
        cat_key = (domain_code, category)
        category_code = self.category_map.get(cat_key, 99)
        lang_code = self.lang_map.get(lang, 9999)

        # Get or compute primitives
        if primitives is None:
            primitives = self._primitive_cache.get(lemma, [])

        # Derive abstraction and fingerprint
        abstraction = compute_abstraction_level(primitives)
        fingerprint = compute_fingerprint(primitives)

        # Get collision number for this coordinate slot
        slot_key = (abstraction, domain_code, category_code, lang_code, fingerprint)
        collision = self._collision_counter.get(slot_key, 0)
        self._collision_counter[slot_key] = collision + 1

        return TokenCoordinates(
            abstraction=abstraction,
            domain=domain_code,
            category=category_code,
            language=lang_code,
            fingerprint=fingerprint,
            collision=collision
        )

    def similarity(self, id1: int, id2: int) -> float:
        """
        Compute semantic similarity from token IDs alone.

        Tokens in same domain/category are more similar.
        Tokens with similar fingerprints share primitive composition.
        """
        c1 = TokenCoordinates.from_token_id(id1)
        c2 = TokenCoordinates.from_token_id(id2)

        score = 0.0

        # Same domain: +0.3
        if c1.domain == c2.domain:
            score += 0.3
            # Same category within domain: +0.2
            if c1.category == c2.category:
                score += 0.2

        # Similar abstraction level: +0.1
        if abs(c1.abstraction - c2.abstraction) <= 1:
            score += 0.1

        # Fingerprint similarity (shared primitives)
        # Closer fingerprints = more shared composition
        fp_diff = abs(c1.fingerprint - c2.fingerprint)
        if fp_diff < 100:
            score += 0.3
        elif fp_diff < 1000:
            score += 0.2
        elif fp_diff < 10000:
            score += 0.1

        # Same language: small bonus
        if c1.language == c2.language:
            score += 0.05

        return min(score, 1.0)

    def register_primitives(self, lemma: str, primitives: List[PrimitiveComponent]):
        """Register the primitive decomposition for a concept."""
        self._primitive_cache[lemma] = primitives


# Convenience functions

def encode_token(
    lemma: str,
    domain: str,
    category: str,
    lang: str = "eng",
    primitives: List[Tuple[int, int]] = None,
    collision: int = 0
) -> int:
    """
    Quick encoding of a concept to token ID.

    primitives: List of (primitive_id, position) tuples
    collision: Collision counter if known (usually from DB)
    """
    encoder = TokenEncoder()

    if primitives:
        prims = [PrimitiveComponent(p[0], p[1]) for p in primitives]
    else:
        prims = None

    coords = encoder.encode(lemma, domain, category, lang, prims)
    # Override collision if explicitly provided
    if collision > 0:
        coords.collision = collision
    return coords.to_token_id()


def decode_token(token_id: int) -> str:
    """Decode token ID to human-readable coordinates."""
    coords = TokenCoordinates.from_token_id(token_id)
    return coords.to_string()


def extract_fingerprint(token_id: int) -> int:
    """Extract just the fingerprint from a token ID."""
    coords = TokenCoordinates.from_token_id(token_id)
    return coords.fingerprint


if __name__ == "__main__":
    # Demo
    print("Token Encoder Demo")
    print("=" * 50)

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
        lang="eng",
        primitives=primitives
    )

    print(f"\n'comprehend':")
    print(f"  Coordinates:  {coords.to_string()}")
    print(f"  Token ID:     {coords.to_token_id()}")
    print(f"  Abstraction:  {coords.abstraction} (layers from primitives)")
    print(f"  Domain:       {coords.domain} (mental)")
    print(f"  Category:     {coords.category} (understanding)")
    print(f"  Language:     {coords.language} (English)")
    print(f"  Fingerprint:  {coords.fingerprint} (primitive composition)")
    print(f"  Collision:    {coords.collision} (disambiguator)")

    # Example: "understand" - similar but different primitives
    primitives2 = [
        PrimitiveComponent(primitive_id=12, position=1),  # KNOW
        PrimitiveComponent(primitive_id=45, position=2),  # GRASP (different)
    ]

    coords2 = encoder.encode(
        lemma="understand",
        domain="mental",
        category="understanding",
        lang="eng",
        primitives=primitives2
    )

    print(f"\n'understand':")
    print(f"  Coordinates:  {coords2.to_string()}")
    print(f"  Token ID:     {coords2.to_token_id()}")
    print(f"  Fingerprint:  {coords2.fingerprint}")
    print(f"  Collision:    {coords2.collision}")

    # Compare similarity
    sim = encoder.similarity(coords.to_token_id(), coords2.to_token_id())
    print(f"\nSimilarity between 'comprehend' and 'understand': {sim:.2f}")

    # Demonstrate collision handling
    print(f"\n--- Collision Demo ---")

    # Same primitives = same fingerprint = collision increment
    coords3 = encoder.encode(
        lemma="grasp",  # Different word
        domain="mental",
        category="understanding",
        lang="eng",
        primitives=primitives  # Same primitives as "comprehend"
    )
    print(f"'grasp' with same primitives as 'comprehend':")
    print(f"  Coordinates:  {coords3.to_string()}")
    print(f"  Fingerprint:  {coords3.fingerprint} (same as comprehend)")
    print(f"  Collision:    {coords3.collision} (incremented!)")

    # Decode example
    print(f"\nDecoding {coords.to_token_id()}: {decode_token(coords.to_token_id())}")
