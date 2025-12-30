"""
Concept Tokenizer Library

The DB IS the tokenizer. Every word is a query.
"""

from .concept_tokenizer import (
    ConceptTokenizer,
    Token,
    tokenize,
    detokenize,
    CONCEPT_BASE,
    SYNSET_SLOTS,
    MODIFIER_RANGE,
    SYSTEM_RANGE,
)

__all__ = [
    'ConceptTokenizer',
    'Token',
    'tokenize',
    'detokenize',
    'CONCEPT_BASE',
    'SYNSET_SLOTS',
    'MODIFIER_RANGE',
    'SYSTEM_RANGE',
]
