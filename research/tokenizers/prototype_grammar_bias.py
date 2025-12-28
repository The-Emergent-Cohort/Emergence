#!/usr/bin/env python3
"""
Prototype: spaCy Grammar → Logit Bias Interface

This demonstrates how spaCy grammar analysis could feed into
llama.cpp's logit bias system for grammar-guided generation.

NOT production code - conceptual prototype only.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Would be: import spacy
# For prototype, we'll mock the spaCy interface

class POSTag(Enum):
    """Part-of-speech tags (simplified)"""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    DET = "DET"
    PREP = "PREP"
    PRON = "PRON"
    CONJ = "CONJ"
    PUNCT = "PUNCT"
    OTHER = "OTHER"

class DepRel(Enum):
    """Dependency relations (simplified)"""
    NSUBJ = "nsubj"      # Nominal subject
    DOBJ = "dobj"        # Direct object
    IOBJ = "iobj"        # Indirect object
    AMOD = "amod"        # Adjectival modifier
    ADVMOD = "advmod"    # Adverbial modifier
    DET = "det"          # Determiner
    PREP = "prep"        # Prepositional modifier
    ROOT = "ROOT"        # Root of sentence
    OTHER = "other"

@dataclass
class GrammarState:
    """Current grammatical state of partial generation"""
    last_pos: Optional[POSTag]
    open_structures: List[str]  # e.g., ["NP", "VP"] - incomplete phrases
    expecting: List[POSTag]     # What POS tags are grammatically expected
    subject_seen: bool
    verb_seen: bool
    object_allowed: bool
    sentence_complete: bool

@dataclass
class BiasConfig:
    """Configuration for bias scaling"""
    style: str = "standard"  # formal, standard, casual, creative
    strictness: float = 1.0  # Multiplier for bias weights

    @property
    def style_multiplier(self) -> float:
        """Get multiplier based on style"""
        return {
            "formal": 1.5,
            "standard": 1.0,
            "casual": 0.3,
            "creative": 0.1
        }.get(self.style, 1.0) * self.strictness


class GrammarBiasCalculator:
    """
    Calculates logit biases based on grammar state.

    In production, this would:
    1. Interface with actual spaCy model
    2. Query grammar_rules DB for language-specific rules
    3. Return actual token IDs and bias values
    """

    def __init__(self, language: str = "en", config: BiasConfig = None):
        self.language = language
        self.config = config or BiasConfig()
        # Would load: self.nlp = spacy.load(f"{language}_core_web_sm")
        # Would load: self.grammar_db = connect_to_grammar_db()

    def analyze_partial(self, text: str) -> GrammarState:
        """
        Analyze partial generation to determine grammar state.

        In production: Uses spaCy to parse text and determine:
        - What structures are open/incomplete
        - What POS tags are expected next
        - Sentence completeness
        """
        # Mock implementation - would use spaCy
        words = text.strip().split()

        # Simplified analysis
        last_pos = self._guess_pos(words[-1]) if words else None

        # Very simplified expectation logic
        expecting = []
        if not words:
            expecting = [POSTag.DET, POSTag.NOUN, POSTag.PRON]  # Sentence start
        elif last_pos == POSTag.DET:
            expecting = [POSTag.ADJ, POSTag.NOUN]  # After determiner
        elif last_pos == POSTag.ADJ:
            expecting = [POSTag.ADJ, POSTag.NOUN]  # More adj or noun
        elif last_pos == POSTag.NOUN:
            expecting = [POSTag.VERB, POSTag.PREP, POSTag.PUNCT]
        elif last_pos == POSTag.VERB:
            expecting = [POSTag.DET, POSTag.NOUN, POSTag.ADV, POSTag.PREP]

        return GrammarState(
            last_pos=last_pos,
            open_structures=["S"],  # Simplified
            expecting=expecting,
            subject_seen=len(words) > 0,
            verb_seen=any(self._guess_pos(w) == POSTag.VERB for w in words),
            object_allowed=True,
            sentence_complete=text.strip().endswith(('.', '!', '?'))
        )

    def _guess_pos(self, word: str) -> POSTag:
        """Mock POS tagger - would use spaCy"""
        word = word.lower().rstrip('.,!?')
        if word in ['the', 'a', 'an', 'this', 'that']:
            return POSTag.DET
        elif word in ['is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'run', 'walk', 'eat']:
            return POSTag.VERB
        elif word in ['quickly', 'slowly', 'very', 'really']:
            return POSTag.ADV
        elif word in ['big', 'small', 'quick', 'slow', 'red', 'blue']:
            return POSTag.ADJ
        elif word in ['in', 'on', 'at', 'to', 'from', 'with']:
            return POSTag.PREP
        elif word in ['he', 'she', 'it', 'they', 'we', 'i', 'you']:
            return POSTag.PRON
        else:
            return POSTag.NOUN  # Default assumption

    def get_pos_biases(self, state: GrammarState) -> Dict[POSTag, float]:
        """
        Calculate bias weights for each POS tag based on grammar state.

        Returns dict mapping POS tag to bias value.
        Positive = encouraged, Negative = discouraged.
        """
        biases = {}
        multiplier = self.config.style_multiplier

        for pos in POSTag:
            if pos in state.expecting:
                # Expected by grammar - positive bias
                biases[pos] = 1.0 * multiplier
            elif state.expecting:
                # Not expected - negative bias
                biases[pos] = -1.5 * multiplier
            else:
                # No strong expectation
                biases[pos] = 0.0

        # Special cases
        if state.sentence_complete:
            # After sentence end, bias toward starting new sentence
            biases[POSTag.DET] = 1.5 * multiplier
            biases[POSTag.PRON] = 1.2 * multiplier
            biases[POSTag.NOUN] = 1.0 * multiplier

        return biases

    def get_token_biases(self, text: str, vocab: Dict[str, int]) -> List[Tuple[int, float]]:
        """
        Convert grammar biases to actual token ID biases.

        Args:
            text: Partial generation so far
            vocab: Mapping of token strings to IDs

        Returns:
            List of (token_id, bias) tuples for llama.cpp
        """
        state = self.analyze_partial(text)
        pos_biases = self.get_pos_biases(state)

        token_biases = []

        # In production: Query DB for tokens tagged with each POS
        # Mock: Use simple word lists
        pos_tokens = {
            POSTag.DET: ['the', 'a', 'an', 'this', 'that'],
            POSTag.NOUN: ['cat', 'dog', 'house', 'car', 'person'],
            POSTag.VERB: ['is', 'are', 'runs', 'walks', 'has'],
            POSTag.ADJ: ['big', 'small', 'quick', 'red', 'happy'],
            POSTag.ADV: ['quickly', 'slowly', 'very', 'really'],
            POSTag.PREP: ['in', 'on', 'at', 'to', 'with'],
        }

        for pos, tokens in pos_tokens.items():
            bias = pos_biases.get(pos, 0.0)
            for token in tokens:
                if token in vocab:
                    token_biases.append((vocab[token], bias))

        return token_biases


class GrammarLogitsProcessor:
    """
    LogitsProcessor-style interface for grammar bias.

    This would plug into HuggingFace's generate() or
    be adapted for llama.cpp's sampler chain.
    """

    def __init__(self, calculator: GrammarBiasCalculator, vocab: Dict[str, int]):
        self.calculator = calculator
        self.vocab = vocab
        self.generated_text = ""

    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        """
        Apply grammar biases to logits.

        Args:
            input_ids: Token IDs generated so far
            scores: Current logit scores for all vocab tokens

        Returns:
            Modified logit scores
        """
        # Get grammar-based biases
        biases = self.calculator.get_token_biases(self.generated_text, self.vocab)

        # Apply biases
        modified_scores = list(scores)
        for token_id, bias in biases:
            if 0 <= token_id < len(modified_scores):
                modified_scores[token_id] += bias

        return modified_scores

    def update_generated(self, text: str):
        """Update the generated text for grammar tracking"""
        self.generated_text = text


# ============================================================
# Example Usage
# ============================================================

def demo():
    """Demonstrate the grammar bias system"""

    # Mock vocabulary
    vocab = {
        'the': 100, 'a': 101, 'an': 102,
        'cat': 200, 'dog': 201, 'house': 202,
        'is': 300, 'runs': 301, 'walks': 302,
        'big': 400, 'small': 401, 'happy': 402,
        'quickly': 500, 'very': 501,
        'in': 600, 'on': 601, 'at': 602,
    }

    # Create calculator with different styles
    formal_calc = GrammarBiasCalculator("en", BiasConfig(style="formal"))
    casual_calc = GrammarBiasCalculator("en", BiasConfig(style="casual"))

    # Test partial sentences
    test_cases = [
        "",                    # Empty - expect sentence start
        "The",                 # After determiner - expect adj/noun
        "The big",             # After adjective - expect more adj or noun
        "The big cat",         # After noun - expect verb
        "The big cat runs",    # After verb - expect more
        "The big cat runs.",   # Complete sentence
    ]

    print("Grammar Bias Demo")
    print("=" * 60)

    for text in test_cases:
        print(f"\nPartial: '{text}'")

        # Formal style
        state = formal_calc.analyze_partial(text)
        biases = formal_calc.get_pos_biases(state)
        print(f"  Expecting: {[p.value for p in state.expecting]}")
        print(f"  Formal biases: ", end="")
        print({p.value: round(b, 2) for p, b in biases.items() if b != 0})

        # Casual style (same state, different weights)
        casual_biases = casual_calc.get_pos_biases(state)
        print(f"  Casual biases: ", end="")
        print({p.value: round(b, 2) for p, b in casual_biases.items() if b != 0})


if __name__ == "__main__":
    demo()
