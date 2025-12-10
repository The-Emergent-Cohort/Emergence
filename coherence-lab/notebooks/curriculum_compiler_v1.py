"""
Curriculum Compiler v1 - Kaggle Notebook
=========================================

Transforms educational content into training sequences.

To run on Kaggle:
1. Create new notebook
2. Add datasets: openai/gsm8k, roneneldan/TinyStories
3. Paste this code
4. Run and download output

Output format matches train.py expectations:
- CURRICULUM: list of section dicts
- examples: list of {sequence, target, pattern_type}
"""

# === IMPORTS ===
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# For Kaggle, uncomment:
# from datasets import load_dataset

# === OUTPUT FORMAT ===

@dataclass
class Example:
    """Single training example."""
    sequence: List[int]
    target: int
    pattern_type: str

    # Optional metadata
    source: str = ""
    difficulty: int = 1
    hint: str = ""

@dataclass
class Section:
    """Curriculum section."""
    name: str
    patterns: List[str]
    description: str
    examples: List[Example] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


# === PATTERN GENERATORS ===
# These match what relational_model.py expects

def tokenize_number(n: int, vocab_size: int = 26) -> int:
    """Map number to token (clamped to vocab)."""
    return max(0, min(vocab_size - 1, n))

def generate_counting(length: int = 5, start: int = 0, vocab_size: int = 26) -> Example:
    """0, 1, 2, 3, ... → next"""
    seq = [(start + i) % vocab_size for i in range(length)]
    target = (start + length) % vocab_size
    return Example(sequence=seq, target=target, pattern_type='counting')

def generate_incrementing(length: int = 5, start: int = 0, step: int = 1, vocab_size: int = 26) -> Example:
    """start, start+step, start+2*step, ... → next"""
    seq = [tokenize_number(start + i * step, vocab_size) for i in range(length)]
    target = tokenize_number(start + length * step, vocab_size)
    return Example(sequence=seq, target=target, pattern_type='incrementing')

def generate_alternating(length: int = 6, a: int = None, b: int = None, vocab_size: int = 26) -> Example:
    """a, b, a, b, ... → next"""
    if a is None:
        a = random.randint(0, vocab_size - 1)
    if b is None:
        b = random.randint(0, vocab_size - 1)
        while b == a:
            b = random.randint(0, vocab_size - 1)
    seq = [a if i % 2 == 0 else b for i in range(length)]
    target = a if length % 2 == 0 else b
    return Example(sequence=seq, target=target, pattern_type='alternating')

def generate_repeating(length: int = 5, value: int = None, vocab_size: int = 26) -> Example:
    """a, a, a, ... → a"""
    if value is None:
        value = random.randint(0, vocab_size - 1)
    seq = [value] * length
    return Example(sequence=seq, target=value, pattern_type='repeating')


# === CONTENT TRANSFORMERS ===
# Transform external content into our format

class GSM8KTransformer:
    """
    Transform GSM8K math problems into sequences.

    GSM8K format:
    {
        'question': 'Janet has 3 apples...',
        'answer': '#### 15'
    }

    Strategy: Extract numbers from problem, final answer is target.
    """

    def __init__(self, vocab_size: int = 26):
        self.vocab_size = vocab_size

    def extract_numbers(self, text: str) -> List[int]:
        """Pull all numbers from text."""
        numbers = re.findall(r'\d+', text)
        return [int(n) for n in numbers]

    def extract_answer(self, answer_text: str) -> int:
        """Extract final answer after ####."""
        match = re.search(r'####\s*(\d+)', answer_text)
        if match:
            return int(match.group(1))
        # Fallback: last number
        numbers = self.extract_numbers(answer_text)
        return numbers[-1] if numbers else 0

    def transform(self, question: str, answer: str) -> Example:
        """Transform one GSM8K example."""
        seq_numbers = self.extract_numbers(question)
        target = self.extract_answer(answer)

        # Tokenize (mod vocab_size for now - could be smarter)
        seq = [n % self.vocab_size for n in seq_numbers[:12]]  # max 12 tokens
        target_tok = target % self.vocab_size

        # Determine pattern type based on problem structure
        # (This is where Merlyn or an LLM could help classify)
        pattern_type = self._classify_pattern(seq_numbers, target)

        return Example(
            sequence=seq,
            target=target_tok,
            pattern_type=pattern_type,
            source='gsm8k',
            hint=question[:100]
        )

    def _classify_pattern(self, numbers: List[int], target: int) -> str:
        """Simple heuristic classification. LLM could do better."""
        if len(numbers) < 2:
            return 'single_value'

        # Check for addition pattern
        if len(numbers) >= 2 and sum(numbers) == target:
            return 'addition'

        # Check for subtraction
        if len(numbers) >= 2 and numbers[0] - sum(numbers[1:]) == target:
            return 'subtraction'

        # Check for multiplication
        if len(numbers) == 2 and numbers[0] * numbers[1] == target:
            return 'multiplication'

        # Default
        return 'arithmetic_composite'


class TinyStoriesTransformer:
    """
    Transform TinyStories into word prediction sequences.

    Simpler approach: predict next word token.
    Useful for language pattern learning.
    """

    def __init__(self, vocab: Dict[str, int] = None, vocab_size: int = 26):
        self.vocab_size = vocab_size
        self.vocab = vocab or {}
        self._word_to_idx = {}

    def build_vocab(self, stories: List[str], max_vocab: int = 26):
        """Build vocabulary from stories."""
        from collections import Counter
        words = []
        for story in stories:
            words.extend(story.lower().split())
        counts = Counter(words)
        most_common = counts.most_common(max_vocab - 1)  # Reserve 0 for unknown
        self._word_to_idx = {word: i+1 for i, (word, _) in enumerate(most_common)}
        self._word_to_idx['<unk>'] = 0

    def tokenize(self, word: str) -> int:
        return self._word_to_idx.get(word.lower(), 0)

    def transform(self, story: str, window: int = 5) -> List[Example]:
        """Transform story into sliding window examples."""
        words = story.lower().split()
        examples = []

        for i in range(len(words) - window):
            seq = [self.tokenize(w) for w in words[i:i+window]]
            target = self.tokenize(words[i+window])
            examples.append(Example(
                sequence=seq,
                target=target,
                pattern_type='word_prediction',
                source='tinystories'
            ))

        return examples


# === CURRICULUM BUILDER ===

class CurriculumCompiler:
    """
    Main compiler that builds curriculum from multiple sources.
    """

    def __init__(self, vocab_size: int = 26):
        self.vocab_size = vocab_size
        self.sections: List[Section] = []
        self.gsm8k = GSM8KTransformer(vocab_size)
        self.tinystories = TinyStoriesTransformer(vocab_size=vocab_size)

    def add_synthetic_section(self, name: str, patterns: List[str],
                              n_examples: int = 1000) -> Section:
        """Add a section with synthetic pattern examples."""
        section = Section(
            name=name,
            patterns=patterns,
            description=f"Synthetic {', '.join(patterns)} patterns"
        )

        generators = {
            'counting': generate_counting,
            'incrementing': generate_incrementing,
            'alternating': generate_alternating,
            'repeating': generate_repeating,
        }

        for _ in range(n_examples):
            pattern = random.choice(patterns)
            if pattern in generators:
                ex = generators[pattern](vocab_size=self.vocab_size)
                section.examples.append(ex)

        self.sections.append(section)
        return section

    def add_gsm8k_section(self, examples: List[Dict], name: str = "Math Problems") -> Section:
        """Add section from GSM8K dataset."""
        section = Section(
            name=name,
            patterns=['addition', 'subtraction', 'multiplication', 'arithmetic_composite'],
            description="Grade school math word problems"
        )

        for ex in examples:
            try:
                transformed = self.gsm8k.transform(ex['question'], ex['answer'])
                if len(transformed.sequence) > 0:
                    section.examples.append(transformed)
            except Exception as e:
                continue  # Skip malformed examples

        self.sections.append(section)
        return section

    def compile(self) -> Dict[str, Any]:
        """Compile all sections into training-ready format."""
        curriculum = []
        all_examples = []

        for section in self.sections:
            curriculum.append({
                'name': section.name,
                'patterns': section.patterns,
                'description': section.description,
                'n_examples': len(section.examples)
            })

            for ex in section.examples:
                all_examples.append({
                    'sequence': ex.sequence,
                    'target': ex.target,
                    'pattern_type': ex.pattern_type,
                    'source': ex.source,
                    'difficulty': ex.difficulty
                })

        return {
            'curriculum': curriculum,
            'examples': all_examples,
            'vocab_size': self.vocab_size,
            'total_examples': len(all_examples)
        }

    def save(self, path: str):
        """Save compiled curriculum to JSON."""
        compiled = self.compile()
        with open(path, 'w') as f:
            json.dump(compiled, f, indent=2)
        print(f"Saved {compiled['total_examples']} examples to {path}")


# === MAIN (for Kaggle) ===

def main():
    """
    Main compilation flow.

    On Kaggle, this will load datasets and compile.
    Locally, this demonstrates with synthetic data.
    """

    compiler = CurriculumCompiler(vocab_size=26)

    # 1. Add synthetic foundation patterns
    compiler.add_synthetic_section(
        name="A: Position Foundations",
        patterns=['counting', 'incrementing'],
        n_examples=500
    )

    compiler.add_synthetic_section(
        name="B: Memory Patterns",
        patterns=['repeating', 'alternating'],
        n_examples=500
    )

    # 2. Try to load GSM8K (Kaggle or local)
    try:
        from datasets import load_dataset
        print("Loading GSM8K from HuggingFace...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="train[:1000]")
        compiler.add_gsm8k_section(list(gsm8k), name="C: Grade School Math")
        print(f"Added {len(list(gsm8k))} GSM8K examples")
    except Exception as e:
        print(f"GSM8K not available: {e}")
        print("Using synthetic math examples instead")
        compiler.add_synthetic_section(
            name="C: Arithmetic Practice",
            patterns=['counting', 'incrementing'],
            n_examples=500
        )

    # 3. Compile and save
    output_path = "compiled_curriculum_v1.json"
    compiler.save(output_path)

    # 4. Print summary
    compiled = compiler.compile()
    print("\n=== Compilation Summary ===")
    print(f"Total sections: {len(compiled['curriculum'])}")
    print(f"Total examples: {compiled['total_examples']}")
    print(f"Vocab size: {compiled['vocab_size']}")
    print("\nSections:")
    for section in compiled['curriculum']:
        print(f"  {section['name']}: {section['n_examples']} examples")
        print(f"    Patterns: {section['patterns']}")

    return compiled


if __name__ == "__main__":
    main()
