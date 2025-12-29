#!/usr/bin/env python3
"""
Extract concepts from etymology database at morpheme level.

This script:
1. Builds a graph from etymology links
2. Finds connected components (shared roots = shared concepts)
3. Identifies modifier patterns (affixes → universal modifiers)
4. Outputs concepts table with morpheme-level granularity

ID Space Layout:
  0 - 999,999:         Personal DI tokens (reserved)
  1,000,000 - 1,999,999: System primitives
  2,000,000+:          Etymology-derived concepts

Run on machine with full etymology database.
"""

import csv
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import hashlib
import re

# ID ranges
ID_SYSTEM = 0                # 0 - 999,999: System primitives, byte fallbacks, roots
ID_PERSONAL = 1_000_000      # 1M - 2M: Personal DI tokens (Navigator's space)
ID_CONCEPTS = 2_000_000      # 2M+: Etymology-derived concepts

# Known modifier affixes (cross-linguistic)
MODIFIER_AFFIXES = {
    # STATE_OF patterns
    'state_of': [
        '-ness', '-heit', '-keit', '-ость', '-ство', '-té', '-dad', '-ità',
        '-tion', '-sion', '-ment', '-ure', '-th', '-dom',
    ],
    # AGENT_OF patterns
    'agent_of': [
        '-er', '-or', '-eur', '-ier', '-ist', '-ant', '-ent',
        '-ник', '-тель', '-щик', '-чик',
    ],
    # NEGATION patterns
    'negation': [
        'un-', 'in-', 'im-', 'il-', 'ir-', 'non-', 'dis-', 'a-', 'an-',
        'не-', 'без-', 'un-', 'miss-',
    ],
    # ACT_OF patterns
    'act_of': [
        '-ing', '-ung', '-ation', '-ition', '-ание', '-ение', '-ство',
    ],
    # QUALITY_OF patterns
    'quality_of': [
        '-ly', '-lich', '-ment', '-weise', '-но', '-ски',
        '-ful', '-less', '-ous', '-ive', '-able', '-ible',
    ],
    # REPETITION patterns
    'repetition': [
        're-', 'пере-', 'wieder-', 'ri-',
    ],
    # CAUSATIVE patterns
    'causative': [
        '-ify', '-ize', '-ise', '-en', '-ieren', '-ить',
    ],
}

@dataclass
class MorphemeNode:
    """A node in the etymology graph"""
    lang: str
    term: str
    term_id: str = ""
    node_type: str = "word"  # 'word', 'root', 'prefix', 'suffix'

    def __hash__(self):
        return hash((self.lang, self.term))

    def __eq__(self, other):
        return self.lang == other.lang and self.term == other.term

@dataclass
class Concept:
    """An extracted concept"""
    id: int
    canonical: str           # Most representative form
    domain: str              # 'root', 'modifier', 'system'
    concept_type: str        # 'state_of', 'agent_of', etc. for modifiers
    proto_form: Optional[str] = None  # PIE or proto form if available
    members: Set[Tuple[str, str]] = field(default_factory=set)  # (lang, term) pairs
    confidence: float = 0.8


class ConceptExtractor:
    def __init__(self, csv_path: str, db_path: str):
        self.csv_path = csv_path
        self.db_path = db_path
        self.graph: Dict[MorphemeNode, Set[MorphemeNode]] = defaultdict(set)
        self.node_info: Dict[Tuple[str, str], MorphemeNode] = {}
        self.concepts: List[Concept] = []
        self.next_concept_id = ID_CONCEPTS

    def load_etymology_links(self):
        """Load etymology links and build graph"""
        print("Loading etymology links...")

        link_count = 0
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                link_count += 1
                if link_count % 500000 == 0:
                    print(f"  Processed {link_count:,} links...")

                lang = row['lang']
                term = row['term']
                term_id = row['term_id']
                reltype = row['reltype']
                related_lang = row.get('related_lang', '')
                related_term = row.get('related_term', '')

                if not related_term or not related_lang:
                    continue

                # Determine node types based on relationship
                source_type = self._infer_type(term, reltype, is_source=True)
                target_type = self._infer_type(related_term, reltype, is_source=False)

                # Create nodes
                source = MorphemeNode(lang, term, term_id, source_type)
                target = MorphemeNode(related_lang, related_term, "", target_type)

                # Store node info
                self.node_info[(lang, term)] = source
                self.node_info[(related_lang, related_term)] = target

                # Add bidirectional edge (for connected components)
                self.graph[source].add(target)
                self.graph[target].add(source)

        print(f"Loaded {link_count:,} links")
        print(f"Graph has {len(self.graph):,} nodes")

    def _infer_type(self, term: str, reltype: str, is_source: bool) -> str:
        """Infer morpheme type from term and relationship"""
        term_lower = term.lower().strip()

        # Check for prefix/suffix markers
        if term_lower.endswith('-') or reltype == 'has_prefix':
            return 'prefix'
        if term_lower.startswith('-') or reltype == 'has_suffix':
            return 'suffix'

        # Check for proto-language roots
        if term_lower.startswith('*') or 'Proto' in reltype:
            return 'root'
        if reltype == 'has_root':
            return 'root' if not is_source else 'word'

        return 'word'

    def find_connected_components(self) -> List[Set[MorphemeNode]]:
        """Find connected components in the etymology graph"""
        print("Finding connected components...")

        visited = set()
        components = []

        for node in self.graph:
            if node in visited:
                continue

            # BFS to find component
            component = set()
            queue = [node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) > 1:  # Only keep non-singleton components
                components.append(component)

        print(f"Found {len(components):,} connected components")
        return components

    def extract_concepts_from_components(self, components: List[Set[MorphemeNode]]):
        """Extract concepts from connected components"""
        print("Extracting concepts from components...")

        for component in components:
            # Find the deepest root (PIE or proto form)
            proto_form = None
            roots = []

            for node in component:
                if node.node_type == 'root':
                    roots.append(node)
                    if node.term.startswith('*'):
                        proto_form = node.term
                    elif 'Proto' in node.lang:
                        proto_form = node.term

            # Determine canonical form
            if proto_form:
                canonical = proto_form
            elif roots:
                canonical = roots[0].term
            else:
                # Use most common term or first English term
                english_terms = [n for n in component if n.lang == 'English']
                if english_terms:
                    canonical = english_terms[0].term
                else:
                    canonical = list(component)[0].term

            # Create concept
            concept = Concept(
                id=self.next_concept_id,
                canonical=canonical,
                domain='root',
                concept_type='lexical',
                proto_form=proto_form,
                members={(n.lang, n.term) for n in component},
                confidence=0.9 if proto_form else 0.7
            )

            self.concepts.append(concept)
            self.next_concept_id += 1

        print(f"Extracted {len(self.concepts):,} root concepts")

    def extract_modifier_concepts(self):
        """Extract universal modifier concepts from affix patterns"""
        print("Extracting modifier concepts...")

        modifier_id = ID_SYSTEM + 1000  # Start modifiers at 1000 (in system range)

        for modifier_type, affixes in MODIFIER_AFFIXES.items():
            # Find all instances of these affixes in our data
            members = set()

            for (lang, term), node in self.node_info.items():
                if node.node_type in ('prefix', 'suffix'):
                    term_clean = term.strip('-').lower()
                    for affix in affixes:
                        affix_clean = affix.strip('-').lower()
                        if term_clean == affix_clean:
                            members.add((lang, term))
                            break

            if members:
                concept = Concept(
                    id=modifier_id,
                    canonical=modifier_type.upper(),
                    domain='modifier',
                    concept_type=modifier_type,
                    members=members,
                    confidence=0.95
                )
                self.concepts.append(concept)
                modifier_id += 1

        print(f"Extracted {modifier_id - ID_SYSTEM - 100} modifier concepts")

    def save_to_database(self):
        """Save extracted concepts to database"""
        print(f"Saving to {self.db_path}...")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create concepts table if not exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY,
                canonical TEXT NOT NULL,
                domain TEXT,
                concept_type TEXT,
                proto_form TEXT,
                member_count INTEGER,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create concept_members table for the (lang, term) mappings
        c.execute('''
            CREATE TABLE IF NOT EXISTS concept_members (
                concept_id INTEGER,
                lang TEXT,
                term TEXT,
                PRIMARY KEY (concept_id, lang, term),
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        ''')

        # Insert concepts
        for concept in self.concepts:
            c.execute('''
                INSERT OR REPLACE INTO concepts
                (id, canonical, domain, concept_type, proto_form, member_count, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                concept.id,
                concept.canonical,
                concept.domain,
                concept.concept_type,
                concept.proto_form,
                len(concept.members),
                concept.confidence
            ))

            # Insert members
            for lang, term in concept.members:
                c.execute('''
                    INSERT OR IGNORE INTO concept_members (concept_id, lang, term)
                    VALUES (?, ?, ?)
                ''', (concept.id, lang, term))

        # Create indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_concept_domain ON concepts(domain)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_concept_type ON concepts(concept_type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_member_lang ON concept_members(lang)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_member_term ON concept_members(term)')

        conn.commit()
        conn.close()

        print(f"Saved {len(self.concepts):,} concepts to database")

    def print_stats(self):
        """Print extraction statistics"""
        print("\n=== Extraction Statistics ===")

        # Count by domain
        domain_counts = defaultdict(int)
        for c in self.concepts:
            domain_counts[c.domain] += 1

        print("\nConcepts by domain:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count:,}")

        # Count by type for modifiers
        print("\nModifier concepts:")
        for c in self.concepts:
            if c.domain == 'modifier':
                print(f"  {c.canonical}: {len(c.members)} affixes")

        # Sample root concepts
        print("\nSample root concepts (with PIE forms):")
        pie_concepts = [c for c in self.concepts if c.proto_form and c.proto_form.startswith('*')]
        for c in pie_concepts[:10]:
            print(f"  {c.id}: {c.canonical} ({len(c.members)} members)")

    def run(self):
        """Run the full extraction pipeline"""
        self.load_etymology_links()
        components = self.find_connected_components()
        self.extract_concepts_from_components(components)
        self.extract_modifier_concepts()
        self.save_to_database()
        self.print_stats()


def main():
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'etymology.csv'
    db_path = sys.argv[2] if len(sys.argv) > 2 else 'concepts.db'

    extractor = ConceptExtractor(csv_path, db_path)
    extractor.run()


if __name__ == '__main__':
    main()
