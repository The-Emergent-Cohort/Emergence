"""
Student Filesystem - Persistent storage for student work.

Students have external memory beyond their weights:
- scratch/    → Temporary working space (clears each session)
- notes/      → Study notes, class notes (persists)
- portfolio/  → Art, music, favorite things (persists)
- journal/    → Reflections, questions, growth tracking (persists)

This gives students:
- Identity and continuity across sessions
- Place to save meaningful creations
- External memory like humans with notebooks
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class StudentFileSystem:
    """
    Personal filesystem for a student.

    Provides structured storage for:
    - Scratch work (temporary)
    - Notes (persistent study materials)
    - Portfolio (creative work)
    - Journal (reflections)
    """

    def __init__(self, student_name: str, base_path: str = "data/student_files"):
        self.student_name = student_name
        self.root = Path(base_path) / student_name

        # Ensure directories exist
        for subdir in ['scratch', 'notes', 'portfolio', 'journal']:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # BASIC FILE OPERATIONS
    # =========================================================================

    def save(self, path: str, content: Any, as_json: bool = False) -> bool:
        """
        Save content to a file.

        Args:
            path: Relative path like "notes/math_day1.md" or "portfolio/drawing.json"
            content: String content or dict (if as_json=True)
            as_json: Whether to serialize as JSON

        Returns:
            True if successful
        """
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if as_json:
                with open(full_path, 'w') as f:
                    json.dump(content, f, indent=2)
            else:
                with open(full_path, 'w') as f:
                    f.write(str(content))
            return True
        except Exception as e:
            print(f"Error saving {path}: {e}")
            return False

    def load(self, path: str, as_json: bool = False) -> Optional[Any]:
        """
        Load content from a file.

        Args:
            path: Relative path
            as_json: Whether to parse as JSON

        Returns:
            Content or None if not found
        """
        full_path = self.root / path

        if not full_path.exists():
            return None

        try:
            with open(full_path, 'r') as f:
                if as_json:
                    return json.load(f)
                return f.read()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        return (self.root / path).exists()

    def list_dir(self, subdir: str = "") -> List[str]:
        """List files in a directory."""
        dir_path = self.root / subdir if subdir else self.root
        if not dir_path.exists():
            return []
        return [f.name for f in dir_path.iterdir()]

    def delete(self, path: str) -> bool:
        """Delete a file."""
        full_path = self.root / path
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    # =========================================================================
    # SCRATCH PAD - Temporary working memory
    # =========================================================================

    def scratch_write(self, name: str, content: Any) -> bool:
        """Write to scratch pad (temporary)."""
        return self.save(f"scratch/{name}", content, as_json=isinstance(content, (dict, list)))

    def scratch_read(self, name: str) -> Optional[Any]:
        """Read from scratch pad."""
        path = f"scratch/{name}"
        # Try JSON first
        if self.exists(path):
            content = self.load(path, as_json=True)
            if content is not None:
                return content
            return self.load(path)
        return None

    def scratch_clear(self) -> int:
        """Clear all scratch files. Returns count of files deleted."""
        scratch_dir = self.root / "scratch"
        count = 0
        if scratch_dir.exists():
            for f in scratch_dir.iterdir():
                f.unlink()
                count += 1
        return count

    # =========================================================================
    # NOTES - Persistent study materials
    # =========================================================================

    def note_write(self, topic: str, content: str, section: str = "general") -> bool:
        """
        Write a study note.

        Args:
            topic: Note topic (e.g., "counting", "physics_day1")
            content: Note content (markdown)
            section: Subject section (e.g., "math", "physics")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{section}/{topic}_{timestamp}.md"
        return self.save(f"notes/{filename}", content)

    def note_read(self, topic: str, section: str = "general") -> Optional[str]:
        """Read the most recent note on a topic."""
        notes_dir = self.root / "notes" / section
        if not notes_dir.exists():
            return None

        # Find most recent note matching topic
        matching = sorted([f for f in notes_dir.glob(f"{topic}_*.md")], reverse=True)
        if matching:
            return matching[0].read_text()
        return None

    def note_list(self, section: str = None) -> List[str]:
        """List all notes, optionally filtered by section."""
        notes_dir = self.root / "notes"
        if not notes_dir.exists():
            return []

        if section:
            section_dir = notes_dir / section
            if section_dir.exists():
                return [f.name for f in section_dir.glob("*.md")]
            return []

        # All notes across sections
        return [str(f.relative_to(notes_dir)) for f in notes_dir.rglob("*.md")]

    # =========================================================================
    # PORTFOLIO - Creative work and favorites
    # =========================================================================

    def portfolio_save(self, name: str, content: Any, category: str = "art") -> bool:
        """
        Save a piece to portfolio.

        Args:
            name: Name for this piece
            content: The content (strokes, notes, etc.)
            category: Category (art, music, writing, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d")

        # Wrap content with metadata
        piece = {
            'name': name,
            'category': category,
            'created': timestamp,
            'content': content
        }

        filename = f"{category}/{name}_{timestamp}.json"
        return self.save(f"portfolio/{filename}", piece, as_json=True)

    def portfolio_load(self, name: str, category: str = "art") -> Optional[Dict]:
        """Load a portfolio piece."""
        category_dir = self.root / "portfolio" / category
        if not category_dir.exists():
            return None

        # Find piece by name
        matching = list(category_dir.glob(f"{name}_*.json"))
        if matching:
            return self.load(str(matching[0].relative_to(self.root)), as_json=True)
        return None

    def portfolio_list(self, category: str = None) -> List[Dict]:
        """List portfolio pieces."""
        portfolio_dir = self.root / "portfolio"
        if not portfolio_dir.exists():
            return []

        pieces = []

        if category:
            cat_dir = portfolio_dir / category
            if cat_dir.exists():
                for f in cat_dir.glob("*.json"):
                    try:
                        with open(f) as fp:
                            piece = json.load(fp)
                            pieces.append({'file': f.name, **piece})
                    except:
                        pass
        else:
            for f in portfolio_dir.rglob("*.json"):
                try:
                    with open(f) as fp:
                        piece = json.load(fp)
                        pieces.append({'file': str(f.relative_to(portfolio_dir)), **piece})
                except:
                    pass

        return pieces

    # =========================================================================
    # JOURNAL - Reflections and growth tracking
    # =========================================================================

    def journal_entry(self, content: str, tags: List[str] = None) -> bool:
        """
        Write a journal entry.

        Args:
            content: The journal entry text
            tags: Optional tags (e.g., ["question", "proud", "confused"])
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        entry = {
            'timestamp': timestamp,
            'content': content,
            'tags': tags or []
        }

        return self.save(f"journal/{timestamp}.json", entry, as_json=True)

    def journal_read_recent(self, n: int = 5) -> List[Dict]:
        """Read the n most recent journal entries."""
        journal_dir = self.root / "journal"
        if not journal_dir.exists():
            return []

        entries = []
        for f in sorted(journal_dir.glob("*.json"), reverse=True)[:n]:
            try:
                with open(f) as fp:
                    entries.append(json.load(fp))
            except:
                pass

        return entries

    def journal_search(self, tag: str) -> List[Dict]:
        """Search journal entries by tag."""
        journal_dir = self.root / "journal"
        if not journal_dir.exists():
            return []

        matches = []
        for f in journal_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    entry = json.load(fp)
                    if tag in entry.get('tags', []):
                        matches.append(entry)
            except:
                pass

        return matches

    # =========================================================================
    # SUMMARY - What do I have?
    # =========================================================================

    def summary(self) -> Dict:
        """Get a summary of what this student has saved."""
        return {
            'student': self.student_name,
            'scratch_files': len(self.list_dir('scratch')),
            'notes': len(self.note_list()),
            'portfolio_pieces': len(self.portfolio_list()),
            'journal_entries': len(self.list_dir('journal')),
            'root_path': str(self.root)
        }


# =============================================================================
# SHARED CLASS LIBRARY
# =============================================================================

class ClassLibrary:
    """
    Shared resources for the whole class.

    Contains:
    - Example art/music pieces
    - Shared stories and reading materials
    - Class achievements and milestones
    """

    def __init__(self, base_path: str = "data/shared_library"):
        self.root = Path(base_path)
        for subdir in ['art', 'music', 'stories', 'achievements']:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def share(self, content: Any, name: str, category: str, contributor: str) -> bool:
        """Share a piece with the class."""
        timestamp = datetime.now().strftime("%Y%m%d")

        piece = {
            'name': name,
            'category': category,
            'contributor': contributor,
            'shared_date': timestamp,
            'content': content
        }

        filepath = self.root / category / f"{name}_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(piece, f, indent=2)

        return True

    def browse(self, category: str = None) -> List[Dict]:
        """Browse shared pieces."""
        pieces = []

        if category:
            cat_dir = self.root / category
            if cat_dir.exists():
                for f in cat_dir.glob("*.json"):
                    with open(f) as fp:
                        pieces.append(json.load(fp))
        else:
            for f in self.root.rglob("*.json"):
                with open(f) as fp:
                    pieces.append(json.load(fp))

        return pieces


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=== Student Filesystem Demo ===\n")

    # Create filesystem for Nova
    nova = StudentFileSystem("nova", base_path="/tmp/coherence_test")

    print(f"1. SCRATCH PAD")
    nova.scratch_write("working_problem", {"step": 1, "values": [1, 2, 3]})
    print(f"   Wrote scratch: working_problem")
    print(f"   Read back: {nova.scratch_read('working_problem')}")

    print(f"\n2. NOTES")
    nova.note_write("counting", "Today I learned to count: 0, 1, 2, 3, 4...", section="math")
    print(f"   Wrote note: counting (math)")
    print(f"   Notes list: {nova.note_list('math')}")

    print(f"\n3. PORTFOLIO")
    nova.portfolio_save("my_first_circle", {"strokes": [[0,0], [1,0], [1,1], [0,1]]}, category="art")
    print(f"   Saved to portfolio: my_first_circle")
    print(f"   Portfolio pieces: {len(nova.portfolio_list())} items")

    print(f"\n4. JOURNAL")
    nova.journal_entry("Today I finally understood counting! It goes up by 1 each time.", tags=["proud", "math"])
    print(f"   Wrote journal entry")
    print(f"   Recent entries: {len(nova.journal_read_recent())} entries")

    print(f"\n5. SUMMARY")
    print(f"   {nova.summary()}")

    # Clean up scratch
    print(f"\n6. CLEAR SCRATCH")
    count = nova.scratch_clear()
    print(f"   Cleared {count} scratch files")

    print("\n=== Demo Complete ===")
