#!/usr/bin/env python3
"""
Unpack data tarballs to reference directory.

Handles: VerbNet, WordNet, OMW, and other compressed sources.

Run from: /usr/share/databases/scripts/
Expects tarballs in: scripts/data_tarballs/
Unpacks to: reference/
"""

import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
TARBALL_DIR = SCRIPT_DIR / "data_tarballs"
REF_DIR = BASE_DIR / "reference"

# Known tarball patterns and their target directories
KNOWN_SOURCES = {
    "verbnet": ["verbnet*.tar.gz", "verbnet*.tgz", "verbnet*.zip"],
    "wordnet": ["wordnet*.tar.gz", "wn*.tar.gz", "wordnet*.zip", "WNdb*.gz"],
    "omw": ["omw*.tar.gz", "omw*.zip", "open-multilingual*.tar.gz"],
    "framenet": ["framenet*.tar.gz", "fndata*.zip"],
    "propbank": ["propbank*.tar.gz", "propbank*.zip"],
}


def unpack_file(src: Path, dest_dir: Path) -> bool:
    """Unpack a single archive file."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if src.suffix == ".zip":
            print(f"  Unpacking ZIP: {src.name}")
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(dest_dir)
            return True

        elif src.name.endswith(".tar.gz") or src.name.endswith(".tgz"):
            print(f"  Unpacking TAR.GZ: {src.name}")
            with tarfile.open(src, "r:gz") as tf:
                tf.extractall(dest_dir)
            return True

        elif src.name.endswith(".tar.bz2"):
            print(f"  Unpacking TAR.BZ2: {src.name}")
            with tarfile.open(src, "r:bz2") as tf:
                tf.extractall(dest_dir)
            return True

        elif src.name.endswith(".tar"):
            print(f"  Unpacking TAR: {src.name}")
            with tarfile.open(src, "r:") as tf:
                tf.extractall(dest_dir)
            return True

        elif src.suffix == ".gz" and not src.name.endswith(".tar.gz"):
            # Single gzipped file
            print(f"  Unpacking GZ: {src.name}")
            dest_file = dest_dir / src.stem
            with gzip.open(src, "rb") as f_in:
                with open(dest_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True

        else:
            print(f"  Unknown format: {src.name}")
            return False

    except Exception as e:
        print(f"  ERROR unpacking {src.name}: {e}")
        return False


def find_and_unpack(source_name: str, patterns: list) -> int:
    """Find and unpack files matching patterns for a source."""
    dest_dir = REF_DIR / source_name
    unpacked = 0

    for pattern in patterns:
        for src in TARBALL_DIR.glob(pattern):
            if unpack_file(src, dest_dir):
                unpacked += 1

    return unpacked


def main():
    print("=" * 60)
    print("Unpacking Data Tarballs")
    print("=" * 60)

    if not TARBALL_DIR.exists():
        print(f"\nNo tarball directory found: {TARBALL_DIR}")
        print("Create it and add your data archives.")
        return 0

    # List what we have
    archives = list(TARBALL_DIR.glob("*"))
    if not archives:
        print(f"\nNo archives found in {TARBALL_DIR}")
        return 0

    print(f"\nFound {len(archives)} files in {TARBALL_DIR}:")
    for a in sorted(archives):
        print(f"  {a.name}")

    print(f"\nUnpacking to: {REF_DIR}")
    REF_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for source_name, patterns in KNOWN_SOURCES.items():
        count = find_and_unpack(source_name, patterns)
        if count > 0:
            print(f"\n{source_name}: unpacked {count} archive(s)")
            total += count

    # Handle any unrecognized archives
    print("\nChecking for other archives...")
    handled = set()
    for source_name, patterns in KNOWN_SOURCES.items():
        for pattern in patterns:
            for match in TARBALL_DIR.glob(pattern):
                handled.add(match.name)

    for archive in archives:
        if archive.name not in handled and archive.is_file():
            # Try to unpack to a directory named after the file
            dest_name = archive.stem
            if dest_name.endswith(".tar"):
                dest_name = dest_name[:-4]
            dest_dir = REF_DIR / dest_name
            if unpack_file(archive, dest_dir):
                total += 1

    print("\n" + "=" * 60)
    print(f"Unpacking complete! Total: {total} archive(s)")
    print("=" * 60)

    # Show what's in reference now
    print(f"\nReference directory contents:")
    for item in sorted(REF_DIR.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/  ({count} files)")

    return 0


if __name__ == "__main__":
    exit(main())
