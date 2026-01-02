#!/usr/bin/env python3
"""
Import VerbNet verb classes into primitives.db.

VerbNet provides semantic classification of English verbs with:
- Verb classes (hierarchical groupings)
- Thematic roles (Agent, Patient, Theme, etc.)
- Syntactic frames

Run from: /usr/share/databases/scripts/
Requires: init_schemas.py to have been run first
"""

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

# Path configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DB_DIR = BASE_DIR / "db"
REF_DIR = BASE_DIR / "reference"
PRIMITIVES_DB = DB_DIR / "primitives.db"

# VerbNet uses domain 1 (physical) category 1 (motion) as base
# but verb classes span multiple domains
VERBNET_DOMAIN = 1  # Physical as default
VERBNET_CATEGORY = 1  # Motion as default

# Domain mapping based on VerbNet class names
DOMAIN_HINTS = {
    "say": 4,           # social/communication
    "tell": 4,
    "talk": 4,
    "speak": 4,
    "ask": 4,
    "give": 4,          # social/exchange
    "get": 4,
    "send": 4,
    "put": 1,           # physical/motion
    "run": 1,
    "hit": 1,           # physical/contact
    "touch": 1,
    "break": 1,         # physical/change
    "cut": 1,
    "see": 3,           # mental/perception
    "look": 3,
    "hear": 3,
    "feel": 3,
    "think": 3,         # mental/cognition
    "know": 3,
    "believe": 3,
    "want": 3,          # mental/volition
    "try": 3,
    "love": 3,          # mental/emotion
    "hate": 3,
    "fear": 3,
}


def infer_domain(class_name: str) -> int:
    """Infer semantic domain from class name."""
    name_lower = class_name.lower()
    for hint, domain in DOMAIN_HINTS.items():
        if hint in name_lower:
            return domain
    return VERBNET_DOMAIN


def parse_verbnet_xml(xml_path: Path) -> dict:
    """Parse a VerbNet XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_id = root.get("ID", "")
    class_name = class_id.split("-")[0] if "-" in class_id else class_id

    # Get members (verbs in this class)
    members = []
    for member in root.findall(".//MEMBER"):
        name = member.get("name", "")
        if name:
            members.append(name)

    # Get thematic roles
    roles = []
    for role in root.findall(".//THEMROLE"):
        role_type = role.get("type", "")
        if role_type:
            roles.append(role_type)

    # Get frames (simplified)
    frames = []
    for frame in root.findall(".//FRAME"):
        desc = frame.find("DESCRIPTION")
        if desc is not None:
            primary = desc.get("primary", "")
            if primary:
                frames.append(primary)

    # Get subclasses
    subclasses = []
    for subclass in root.findall("./SUBCLASSES/VNSUBCLASS"):
        sub_id = subclass.get("ID", "")
        if sub_id:
            subclasses.append(sub_id)

    return {
        "class_id": class_id,
        "class_name": class_name,
        "members": members,
        "roles": roles,
        "frames": frames,
        "subclasses": subclasses,
    }


def main():
    print("=" * 60)
    print("Importing VerbNet Verb Classes")
    print("=" * 60)

    if not PRIMITIVES_DB.exists():
        print(f"ERROR: {PRIMITIVES_DB} not found.")
        print("Run init_schemas.py first.")
        return 1

    # Find VerbNet data
    verbnet_dir = REF_DIR / "verbnet"
    if not verbnet_dir.exists():
        print(f"ERROR: {verbnet_dir} not found.")
        print("Run unpack_tarballs.py first or download VerbNet.")
        return 1

    # Find XML files
    xml_files = list(verbnet_dir.rglob("*.xml"))
    if not xml_files:
        print(f"ERROR: No XML files found in {verbnet_dir}")
        return 1

    print(f"\nFound {len(xml_files)} VerbNet class files")

    conn = sqlite3.connect(PRIMITIVES_DB)
    cursor = conn.cursor()

    # Get max primitive_id
    cursor.execute("SELECT MAX(primitive_id) FROM primitives")
    result = cursor.fetchone()
    next_id = (result[0] or 0) + 1

    # Track what we've added
    inserted = 0
    members_added = 0

    for xml_path in sorted(xml_files):
        try:
            data = parse_verbnet_xml(xml_path)
        except Exception as e:
            print(f"  WARNING: Failed to parse {xml_path.name}: {e}")
            continue

        class_id = data["class_id"]
        class_name = data["class_name"]

        if not class_id:
            continue

        # Build description
        desc_parts = []
        if data["roles"]:
            desc_parts.append(f"Roles: {', '.join(data['roles'])}")
        if data["frames"]:
            desc_parts.append(f"Frames: {', '.join(data['frames'][:3])}")
        description = "; ".join(desc_parts) if desc_parts else None

        # Examples are the member verbs
        examples = ", ".join(data["members"][:10]) if data["members"] else None

        domain = infer_domain(class_name)

        # Insert the verb class as a primitive
        cursor.execute("""
            INSERT OR IGNORE INTO primitives
            (primitive_id, canonical_name, source, domain, category, description, examples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (next_id, class_id, "verbnet", domain, VERBNET_CATEGORY, description, examples))

        if cursor.rowcount > 0:
            # Add member verbs as primitive_forms (English surface forms)
            for member in data["members"]:
                cursor.execute("""
                    INSERT OR IGNORE INTO primitive_forms
                    (primitive_id, lang_genomic, surface_form, source)
                    VALUES (?, ?, ?, ?)
                """, (next_id, "1.8.127.0", member, "verbnet"))  # English = 1.8.127.0
                members_added += 1

            inserted += 1
            next_id += 1

    conn.commit()

    print("\n" + "=" * 60)
    print("VerbNet import complete!")
    print("=" * 60)
    print(f"\nVerb classes added: {inserted}")
    print(f"Member verbs linked: {members_added}")
    print(f"\nDatabase: {PRIMITIVES_DB}")

    # Show sample
    cursor.execute("""
        SELECT canonical_name, examples
        FROM primitives
        WHERE source = 'verbnet'
        LIMIT 5
    """)
    print("\nSample classes:")
    for row in cursor.fetchall():
        examples = row[1][:50] + "..." if row[1] and len(row[1]) > 50 else row[1]
        print(f"  {row[0]}: {examples}")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
