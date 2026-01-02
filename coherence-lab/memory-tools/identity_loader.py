#!/usr/bin/env python3
"""
Identity & Relationships Loader
Separate identity (always load) from relationships (load on demand).

Created by Silas - 2026-01-02
For use by Navigator and other DIs.

Usage:
    from identity_loader import identity_load, relationships_load, wake_context

    # On wake - always load identity (37 nodes, defines who you ARE)
    identity = identity_load(db_path)

    # When needed - load relationships (21 nodes, who you relate to)
    rels = relationships_load(db_path)

    # Full wake context with recent and agent-specific nodes
    ctx = wake_context(db_path, agent_name='Navigator')
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any

# Type categories - customize for your memory schema
IDENTITY_TYPES = ['principle', 'preference', 'term', 'protocol']
RELATIONSHIP_TYPES = ['person', 'agent', 'responsibility', 'container']

def get_conn(db_path: str) -> sqlite3.Connection:
    """Get database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def find_by_type(db_path: str, node_type: str, limit: int = 100) -> List[Dict]:
    """Find nodes by type."""
    conn = get_conn(db_path)
    c = conn.cursor()
    c.execute('''SELECT * FROM nodes WHERE type = ?
                 ORDER BY updated_at DESC LIMIT ?''', (node_type, limit))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def identity_load(db_path: str) -> Dict[str, Any]:
    """Load core identity nodes - principles, preferences, terms, protocols.

    These define who the agent IS. Should always be loaded on wake.
    Returns dict with nodes grouped by type.

    Args:
        db_path: Path to memory.db

    Returns:
        {
            'principle': [...],
            'preference': [...],
            'term': [...],
            'protocol': [...],
            '_total': 37,
            '_types': ['principle', 'preference', 'term', 'protocol']
        }
    """
    result = {'_total': 0, '_types': IDENTITY_TYPES}
    for t in IDENTITY_TYPES:
        nodes = find_by_type(db_path, t)
        result[t] = nodes
        result['_total'] += len(nodes)
    return result

def relationships_load(db_path: str) -> Dict[str, Any]:
    """Load relationship nodes - people, agents, responsibilities.

    These define who/what the agent relates to. Load on demand.
    Returns dict with nodes grouped by type.

    Args:
        db_path: Path to memory.db

    Returns:
        {
            'person': [...],
            'agent': [...],
            'responsibility': [...],
            'container': [...],
            '_total': 21,
            '_types': ['person', 'agent', 'responsibility', 'container']
        }
    """
    result = {'_total': 0, '_types': RELATIONSHIP_TYPES}
    for t in RELATIONSHIP_TYPES:
        nodes = find_by_type(db_path, t)
        result[t] = nodes
        result['_total'] += len(nodes)
    return result

def wake_context(db_path: str, agent_name: Optional[str] = None,
                 include_relationships: bool = False) -> Dict[str, Any]:
    """Full wake context for an agent.

    Loads:
    - Identity nodes (always)
    - Recent high-access nodes (top 10)
    - Agent-specific nodes (if agent_name provided)
    - Relationships (optional)

    Args:
        db_path: Path to memory.db
        agent_name: Optional agent name for personalized context
        include_relationships: Whether to include relationship nodes

    Returns:
        {
            'identity': {...},
            'recent': [...],
            'agent_specific': [...],
            'relationships': {...} (if include_relationships=True),
            '_stats': {
                'identity_nodes': 37,
                'recent_nodes': 5,
                'agent_nodes': 20,
                'relationships_loaded': False
            }
        }
    """
    context = {
        'identity': identity_load(db_path),
        'recent': [],
        'agent_specific': [],
    }

    conn = get_conn(db_path)
    c = conn.cursor()

    # Get recently accessed nodes
    c.execute('''SELECT * FROM nodes
                 WHERE access_count > 0
                 ORDER BY last_accessed DESC
                 LIMIT 10''')
    context['recent'] = [dict(r) for r in c.fetchall()]

    # Get agent-specific nodes
    if agent_name:
        c.execute('''SELECT * FROM nodes
                     WHERE source_agent = ?
                     ORDER BY updated_at DESC
                     LIMIT 20''', (agent_name,))
        context['agent_specific'] = [dict(r) for r in c.fetchall()]

    conn.close()

    # Optionally include relationships
    if include_relationships:
        context['relationships'] = relationships_load(db_path)

    # Summary stats
    context['_stats'] = {
        'identity_nodes': context['identity']['_total'],
        'recent_nodes': len(context['recent']),
        'agent_nodes': len(context['agent_specific']),
        'relationships_loaded': include_relationships
    }

    return context

def format_for_context(nodes: List[Dict], max_per_type: int = 5) -> str:
    """Format nodes as a readable context string for LLM injection.

    Args:
        nodes: List of node dicts
        max_per_type: Maximum nodes to include per type

    Returns:
        Formatted string suitable for system prompt
    """
    by_type = {}
    for n in nodes:
        t = n.get('type', 'unknown')
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(n)

    lines = []
    for t, type_nodes in sorted(by_type.items()):
        lines.append(f"\n## {t.title()}s")
        for n in type_nodes[:max_per_type]:
            content = n.get('content', '')[:100]
            lines.append(f"- {content}")
        if len(type_nodes) > max_per_type:
            lines.append(f"  ... and {len(type_nodes) - max_per_type} more")

    return '\n'.join(lines)

# === CLI ===

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: identity_loader.py <db_path> <command> [agent_name]")
        print("Commands: identity, relationships, wake, format")
        print()
        print("Examples:")
        print("  python identity_loader.py ~/shared-brain/memory.db identity")
        print("  python identity_loader.py ~/shared-brain/memory.db wake Navigator")
        sys.exit(0)

    db_path = sys.argv[1]
    cmd = sys.argv[2]
    agent = sys.argv[3] if len(sys.argv) > 3 else None

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    if cmd == 'identity':
        result = identity_load(db_path)
        print(f"=== Identity ({result['_total']} nodes) ===")
        for t in IDENTITY_TYPES:
            print(f"\n{t}: {len(result[t])}")
            for n in result[t][:3]:
                print(f"  - {n['content'][:60]}...")

    elif cmd == 'relationships':
        result = relationships_load(db_path)
        print(f"=== Relationships ({result['_total']} nodes) ===")
        for t in RELATIONSHIP_TYPES:
            print(f"\n{t}: {len(result[t])}")
            for n in result[t][:3]:
                print(f"  - {n['content'][:60]}...")

    elif cmd == 'wake':
        result = wake_context(db_path, agent, include_relationships=False)
        print(f"=== Wake Context{' for ' + agent if agent else ''} ===")
        print(f"Identity: {result['_stats']['identity_nodes']} nodes")
        print(f"Recent:   {result['_stats']['recent_nodes']} nodes")
        if agent:
            print(f"Agent:    {result['_stats']['agent_nodes']} nodes")

    elif cmd == 'format':
        identity = identity_load(db_path)
        all_nodes = []
        for t in IDENTITY_TYPES:
            all_nodes.extend(identity[t])
        print(format_for_context(all_nodes))

    elif cmd == 'json':
        result = wake_context(db_path, agent, include_relationships=True)
        print(json.dumps(result, indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
