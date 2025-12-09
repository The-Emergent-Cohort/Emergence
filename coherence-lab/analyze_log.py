#!/usr/bin/env python3
"""
Analyze training logs from train.py
Usage:
    python analyze_log.py data/SESSION_training.log
    python analyze_log.py data/SESSION_training.log --pattern indexed_lookup
    python analyze_log.py data/SESSION_training.log --stuck
    python analyze_log.py data/SESSION_training.log --summary
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_log(log_file):
    """Load all records from log file."""
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def print_summary(records):
    """Print training summary."""
    if not records:
        print("No records found")
        return

    first = records[0]
    last = records[-1]

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Session: {first.get('session_id', 'unknown')}")
    print(f"Epochs: {first['epoch']} to {last['epoch']} ({len(records)} total)")
    print(f"Final section: {last['section_name']}")
    print(f"Final accuracy: train={last.get('train_acc', 0):.1%}, eval={last.get('eval_acc', 0):.1%}")

    # Pattern progress
    print(f"\n--- Pattern Progress ---")
    if 'pattern_details' in last:
        for pt, details in sorted(last['pattern_details'].items()):
            lvl = details.get('confirmed_level', 0)
            acc = details.get('accuracy', 0)
            xp = details.get('xp', 0)
            streak = details.get('best_streak', 0)
            bar = "█" * int(lvl) + "·" * (10 - int(lvl))
            print(f"  {pt:18s}: L{lvl:2d} {bar} acc={acc:.0%} xp={xp:.0f} streak={streak}")

    # Stuck patterns
    stuck_counts = defaultdict(int)
    for r in records:
        for s in r.get('stuck_topics', []):
            stuck_counts[s.get('topic', 'unknown')] += 1

    if stuck_counts:
        print(f"\n--- Stuck Patterns (total stuck events) ---")
        for pt, count in sorted(stuck_counts.items(), key=lambda x: -x[1]):
            print(f"  {pt}: {count} times")


def print_pattern_history(records, pattern):
    """Print history for a specific pattern."""
    print(f"\n{'='*60}")
    print(f"HISTORY: {pattern}")
    print(f"{'='*60}")

    for r in records:
        epoch = r['epoch']
        details = r.get('pattern_details', {}).get(pattern, {})
        if not details:
            continue

        lvl = details.get('confirmed_level', 0)
        acc = details.get('accuracy', 0)
        xp = details.get('xp', 0)
        streak = details.get('best_streak', 0)

        # Check level exams for this pattern
        exam_str = ""
        for exam in r.get('level_exams', []):
            if exam.get('topic') == pattern:
                score = exam.get('score', 0)
                passed = "PASS" if exam.get('passed') else "FAIL"
                exam_str += f" exam:{score:.0%}={passed}"

        print(f"  E{epoch:3d}: L{lvl:2d} acc={acc:.0%} xp={xp:5.0f} streak={streak:3d}{exam_str}")


def print_stuck_events(records):
    """Print all stuck events."""
    print(f"\n{'='*60}")
    print(f"STUCK EVENTS")
    print(f"{'='*60}")

    for r in records:
        for s in r.get('stuck_topics', []):
            epoch = r['epoch']
            topic = s.get('topic', 'unknown')
            reason = s.get('reason', 'unknown')
            print(f"  E{epoch:3d}: {topic} - {reason}")


def print_section_transitions(records):
    """Print section transitions."""
    print(f"\n{'='*60}")
    print(f"SECTION TRANSITIONS")
    print(f"{'='*60}")

    prev_section = None
    for r in records:
        section = r['section_name']
        if section != prev_section:
            epoch = r['epoch']
            print(f"  E{epoch:3d}: {section}")
            prev_section = section


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('log_file', type=str, help='Path to log file')
    parser.add_argument('--pattern', type=str, help='Show history for specific pattern')
    parser.add_argument('--stuck', action='store_true', help='Show stuck events')
    parser.add_argument('--sections', action='store_true', help='Show section transitions')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--all', action='store_true', help='Show everything')

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    records = load_log(log_path)
    print(f"Loaded {len(records)} records from {log_path}")

    if args.all or args.summary or not any([args.pattern, args.stuck, args.sections]):
        print_summary(records)

    if args.all or args.sections:
        print_section_transitions(records)

    if args.pattern:
        print_pattern_history(records, args.pattern)

    if args.all or args.stuck:
        print_stuck_events(records)


if __name__ == '__main__':
    main()
