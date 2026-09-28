#!/usr/bin/env python3
"""
reflect.py - Post-Mortem Reflection Engine for Self-Improving Developer Agent

Records new lessons learned from bugs, regressions, or user feedback into the
episodic memory store. Automatically deduplicates similar lessons by incrementing
their frequency counter, ensuring high-impact patterns bubble up without noise.
Pure Python standard library.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_STORE = Path(__file__).parent / "store" / "experience_store.jsonl"


def get_current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_all_memories(store_path: Path = DEFAULT_STORE) -> List[Dict[str, Any]]:
    """Read all entries from experience store."""
    if not store_path.exists():
        return []
    records = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_all_memories(records: List[Dict[str, Any]], store_path: Path = DEFAULT_STORE) -> None:
    """Atomic write of all entries back to experience store."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = store_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    temp_path.replace(store_path)


def compute_token_overlap(str1: str, str2: str) -> float:
    """Compute Jaccard similarity between two string token sets."""
    toks1 = set(str1.lower().split())
    toks2 = set(str2.lower().split())
    if not toks1 or not toks2:
        return 0.0
    intersection = toks1.intersection(toks2)
    union = toks1.union(toks2)
    return len(intersection) / len(union)


def record_reflection(
    domain: str,
    trigger: str,
    symptom: str,
    root_cause: str,
    permanent_rule: str,
    fix_applied: str = "",
    tags: Optional[List[str]] = None,
    severity: str = "medium",
    store_path: Path = DEFAULT_STORE
) -> Dict[str, Any]:
    """
    Append or update an episodic memory. If an existing memory shares high similarity
    in permanent_rule or symptom, increment its frequency count instead of duplicating.
    """
    records = load_all_memories(store_path)
    now = get_current_iso_timestamp()
    clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]

    # Check for near duplicates
    matched_idx = -1
    for idx, r in enumerate(records):
        rule_sim = compute_token_overlap(r.get("permanent_rule", ""), permanent_rule)
        symptom_sim = compute_token_overlap(r.get("symptom", ""), symptom)
        if rule_sim > 0.65 or symptom_sim > 0.75:
            matched_idx = idx
            break

    if matched_idx >= 0:
        existing = records[matched_idx]
        existing["frequency"] = existing.get("frequency", 1) + 1
        existing["updated_at"] = now
        # Merge tags
        combined_tags = list(set(existing.get("tags", []) + clean_tags))
        existing["tags"] = combined_tags
        # Upgrade severity if higher
        sev_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if sev_rank.get(severity, 2) > sev_rank.get(existing.get("severity", "medium"), 2):
            existing["severity"] = severity
        save_all_memories(records, store_path)
        return {
            "status": "updated_frequency",
            "id": existing.get("id"),
            "frequency": existing["frequency"],
            "memory": existing
        }

    # Generate new ID: MEM-001, MEM-002, etc.
    highest_id_num = 0
    for r in records:
        mid = r.get("id", "")
        if mid.startswith("MEM-"):
            try:
                num = int(mid.split("-")[1])
                highest_id_num = max(highest_id_num, num)
            except (IndexError, ValueError):
                pass

    new_id = f"MEM-{(highest_id_num + 1):03d}"
    new_record = {
        "id": new_id,
        "created_at": now,
        "updated_at": now,
        "domain": domain.strip().lower(),
        "trigger": trigger.strip(),
        "symptom": symptom.strip(),
        "root_cause": root_cause.strip(),
        "fix_applied": fix_applied.strip(),
        "permanent_rule": permanent_rule.strip(),
        "tags": clean_tags,
        "severity": severity.strip().lower(),
        "frequency": 1
    }

    records.append(new_record)
    save_all_memories(records, store_path)
    return {
        "status": "created",
        "id": new_id,
        "frequency": 1,
        "memory": new_record
    }


def main():
    parser = argparse.ArgumentParser(description="Log a new post-mortem reflection into episodic memory.")
    parser.add_argument("--domain", "-d", required=True, help="Engineering domain (e.g., 'css-layout', 'routing', 'copywriting')")
    parser.add_argument("--trigger", "-t", required=True, help="Circumstance or task scenario when this occurs")
    parser.add_argument("--symptom", "-s", required=True, help="What broke or was flagged by user/tests")
    parser.add_argument("--root-cause", "-r", required=True, help="Underlying technical reason for failure")
    parser.add_argument("--rule", "-u", required=True, help="Universal permanent heuristic to prevent recurrences")
    parser.add_argument("--fix", "-f", default="", help="Specific code fix applied this time")
    parser.add_argument("--tags", "-g", help="Comma-separated tags (e.g., 'css,layout,flexbox')")
    parser.add_argument("--severity", "-v", choices=["low", "medium", "high", "critical"], default="medium")
    parser.add_argument("--store", help="Custom path to experience_store.jsonl")

    args = parser.parse_args()
    store_path = Path(args.store) if args.store else DEFAULT_STORE
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    res = record_reflection(
        domain=args.domain,
        trigger=args.trigger,
        symptom=args.symptom,
        root_cause=args.root_cause,
        permanent_rule=args.rule,
        fix_applied=args.fix,
        tags=tag_list,
        severity=args.severity,
        store_path=store_path
    )

    try:
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if res["status"] == "created":
        print(f"[NEW] Successfully recorded new lesson [{res['id']}]: '{res['memory']['permanent_rule']}'")
    else:
        print(f"[UPDATE] Reinforced existing lesson [{res['id']}] (frequency incremented to {res['frequency']})")


if __name__ == "__main__":
    main()
