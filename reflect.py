#!/usr/bin/env python3
"""
reflect.py - Post-Mortem Reflection Engine for Self-Improving Developer Agent

Records new lessons learned from bugs, regressions, or user feedback into the
episodic memory store. Automatically deduplicates similar lessons by incrementing
their frequency counter, ensuring high-impact patterns bubble up without noise.
Pure Python standard library.
"""

import os
import re
import sys
import json
import time
import argparse
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_STORE = Path(__file__).parent / "store" / "experience_store.jsonl"

# Credential shapes that must never reach the store (it is read back into agent
# prompts and committed to git). Replaced before dedup and before writing.
SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                                    # Groq
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),                                  # Google API key
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"),                   # OpenAI / Anthropic
    re.compile(r"[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{27,}"),  # Discord bot token
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),                     # Authorization header
]
REDACTED = "[REDACTED]"

LOCK_TIMEOUT_S = 10.0

if os.name == "nt":
    import msvcrt

    def _try_lock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # OSError if held

    def _unlock(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # OSError if held

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def redact_secrets(text: str) -> str:
    """Replace credential-shaped substrings with [REDACTED]."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, text)
    return text


@contextmanager
def store_lock(store_path: Path):
    """
    Cross-process mutex around a read-modify-write of the store. Without it,
    two reflect.py runs both read N records and both write N+1, and one lesson
    is lost.

    An OS file lock (msvcrt / fcntl) on store.jsonl.lock, held for the whole
    read-modify-write including the final os.replace. The kernel releases it
    if the holder dies, so there is no stale lock to reclaim, and a slow but
    live writer never loses it: other writers wait, then raise TimeoutError.
    The lock file itself stays on disk (it is gitignored); deleting it would
    let two writers lock two different files.
    """
    lock_path = store_path.with_name(store_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                _try_lock(f)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"store is locked by another writer: {lock_path}")
                time.sleep(0.02)
        try:
            yield
        finally:
            _unlock(f)
    finally:
        f.close()


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
    # A unique temp name per writer: a shared ".tmp" let two writers clobber
    # each other's half-written file.
    fd, temp_name = tempfile.mkstemp(dir=store_path.parent, prefix=store_path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # On Windows the rename fails while a reader (recall.py) has the store
        # open; that window is milliseconds, so retry briefly.
        for attempt in range(50):
            try:
                os.replace(temp_name, store_path)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.02)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


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
    Credential-shaped strings are redacted first; the whole update holds store_lock.
    """
    domain, trigger, symptom, root_cause, permanent_rule, fix_applied = (
        redact_secrets(s) for s in (domain, trigger, symptom, root_cause, permanent_rule, fix_applied)
    )
    tags = [redact_secrets(t) for t in (tags or [])]
    with store_lock(store_path):
        return _record_locked(domain, trigger, symptom, root_cause, permanent_rule,
                              fix_applied, tags, severity, store_path)


def _record_locked(domain, trigger, symptom, root_cause, permanent_rule,
                   fix_applied, tags, severity, store_path) -> Dict[str, Any]:
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
