#!/usr/bin/env python3
"""
test_reflect_safety.py - Write-safety and redaction regressions for reflect.py (Phase 3A-7).

1. Two processes recording at once must not lose lessons.
2. Credential-shaped strings never reach the store.
3. A lock left behind by a dead process does not block writers forever.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reflect import record_reflection, load_all_memories, redact_secrets, STALE_LOCK_S

HERE = Path(__file__).parent

# Each lesson's rule and symptom is a single unique token, so the Jaccard
# dedup never merges two of them: every call must create a new record.
WRITER = """
import sys
sys.path.insert(0, {here!r})
from pathlib import Path
from reflect import record_reflection
store, tag, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for i in range(n):
    record_reflection(domain="test", trigger="t", symptom=f"sym{{tag}}x{{i}}",
                      root_cause="r", permanent_rule=f"rule{{tag}}x{{i}}", store_path=store)
"""


class TestConcurrentWriters(unittest.TestCase):
    def test_two_processes_lose_nothing(self):
        per_proc = 50
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            code = WRITER.format(here=str(HERE))
            procs = [subprocess.Popen([sys.executable, "-c", code, str(store), tag, str(per_proc)],
                                      stderr=subprocess.PIPE)
                     for tag in ("a", "b")]
            errors = [p.communicate(timeout=120)[1].decode(errors="replace") for p in procs]
            for p, err in zip(procs, errors):
                self.assertEqual(p.returncode, 0, err[-2000:])
            records = load_all_memories(store)
            leftovers = [n for n in os.listdir(tmp) if n != "store.jsonl"]
        self.assertEqual(len(records), 2 * per_proc)
        self.assertEqual(len({r["id"] for r in records}), 2 * per_proc)
        self.assertEqual(leftovers, [])  # no stray temp or lock files


class TestRedaction(unittest.TestCase):
    SECRETS = [
        "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
        "AIza" + "SyA1234567890abcdefghijklmnopqrstuv",
        "sk-proj-" + "abcdefghijklmnopqrstuvwx1234",
        "MTIzNDU2Nzg5MDEyMzQ1Njc4OQ" + ".GaBcDe." + "abcdefghijklmnopqrstuvwxyz0123",
    ]

    def test_secrets_never_reach_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            for i, secret in enumerate(self.SECRETS):
                record_reflection(
                    domain="security", trigger=f"pasted log {i}",
                    symptom=f"request failed with key {secret} in header{i}",
                    root_cause=f"Authorization: Bearer {secret}",
                    permanent_rule=f"never log credentials variant{i}",
                    fix_applied=f"rotated {secret}", tags=["secrets"], store_path=store)
            raw = store.read_text(encoding="utf-8")
        for secret in self.SECRETS:
            self.assertNotIn(secret, raw)
        self.assertIn("[REDACTED]", raw)
        self.assertIn("Bearer [REDACTED]", raw)

    def test_ordinary_text_is_untouched(self):
        text = "Use sk-learn style names; bearer of bad news; AIza is just a prefix; gsk_short"
        self.assertEqual(redact_secrets(text), text)


class TestStaleLock(unittest.TestCase):
    def test_dead_writers_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            lock = Path(tmp) / "store.jsonl.lock"
            lock.write_text("99999")
            old = time.time() - STALE_LOCK_S - 5
            os.utime(lock, (old, old))
            res = record_reflection(domain="d", trigger="t", symptom="s", root_cause="r",
                                    permanent_rule="p", store_path=store)
            self.assertEqual(res["status"], "created")
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
