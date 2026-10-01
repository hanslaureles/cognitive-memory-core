#!/usr/bin/env python3
"""
test_reflect_safety.py - Write-safety and redaction regressions for reflect.py (Phase 3A-7).

1. Two processes recording at once must not lose lessons.
2. Credential-shaped strings never reach the store.
3. A writer that died holding the lock does not block the next one.
4. A slow but live writer keeps the lock, however long it holds it.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import reflect
from reflect import record_reflection, load_all_memories, redact_secrets, store_lock

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
            leftovers = [n for n in os.listdir(tmp) if n not in ("store.jsonl", "store.jsonl.lock")]
        self.assertEqual(len(records), 2 * per_proc)
        self.assertEqual(len({r["id"] for r in records}), 2 * per_proc)
        self.assertEqual(leftovers, [])  # no stray temp files


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
        text = ("Use sk-learn style names; bearer of bad news; AIza is just a prefix; gsk_short; "
                "send a Bearer token header; Bearer authentication failed")
        self.assertEqual(redact_secrets(text), text)

    def test_long_bearer_value_is_redacted_even_if_harmless(self):
        # Deliberate fail-safe: after "Bearer", 16+ token characters are treated as a
        # credential even when they might be prose or a placeholder. The store is
        # replayed into agent prompts and committed, so over-redacting is the safe error.
        self.assertEqual(redact_secrets("Bearer your-access-token-here"), "Bearer [REDACTED]")
        self.assertEqual(redact_secrets("Bearer token-authentication-flow"), "Bearer [REDACTED]")


HOLDER = """
import sys, time
sys.path.insert(0, {here!r})
from pathlib import Path
from reflect import store_lock
with store_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(60)
"""


def record(store, n=0):
    return record_reflection(domain="d", trigger="t", symptom=f"s{n}", root_cause="r",
                             permanent_rule=f"p{n}", store_path=store)


class TestDeadHolder(unittest.TestCase):
    def test_lock_of_a_killed_writer_is_released_by_the_os(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            holder = subprocess.Popen([sys.executable, "-c", HOLDER.format(here=str(HERE)), str(store)],
                                      stdout=subprocess.PIPE)
            try:
                self.assertEqual(holder.stdout.readline().strip(), b"locked")
                holder.kill()                     # dies mid-critical-section
                holder.wait(timeout=10)
            finally:
                holder.stdout.close()
            t0 = time.monotonic()
            self.assertEqual(record(store)["status"], "created")
            self.assertLess(time.monotonic() - t0, 2.0)  # no stale-lock wait

    def test_leftover_lock_file_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            (Path(tmp) / "store.jsonl.lock").write_text("from an old crash")
            self.assertEqual(record(store)["status"], "created")


class TestSlowLiveOwner(unittest.TestCase):
    """A slow but live holder keeps the lock, however old the lock file looks:
    nothing can take it over and write while the holder may still replace the store."""

    def test_other_writers_wait_then_time_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store.jsonl"
            lock = Path(tmp) / "store.jsonl.lock"
            with store_lock(store):
                hour_ago = time.time() - 3600
                os.utime(lock, (hour_ago, hour_ago))  # age is irrelevant now
                with mock.patch.object(reflect, "LOCK_TIMEOUT_S", 0.3):
                    with self.assertRaises(TimeoutError):
                        record(store)
                self.assertFalse(store.exists())       # the waiting writer wrote nothing
            self.assertEqual(record(store)["status"], "created")  # free once released


if __name__ == "__main__":
    unittest.main()
