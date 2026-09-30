#!/usr/bin/env python3
"""
test_memory.py - Test Suite for Self-Improving Developer Agent Engine

Validates:
1. Search and retrieval scoring accuracy (BM25 & tag boosting).
2. Exact ranking on real scenario queries.
3. Post-mortem reflection logging, ID generation, and deduplication logic.
4. Crystallization compiler and markdown rule generation.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from recall import recall_memories, format_injection_header, tokenize
from reflect import record_reflection, load_all_memories, save_all_memories
from crystallize import crystallize, build_markdown_rules, find_workspace_root


class TestRecallEngine(unittest.TestCase):

    def setUp(self):
        self.store_path = Path(__file__).parent / "store" / "experience_store.jsonl"
        self.assertTrue(self.store_path.exists(), "Experience store file must exist.")

    def test_tokenize_stopwords_filtered(self):
        tokens = tokenize("This is a simple test for the recall engine")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("simple", tokens)
        self.assertIn("recall", tokens)
        self.assertIn("engine", tokens)

    def test_recall_copywriting_query(self):
        """Query about robotic/computer terms in coffee store must rank MEM-001 highest."""
        results = recall_memories(
            query="words sound computer robotic manifest specimen coffee store",
            store_path=self.store_path,
            top_k=2
        )
        self.assertGreater(len(results), 0)
        top_mem, score = results[0]
        self.assertEqual(top_mem["id"], "MEM-001")
        self.assertEqual(top_mem["domain"], "copywriting")
        self.assertGreater(score, 1.0)

    def test_recall_css_bracket_wrapping_query(self):
        """Query about brackets wrapping in navigation on wide screens must rank MEM-002 highest."""
        results = recall_memories(
            query="navigation bracket wrapped next line wide desktop screen flexbox",
            store_path=self.store_path,
            top_k=2
        )
        self.assertGreater(len(results), 0)
        top_mem, score = results[0]
        self.assertEqual(top_mem["id"], "MEM-002")
        self.assertIn("bracket", top_mem["permanent_rule"].lower())

    def test_recall_netlify_routing_query(self):
        """Query about Netlify subfolder 404 must rank MEM-003 highest."""
        results = recall_memories(
            query="netlify deploy 404 subfolder route redirect not working",
            store_path=self.store_path,
            top_k=2
        )
        self.assertGreater(len(results), 0)
        top_mem, score = results[0]
        self.assertEqual(top_mem["id"], "MEM-003")
        self.assertIn("netlify", top_mem["tags"])

    def test_recall_philippines_payment_query(self):
        """Query about local checkout in Manila / Philippines must rank MEM-004."""
        results = recall_memories(
            query="philippines checkout payment method gcash maya",
            store_path=self.store_path,
            top_k=2
        )
        self.assertGreater(len(results), 0)
        top_mem, score = results[0]
        self.assertEqual(top_mem["id"], "MEM-004")
        self.assertIn("gcash", top_mem["permanent_rule"].lower())

    def test_recall_powershell_error_query(self):
        """Query about PowerShell unclosed quotes must rank MEM-005."""
        results = recall_memories(
            query="powershell TerminatorExpectedAtEndOfString quotes parsing error",
            store_path=self.store_path,
            top_k=2
        )
        self.assertGreater(len(results), 0)
        top_mem, score = results[0]
        self.assertEqual(top_mem["id"], "MEM-005")
        self.assertIn("powershell", top_mem["permanent_rule"].lower())

    def test_injection_header_formatting(self):
        results = recall_memories(query="netlify 404", store_path=self.store_path, top_k=1)
        header = format_injection_header(results)
        self.assertIn("ACTIVE HEURISTICS & ANTI-PATTERNS", header)
        self.assertIn("MEM-003", header)


class TestReflectionEngine(unittest.TestCase):

    def setUp(self):
        # Create a temporary JSONL store
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_store = Path(self.temp_dir.name) / "test_store.jsonl"
        # Seed with 1 memory
        seed = [
            {
                "id": "MEM-001",
                "created_at": "2026-09-17T00:00:00Z",
                "updated_at": "2026-09-17T00:00:00Z",
                "domain": "frontend",
                "trigger": "Button hover animation",
                "symptom": "Button jumps abruptly",
                "root_cause": "Missing transition property",
                "fix_applied": "Added transition: all 0.2s ease",
                "permanent_rule": "Always define transition property on interactive elements before transform",
                "tags": ["css", "animation"],
                "severity": "medium",
                "frequency": 1
            }
        ]
        save_all_memories(seed, self.test_store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_new_reflection_appends_and_increments_id(self):
        res = record_reflection(
            domain="database",
            trigger="Running unindexed SQL queries on large tables",
            symptom="Query timeout after 30 seconds",
            root_cause="Missing compound index on user_id and status",
            permanent_rule="Always create compound indices for multi-column WHERE clauses on high-cardinality tables",
            fix_applied="Added CREATE INDEX idx_user_status ON orders(user_id, status)",
            tags=["sql", "database", "performance"],
            severity="high",
            store_path=self.test_store
        )
        self.assertEqual(res["status"], "created")
        self.assertEqual(res["id"], "MEM-002")

        all_records = load_all_memories(self.test_store)
        self.assertEqual(len(all_records), 2)
        self.assertEqual(all_records[1]["id"], "MEM-002")
        self.assertEqual(all_records[1]["frequency"], 1)

    def test_similar_reflection_increments_frequency(self):
        # Attempt to add almost identical rule
        res = record_reflection(
            domain="frontend",
            trigger="Button hover animation",
            symptom="Button jumps abruptly during hover",
            root_cause="Missing transition property",
            permanent_rule="Always define transition property on interactive elements before transform",
            fix_applied="Added transition: all 0.2s ease",
            tags=["css", "animation", "transitions"],
            severity="high",
            store_path=self.test_store
        )
        self.assertEqual(res["status"], "updated_frequency")
        self.assertEqual(res["id"], "MEM-001")
        self.assertEqual(res["frequency"], 2)

        all_records = load_all_memories(self.test_store)
        self.assertEqual(len(all_records), 1)
        self.assertEqual(all_records[0]["frequency"], 2)
        # Verify tag was merged
        self.assertIn("transitions", all_records[0]["tags"])


class TestCrystallizeEngine(unittest.TestCase):

    def test_crystallize_generates_markdown(self):
        temp_dir = tempfile.TemporaryDirectory()
        test_output = Path(temp_dir.name) / "rules.md"
        store_path = Path(__file__).parent / "store" / "experience_store.jsonl"

        out_path = crystallize(store_path=store_path, output_path=test_output)
        self.assertTrue(out_path.exists())

        content = out_path.read_text(encoding="utf-8")
        self.assertIn("# Workspace Learned Rules & Heuristics", content)
        self.assertIn("Copywriting Heuristics", content)
        self.assertIn("Css-Layout Heuristics", content)
        self.assertIn("Edge-Routing Heuristics", content)
        self.assertIn("MEM-001", content)
        self.assertIn("MEM-002", content)
        self.assertIn("MEM-003", content)
        temp_dir.cleanup()

    def test_workspace_root_found_through_nested_projects_dir(self):
        """Rules must resolve to the ancestor that owns .agents/, not the repo's direct parent."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            (workspace / ".agents" / "rules").mkdir(parents=True)
            script = workspace / "Projects" / "agent-memory" / "crystallize.py"
            script.parent.mkdir(parents=True)
            script.touch()
            self.assertEqual(find_workspace_root(script), workspace)

    def test_workspace_root_falls_back_without_agents_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp).resolve() / "repo" / "crystallize.py"
            script.parent.mkdir(parents=True)
            script.touch()
            # No .agents/ anywhere under tmp; ancestors above tmp are assumed not to have one either.
            if any((p / ".agents").is_dir() for p in Path(tmp).resolve().parents):
                self.skipTest("An ancestor of the temp dir has .agents/")
            self.assertEqual(find_workspace_root(script), Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
