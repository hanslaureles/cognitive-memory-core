"""Checks for the eval harness: metric math, and that queries.jsonl stays valid and held out."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import load_memories, load_queries, memory_text, metrics, tokenize, DEFAULT_STORE  # noqa: E402


def test_metrics_math():
    ranked = [["MEM-1", "MEM-2"], ["MEM-2", "MEM-3", "MEM-1"], ["MEM-9"], []]
    m = metrics(ranked, ["MEM-1", "MEM-1", "MEM-1", "MEM-1"])
    assert m["recall@1"] == 1 / 4
    assert m["recall@3"] == 2 / 4
    assert abs(m["mrr"] - (1 + 1 / 3) / 4) < 1e-12


def test_queries_valid_and_held_out():
    queries = load_queries()
    memories = load_memories(DEFAULT_STORE)
    assert len(queries) == 30
    assert len({q["qid"] for q in queries}) == 30
    assert {q["expected"] for q in queries} <= {m["id"] for m in memories}
    # Held out: no query copies a run of 4 tokens from the stored text it is meant to find.
    store_grams = set()
    for m in memories:
        toks = tokenize(memory_text(m))
        store_grams |= {tuple(toks[i:i + 4]) for i in range(len(toks) - 3)}
    for q in queries:
        toks = tokenize(q["query"])
        copied = {tuple(toks[i:i + 4]) for i in range(len(toks) - 3)} & store_grams
        assert not copied, f"{q['qid']} copies {copied}"
