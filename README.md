# 🧠 Cognitive Memory Core

> **Lightweight episodic memory retrieval engine in pure Python standard library using BM25 heuristic scoring for developer coding agents.**  
> Zero external dependencies · Sub-millisecond execution · Tested with pytest and a held-out retrieval eval in CI.

[![CI](https://github.com/hanslaureles/cognitive-memory-core/actions/workflows/ci.yml/badge.svg)](https://github.com/hanslaureles/cognitive-memory-core/actions/workflows/ci.yml)

---

## 🎯 The Problem

Large Language Model coding assistants (Claude, GPT-4, Cursor, Antigravity) suffer from **session amnesia**:
- They make a subtle implementation mistake (e.g., PowerShell nested quote escaping, Windows Python console UTF-8 codec errors, static routing order in edge deployments).
- The engineer corrects the agent.
- In the next session, the context window resets, and the agent repeats the exact same mistake.

Vector databases (Chroma, Pinecone, Qdrant) solve semantic search, but introduce heavy C++ dependencies, GPU drivers, SQLite vector extensions, and high startup latency—overkill for a local coding agent that just needs to query 10–100 past bug heuristics before executing a file edit.

---

## ⚡ The Solution

**Cognitive Memory Core** is an episodic memory retrieval flywheel built with **zero external pip dependencies** (100% Python standard library):
1. **`reflect.py`**: Captures structured post-mortem schemas (`trigger`, `symptom`, `root_cause`, `rule`, `tags`).
2. **`recall.py`**: Executes BM25 keyword scoring + tag boosting over JSONL episodic records, ranking matching rules in `<5ms`.
3. **`crystallize.py`**: Compiles validated episodic reflections into persistent Markdown rule files for LLM system prompts (`Learned_Rules.md`).
4. **Tests + eval**: pytest suites for retrieval, write safety and redaction, plus a held-out retrieval eval (`eval/`) gated in CI.

---

## 🏗️ Architecture

```
[Agent Pre-Flight Query]  e.g. "navigation bracket flexbox wide screen"
           │
           ▼
┌──────────────────────────────────────────────┐
│                  recall.py                   │
│ - Pure standard library tokenization         │
│ - Stopword filtering & normalization         │
│ - Okapi BM25 scoring + tag boosting          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           store/experience_store.jsonl        │
│ 10 Crystallized Heuristic Rules (MEM-001..10)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
[Injected Prompt Header]  Top-K Heuristic Rules (<5ms)
                       │
                       ▼
[Execution & Verification]
                       │
        (If bug occurs and is resolved)
                       ▼
┌──────────────────────────────────────────────┐
│                  reflect.py                  │
│ Logs new post-mortem to JSONL experience store│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                crystallize.py                │
│ Compiles Markdown rule index for agent brain │
└──────────────────────────────────────────────┘
```

---

## 🧪 Test Suite & Verification

Run the same checks as CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)); pytest is the only install:

```bash
python -m pytest -q
python eval/run_eval.py --check
```

- `test_memory.py`: tokenization, BM25 ranking on seeded queries, injection header, reflection dedup and crystallization.
- `test_reflect_safety.py`: concurrent writers, lock release when a writer dies, lock timeout, and credential redaction.
- `eval/test_run_eval.py`: eval metric math, and that every eval query is held out from the store.
- `eval/run_eval.py --check`: fails if BM25 recall@3 or MRR on the held-out queries drops below the measured baseline.

---

## 📏 Retrieval Eval

30 held-out queries in [`eval/queries.jsonl`](eval/queries.jsonl), each paraphrased (no 4-token run copied from the store, enforced by a test) and labelled with one expected rule. Full run: [`eval/results/2026-10-02.md`](eval/results/2026-10-02.md).

| Retriever | recall@1 | recall@3 | MRR |
|---|---|---|---|
| BM25 (`recall.py`) | 0.83 | 0.90 | 0.87 |
| Naive substring baseline | 0.80 | 0.93 | 0.87 |

| Store size | p50 | p95 |
|---|---|---|
| 10 entries (real store) | 0.81 ms | 0.87 ms |
| 1,000 (synthetic) | 67 ms | 70 ms |
| 10,000 (synthetic) | 695 ms | 744 ms |

Measured 2026-10-02 on an Intel Core i5-12400F, Windows 11, Python 3.11.9, commit `a9a7e96`. Latency is per `recall_memories()` call, including reading and parsing the JSONL store.

What it shows: on today's 10-entry store, BM25 is not clearly better than a naive substring baseline (it counts stopwords and breaks ties in store order). It ties on MRR, and with 10 entries a top-3 list covers 30% of the store. Latency grows linearly, because every call re-reads and re-scores the whole store. CI fails if BM25 recall@3 or MRR drops below these numbers minus a tolerance (`python eval/run_eval.py --check`).

---

## 🚀 Quick Usage

```python
from pathlib import Path
from recall import recall_memories, format_injection_header

store_path = Path("store/experience_store.jsonl")

# Query before executing code
query = "PowerShell script with inline python quotes"
results = recall_memories(query, store_path=store_path, top_k=2)

for memory, score in results:
    print(f"[{memory['id']}] Score: {score:.2f} — {memory['rule']}")
```

---

## 👨‍💻 Author

**Hans Aaron Laureles**  
*Applied AI Engineer & Full-Stack Builder*  
- Portfolio: [hanslaureles.vercel.app](https://hanslaureles.vercel.app/)  
- GitHub: [@hanslaureles](https://github.com/hanslaureles)  
- LinkedIn: [linkedin.com/in/hanslaureles](https://linkedin.com/in/hanslaureles)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
