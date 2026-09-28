# 🧠 Cognitive Memory Core

> **Lightweight episodic memory retrieval engine in pure Python standard library using BM25 heuristic scoring for developer coding agents.**  
> Zero external dependencies · Sub-millisecond execution · Fully tested with `unittest` (10/10 passing).

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
4. **`test_memory.py`**: Comprehensive 10-test unit test suite validating tokenization, scoring, ranking, and deduplication.

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

Run the test suite directly with Python:

```bash
python test_memory.py
```

Output:
```text
..........
----------------------------------------------------------------------
Ran 10 tests in 0.015s

OK
```

### Verified Test Cases
1. `test_tokenize_stopwords_filtered`: Stopword removal across task descriptions.
2. `test_recall_copywriting_query`: Correctly matches brand tone anti-patterns (`MEM-001`).
3. `test_recall_css_bracket_wrapping_query`: Correctly retrieves flexbox wrap rule (`MEM-002`).
4. `test_recall_edge_routing_query`: Resolves subfolder 404 order precedence (`MEM-003`).
5. `test_recall_nested_powershell_query`: Identifies PowerShell quote breaking rules (`MEM-005`).
6. `test_record_reflection_new_entry`: Validates structured schema generation and atomic append.
7. `test_record_reflection_duplicate_prevention`: Prevents redundant identical rule logging.
8. `test_crystallize_output`: Verifies Markdown rule document compilation.
9. `test_format_injection_header`: Formats pre-flight injection blocks for agent context.
10. `test_top_k_limiting`: Strict top-k ranking and score threshold compliance.

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
