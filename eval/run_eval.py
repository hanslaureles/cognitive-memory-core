#!/usr/bin/env python3
"""
run_eval.py - Held-out retrieval eval and latency scaling for recall.py.

Quality: recall@1, recall@3 and MRR over eval/queries.jsonl (paraphrased
queries, each labelled with one expected MEM-id) for BM25 (recall_memories)
and a naive substring baseline, overall and per query set.
Latency: p50/p95 of recall_memories() (store read + parse + score, as the CLI
does) on the real store (N=10) and on seeded synthetic stores of 1k and 10k.

Usage:
  python eval/run_eval.py            # full run, writes eval/results/YYYY-MM-DD.md
  python eval/run_eval.py --check    # quality only; exit 1 if BM25 recall@3 < floor
Pure stdlib.
"""

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from recall import DEFAULT_STORE, load_memories, recall_memories, tokenize  # noqa: E402

QUERIES = EVAL_DIR / "queries.jsonl"
TEXT_FIELDS = ("domain", "trigger", "symptom", "root_cause", "fix_applied", "permanent_rule")

# CI gate: BM25 numbers measured on 2026-10-02 (30 queries), minus a tolerance.
# Re-measure and update all four when queries.jsonl changes.
# MRR is gated too: on 10 entries a scorer that ignores ranking still keeps
# recall@3 at 0.87 (one query lost) but drops MRR to 0.71.
RECALL3_BASELINE = 27 / 30
RECALL3_TOLERANCE = 1 / 30  # one query
MRR_BASELINE = 26 / 30
MRR_TOLERANCE = 0.05

LATENCY_SIZES = (10, 1_000, 10_000)
CALLS_PER_SIZE = {10: 500, 1_000: 100, 10_000: 20}


def load_queries(path=QUERIES):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def memory_text(mem):
    tags = mem.get("tags", [])
    return " ".join([str(mem.get(k, "")) for k in TEXT_FIELDS] + (tags if isinstance(tags, list) else []))


def rank_bm25(query, store_path):
    """Every memory with a positive score, best first."""
    return [m["id"] for m, _ in recall_memories(query, store_path=store_path, top_k=10**9, min_score=1e-9)]


def rank_substring(query, memories):
    """Naive substring baseline (not a real grep): count distinct query words found as substrings, ties in store order."""
    words = set(query.lower().split())
    hits = [(sum(w in memory_text(m).lower() for w in words), i, m["id"]) for i, m in enumerate(memories)]
    return [mid for n, _, mid in sorted(hits, key=lambda h: (-h[0], h[1])) if n > 0]


def metrics(ranked_lists, expected):
    """recall@1, recall@3 and MRR; a missing expected id counts as rank infinity."""
    ranks = [r.index(e) + 1 if e in r else None for r, e in zip(ranked_lists, expected)]
    n = len(ranks)
    return {
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@3": sum(1 for r in ranks if r and r <= 3) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def evaluate(queries, store_path=DEFAULT_STORE):
    memories = load_memories(store_path)
    known = {m["id"] for m in memories}
    missing = {q["expected"] for q in queries} - known
    if missing:
        raise SystemExit(f"queries.jsonl expects ids not in the store: {sorted(missing)}")
    runs = {
        "BM25 (recall.py)": [rank_bm25(q["query"], store_path) for q in queries],
        "Substring baseline": [rank_substring(q["query"], memories) for q in queries],
    }
    sets = sorted({q["set"] for q in queries})
    out = {}
    for name, ranked in runs.items():
        out[name] = {"all": metrics(ranked, [q["expected"] for q in queries])}
        for s in sets:
            idx = [i for i, q in enumerate(queries) if q["set"] == s]
            out[name][s] = metrics([ranked[i] for i in idx], [queries[i]["expected"] for i in idx])
    misses = [(q["qid"], q["expected"], r[:3]) for q, r in zip(queries, runs["BM25 (recall.py)"]) if q["expected"] not in r[:3]]
    return out, misses


def synthetic_store(n, real, path, seed=42):
    """Real entries first, then seeded filler built from the real store's vocabulary."""
    rng = random.Random(seed)
    vocab = sorted({t for m in real for t in tokenize(memory_text(m))})
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            if i < len(real):
                mem = real[i]
            else:
                words = lambda k: " ".join(rng.choices(vocab, k=k))  # noqa: E731
                mem = {"id": f"SYN-{i:05d}", "domain": words(1), "trigger": words(10), "symptom": words(12),
                       "root_cause": words(14), "fix_applied": words(12), "permanent_rule": words(20),
                       "tags": rng.choices(vocab, k=6), "severity": rng.choice(["low", "medium", "high"]),
                       "frequency": rng.randint(1, 3)}
            f.write(json.dumps(mem) + "\n")


def percentile(sorted_ms, p):
    """Nearest-rank percentile."""
    k = max(0, min(len(sorted_ms) - 1, round(p / 100 * len(sorted_ms) + 0.5) - 1))
    return sorted_ms[k]


def latency(queries):
    real = load_memories(DEFAULT_STORE)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in LATENCY_SIZES:
            path = DEFAULT_STORE if n == len(real) else Path(tmp) / f"syn_{n}.jsonl"
            if path != DEFAULT_STORE:
                synthetic_store(n, real, path)
            recall_memories(queries[0]["query"], store_path=path)  # warm the OS file cache; not timed
            times = []
            for i in range(CALLS_PER_SIZE[n]):
                t0 = time.perf_counter()
                recall_memories(queries[i % len(queries)]["query"], store_path=path)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            rows.append((n, len(times), statistics.median(times), percentile(times, 95)))
    return rows


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def os_name():
    if sys.platform == "win32" and sys.getwindowsversion().build >= 22000:
        return f"Windows 11 (build {sys.getwindowsversion().build})"  # platform.release() still says "10"
    return f"{platform.system()} {platform.release()}"


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "recall.py", "eval/queries.jsonl"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + (" (recall.py or queries.jsonl modified)" if dirty else "")
    except OSError:
        return "unknown"


def report(queries, quality, misses, lat):
    sets = sorted({q["set"] for q in queries})
    counts = {s: sum(q["set"] == s for q in queries) for s in sets}
    lines = [
        f"# Recall eval: {date.today().isoformat()}",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Hardware: {cpu_name()} · {os_name()}",
        f"- Python: {platform.python_version()}",
        f"- Git SHA: {git_sha()}",
        f"- Store: {len(load_memories(DEFAULT_STORE))} entries (`store/experience_store.jsonl`)",
        f"- Queries: {len(queries)} held-out paraphrases, one expected MEM-id each ("
        + ", ".join(f"set {s}: {counts[s]}" for s in sets) + ")",
        "",
        "## Quality",
        "",
        "| Retriever | Queries | recall@1 | recall@3 | MRR |",
        "|---|---|---|---|---|",
    ]
    for name, by_set in quality.items():
        for key in ["all"] + sets:
            label = f"all ({len(queries)})" if key == "all" else f"set {key} ({counts[key]})"
            m = by_set[key]
            lines.append(f"| {name} | {label} | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['mrr']:.2f} |")
    lines += ["", f"With {len(load_memories(DEFAULT_STORE))} entries, a top-3 list covers a large share of the store, so",
              "recall@3 separates retrievers weakly here; recall@1 and MRR are the sharper numbers.",
              "The baseline counts stopwords as matches and breaks ties in store order."]
    lines += ["", "BM25 misses at k=3 (qid, expected, top 3 returned):", ""]
    lines += [f"- {qid} expected {exp}, got {', '.join(top) or 'nothing'}" for qid, exp, top in misses] or ["- none"]
    lines += [
        "",
        "## Latency of `recall_memories()` (read + parse + score per call)",
        "",
        "| N entries | Calls | p50 ms | p95 ms |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n:,} | {c} | {p50:.2f} | {p95:.2f} |" for n, c, p50, p95 in lat]
    lines += ["", "N=10 is the real store. 1k and 10k are the real entries plus seeded (seed 42) synthetic",
              "entries drawn from the real store's vocabulary. Queries cycle through queries.jsonl.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="quality gate only (CI)")
    args = parser.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    queries = load_queries()
    quality, misses = evaluate(queries)
    if args.check:
        bm25 = quality["BM25 (recall.py)"]["all"]
        ok = True
        for name, base, tol in (("recall@3", RECALL3_BASELINE, RECALL3_TOLERANCE), ("mrr", MRR_BASELINE, MRR_TOLERANCE)):
            floor = base - tol
            passed = bm25[name] >= floor - 1e-9
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'} BM25 {name} = {bm25[name]:.3f} (floor {floor:.3f}, {len(queries)} queries)")
        sys.exit(0 if ok else 1)

    md = report(queries, quality, misses, latency(queries))
    out = EVAL_DIR / "results" / f"{date.today().isoformat()}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
