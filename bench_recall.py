#!/usr/bin/env python3
"""
bench_recall.py - Reproducible latency measurement for recall_memories().

Times N calls (default 2,000) cycling through the five seeded test queries
against the real store, and prints p50 / p95 / max with the conditions needed
to interpret them (store size, Python, OS, CPU). Each call includes reading
and parsing the JSONL store, as the CLI does. Pure stdlib.

Usage: python bench_recall.py [--calls 2000]
"""

import argparse
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recall import DEFAULT_STORE, load_memories, recall_memories

# The same queries test_memory.py checks for top-1 (MEM-001 .. MEM-005).
QUERIES = [
    "words sound computer robotic manifest specimen coffee store",
    "navigation bracket wrapped next line wide desktop screen flexbox",
    "netlify deploy 404 subfolder route redirect not working",
    "philippines checkout payment method gcash maya",
    "powershell TerminatorExpectedAtEndOfString quotes parsing error",
]


def percentile(sorted_ms, p):
    """Nearest-rank percentile."""
    k = max(0, min(len(sorted_ms) - 1, round(p / 100 * len(sorted_ms) + 0.5) - 1))
    return sorted_ms[k]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calls", type=int, default=2000)
    args = parser.parse_args()

    recall_memories(QUERIES[0])  # warm the OS file cache; not timed
    times = []
    for i in range(args.calls):
        t0 = time.perf_counter()
        recall_memories(QUERIES[i % len(QUERIES)], store_path=DEFAULT_STORE)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(f"date:   {date.today().isoformat()}")
    print(f"host:   {platform.processor() or platform.machine()} · {platform.system()} {platform.release()} · Python {platform.python_version()}")
    print(f"store:  {len(load_memories(DEFAULT_STORE))} entries ({DEFAULT_STORE.name})")
    print(f"calls:  {args.calls} (5 seeded queries, cycled)")
    print(f"p50:    {statistics.median(times):.2f} ms")
    print(f"p95:    {percentile(times, 95):.2f} ms")
    print(f"max:    {times[-1]:.2f} ms")


if __name__ == "__main__":
    main()
