#!/usr/bin/env python3
"""
recall.py - Pre-Flight Recall Engine for Self-Improving Developer Agent

Searches episodic memory (experience_store.jsonl) for past lessons, anti-patterns,
and heuristics relevant to an upcoming task or bug fix.
Zero external dependencies (pure Python standard library).
"""

import os
import sys
import json
import math
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

DEFAULT_STORE = Path(__file__).parent / "store" / "experience_store.jsonl"

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    "it", "its", "of", "that", "this", "these", "those", "we", "our", "you",
    "your", "they", "their", "my", "me", "i", "how", "what", "which", "who",
    "whom", "why", "where", "when", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "s", "t", "just",
    "don", "now", "let", "lets", "make", "want", "like"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words, filtering stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def load_memories(store_path: Path = DEFAULT_STORE) -> List[Dict[str, Any]]:
    """Load all memories from the JSONL store."""
    if not store_path.exists():
        return []
    memories = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return memories


def score_memory(
    mem: Dict[str, Any],
    query_tokens: List[str],
    tag_filters: List[str],
    idf_weights: Dict[str, float],
    avg_doc_len: float
) -> float:
    """
    Score a single memory using a BM25-adapted algorithm with tag and severity boosting.
    """
    if not query_tokens:
        return 0.0

    # Field weights
    field_weights = {
        "tags": 3.0,
        "trigger": 2.5,
        "symptom": 2.0,
        "permanent_rule": 2.0,
        "root_cause": 1.5,
        "domain": 2.0,
        "fix_applied": 1.0
    }

    # Extract tokens per field
    field_tokens: Dict[str, List[str]] = {}
    doc_tokens: List[str] = []
    for field, weight in field_weights.items():
        val = mem.get(field, "")
        if isinstance(val, list):
            val = " ".join(val)
        toks = tokenize(str(val))
        field_tokens[field] = toks
        doc_tokens.extend(toks)

    doc_len = len(doc_tokens)
    if doc_len == 0:
        return 0.0

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    score = 0.0

    # Calculate match per query token
    for q in query_tokens:
        idf = idf_weights.get(q, 1.0)
        
        # Calculate weighted term frequency across fields
        weighted_tf = 0.0
        for field, weight in field_weights.items():
            tf = field_tokens[field].count(q)
            weighted_tf += tf * weight

        if weighted_tf > 0:
            numerator = weighted_tf * (k1 + 1)
            denominator = weighted_tf + k1 * (1 - b + b * (doc_len / avg_doc_len if avg_doc_len > 0 else 1))
            score += idf * (numerator / denominator)

    # Tag filter boost
    if tag_filters:
        mem_tags = [t.lower() for t in mem.get("tags", [])]
        matched_tags = sum(1 for tf in tag_filters if tf.lower() in mem_tags)
        score += matched_tags * 2.5

    # Severity & frequency multiplier
    severity = mem.get("severity", "medium").lower()
    severity_mult = {"critical": 1.4, "high": 1.25, "medium": 1.0, "low": 0.85}.get(severity, 1.0)
    
    freq = mem.get("frequency", 1)
    freq_mult = 1.0 + min(0.5, (freq - 1) * 0.1)

    return score * severity_mult * freq_mult


def recall_memories(
    query: str,
    store_path: Path = DEFAULT_STORE,
    tags: List[str] = None,
    top_k: int = 3,
    min_score: float = 0.5
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Retrieve top-k relevant memories for a given query and optional tags.
    """
    memories = load_memories(store_path)
    if not memories:
        return []

    query_tokens = tokenize(query)
    tag_filters = tags or []

    # If query is empty but tags provided, use tags as query tokens
    if not query_tokens and tag_filters:
        query_tokens = tag_filters

    if not query_tokens:
        return []

    # Calculate corpus IDF
    total_docs = len(memories)
    doc_freq: Dict[str, int] = {}
    doc_lengths: List[int] = []

    for mem in memories:
        all_text = " ".join([
            str(mem.get("domain", "")),
            " ".join(mem.get("tags", []) if isinstance(mem.get("tags"), list) else []),
            str(mem.get("trigger", "")),
            str(mem.get("symptom", "")),
            str(mem.get("root_cause", "")),
            str(mem.get("permanent_rule", "")),
            str(mem.get("fix_applied", ""))
        ])
        toks = tokenize(all_text)
        doc_lengths.append(len(toks))
        unique_toks = set(toks)
        for t in unique_toks:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    avg_doc_len = sum(doc_lengths) / max(1, total_docs)

    idf_weights: Dict[str, float] = {}
    for t in query_tokens:
        df = doc_freq.get(t, 0)
        # Standard BM25 IDF formulation with smoothing
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        idf_weights[t] = max(0.2, idf)

    # Score each memory
    scored_memories = []
    for mem in memories:
        sc = score_memory(mem, query_tokens, tag_filters, idf_weights, avg_doc_len)
        if sc >= min_score:
            scored_memories.append((mem, round(sc, 3)))

    # Sort descending by score
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return scored_memories[:top_k]


def format_injection_header(scored_memories: List[Tuple[Dict[str, Any], float]]) -> str:
    """Format matched memories into an agent-friendly markdown injection header."""
    if not scored_memories:
        return "<!-- No specific past failure patterns matched for this task. -->"

    lines = [
        "## ⚡ ACTIVE HEURISTICS & ANTI-PATTERNS (Self-Improving Agent Engine)",
        "The following permanent rules were distilled from past real-world failures and user corrections in this workspace.",
        "Adhere to them strictly before generating or modifying code:\n"
    ]

    for i, (mem, score) in enumerate(scored_memories, 1):
        rule_id = mem.get("id", f"MEM-{i}")
        domain = mem.get("domain", "general").upper()
        rule = mem.get("permanent_rule", "").strip()
        symptom = mem.get("symptom", "").strip()
        root_cause = mem.get("root_cause", "").strip()
        tags = ", ".join(mem.get("tags", []))

        lines.append(f"### [{rule_id}] {domain} (Match Relevance: {score})")
        lines.append(f"- **Permanent Rule**: {rule}")
        if symptom:
            lines.append(f"- **Past Failure / Symptom**: {symptom}")
        if root_cause:
            lines.append(f"- **Root Cause**: {root_cause}")
        lines.append(f"- **Tags**: `{tags}`\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query past developer lessons and heuristics.")
    parser.add_argument("query", nargs="*", help="Task prompt, bug symptom, or file context")
    parser.add_argument("--tags", "-t", help="Comma-separated tags (e.g. 'css,routing,netlify')")
    parser.add_argument("--top", "-k", type=int, default=3, help="Max results to return (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum relevance score threshold")
    parser.add_argument("--format", "-f", choices=["markdown", "json", "concise"], default="markdown",
                        help="Output format: markdown (default), json, or concise")
    parser.add_argument("--store", type=str, help="Custom path to experience_store.jsonl")

    args = parser.parse_args()
    store_path = Path(args.store) if args.store else DEFAULT_STORE

    try:
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    query_str = " ".join(args.query).strip() if args.query else ""
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    if not query_str and not tag_list:
        parser.print_help()
        sys.exit(1)

    results = recall_memories(
        query=query_str,
        store_path=store_path,
        tags=tag_list,
        top_k=args.top,
        min_score=args.min_score
    )

    if args.format == "json":
        payload = [
            {"score": score, "memory": mem}
            for mem, score in results
        ]
        print(json.dumps(payload, indent=2))
    elif args.format == "concise":
        if not results:
            print("No matching heuristics found.")
        for mem, score in results:
            print(f"[{mem.get('id')}] ({score:.2f}) {mem.get('permanent_rule')}")
    else:
        print(format_injection_header(results))


if __name__ == "__main__":
    main()
