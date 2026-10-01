# Claude Code Project Guide — Cognitive Memory Engine (`agent-memory`)

## Project Purpose
Zero-Dependency Cognitive Memory Engine (Self-Improving Agent Flywheel). Enables agents to record post-mortems, categorize failures, retrieve past heuristics using BM25, and auto-crystallize rules into agent guidelines.

## Architecture
- `recall.py`: BM25-driven fast heuristic retrieval over past post-mortems and episodic lessons.
- `reflect.py`: Post-mortem logger, failure categorization, and lesson structuring.
- `crystallize.py`: Auto-compiler turning recurring failure patterns into actionable agent rules.
- `store/`: JSONL persistent episodic experience store (`store/experience_store.jsonl`).
- `test_memory.py`: Unit test suite verifying retrieval, ranking, and crystallization.

## Tech Stack
- Standard Library Python 3.11+
- No external third-party dependencies

## Run Commands
- Query memory heuristics: `python recall.py "<query>"`
- Log post-mortem: `python reflect.py --domain <domain> --trigger "<when>" --symptom "<what broke>" --root-cause "<why>" --rule "<permanent rule>" [--fix "<fix>"] [--tags a,b] [--severity low|medium|high|critical]`
- Crystallize rules: `python crystallize.py`

## Build Commands
- No build step required (pure Python).

## Test Commands
- Run test suite (same as CI): `python -m pytest -q` (covers `test_memory.py` and `test_reflect_safety.py`)

## Important Constraints
- **Zero Dependencies**: Must remain 100% standard library Python (`math`, `json`, `os`, `sys`, `unittest`, `re`).
- **Deterministic**: Pure deterministic scoring for BM25 with clean tokenization.

## Known Issues
- Large JSONL memory store files should be compacted periodically to avoid linear scan overhead.

## Security Requirements
- Never log plaintext API keys, passwords, or personal credentials into episodic memory.
- `reflect.py` redacts credential shapes (Groq, Google, OpenAI/Anthropic keys, Discord tokens, Bearer headers) before writing to `store/experience_store.jsonl`. Other personal data in stack traces is still the caller's job to strip.
- Writes hold `store/experience_store.jsonl.lock` (O_EXCL lockfile; reclaimed after 30 s if a writer died) and replace the store atomically.

## Deployment Information
- Local library and CLI utility used directly by agents and local orchestration scripts.
