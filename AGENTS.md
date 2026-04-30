# AGENTS Guide

## Repository Snapshot
- Project type: script-first Python CLI workspace (no package layout, no `tests/` directory).
- Primary purpose: mine and analyze Qdrant collection data, then write markdown reports.
- Key scripts:
  - `misfit_crew.py`: mining + critique + embedding + upsert pipeline.
  - `misfit_report_pull.py`: export `misfit_reports` rows to markdown review files.
  - `canon_alignment_report.py`: corpus-level concept + clustering report from `meta_reflections`.

## Essential Commands
## Setup
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke checks (also what CI runs)
```bash
python misfit_crew.py --help
python misfit_report_pull.py --help
python canon_alignment_report.py --help
```

## Typical runtime commands
```bash
# Mining pipeline
python misfit_crew.py
python misfit_crew.py --limit 10 --sleep 1 --max-attempts 3

# Pull reports from misfit_reports
python misfit_report_pull.py --list
python misfit_report_pull.py --source <source_file>
python misfit_report_pull.py --all --out ./reviews

# Canon alignment report
python canon_alignment_report.py
python canon_alignment_report.py --k 40 --sample 10000
python canon_alignment_report.py --collection meta_reflections --out ./reviews/canon_alignment_report.md
```

## Environment and External Dependencies
- Python: README states 3.10+, CI uses 3.11 (`.github/workflows/ci.yml`).
- Required services/APIs observed in code:
  - Qdrant at `QDRANT_URL` (default `http://localhost:6333`).
  - Analysis call uses env-driven `DEEPSEEK_CHAT_URL` + `DEEPSEEK_MODEL` and requires `DEEPSEEK_API_KEY`.
  - Embedding call uses env-driven `OPENROUTER_EMBED_URL` + `EMBED_MODEL` and requires `OPENROUTER_API_KEY`.
  - Critic call uses env-driven `OLLAMA_GEN_URL` + `CRITIC_MODEL` (default local Ollama + `gemma4:latest`).
  - Provider labels are env-driven (`ANALYSIS_PROVIDER`, `EMBED_PROVIDER`, `CRITIC_PROVIDER`) for runtime config reporting.
- Declared dependencies (`requirements.txt`):
  - `qdrant-client`, `httpx`, `python-dotenv`, `numpy`, `scikit-learn`.

## Data/Collection Contracts (Important)

### `misfit_crew.py` contracts
- Source collection: `meta_reflections`.
- Destination collection: `misfit_reports`.
- If `misfit_reports` is missing, script creates it with **named vectors**:
  - `claims_vec` (3072-dim, cosine)
  - `summary_vec` (3072-dim, cosine)
- Embedding model defaults to `google/gemini-embedding-001` but is now env-configurable via `EMBED_MODEL`.
- Upsert ID is the source point ID string (lineage preservation).

### Runtime state files
- `misfit_ledger.json`: append-only audit log; explicitly not used for exclusion decisions.
- `misfit_failures.json`: retry/dead-well state, including attempt counts and recent errors.
- Failure handling behavior:
  - Retries DeepSeek calls with backoff `[2, 8, 30]` on 429/5xx.
  - A point becomes "dead" when attempts reach `--max-attempts`; dead IDs are skipped until failure state is cleared.

## Architecture and Control Flow

### 1) Mining pipeline (`misfit_crew.py`)
1. Validate `DEEPSEEK_API_KEY`.
2. Ensure destination collection exists with required vector schema.
3. Scroll `misfit_reports` once at startup to build in-memory mined ID set.
4. Load failures, derive dead ID set.
5. Scroll `meta_reflections` in batches (`DEFAULT_SCROLL_BATCH=128`), skipping mined/dead IDs.
6. For each eligible point:
   - DeepSeek reasoning/report generation.
   - Gemma critique via local Ollama.
   - Embed report + verdict via OpenRouter.
   - Upsert to `misfit_reports` with named vectors + payload.
   - Append to `misfit_ledger.json` and clear failure record on success.
7. Print throughput + ETA counters; obey `--limit`; support graceful SIGINT stop.

### 2) Report extraction (`misfit_report_pull.py`)
- Scrolls `misfit_reports` and either:
  - lists available `source_file` values with counts (`--list`),
  - exports one source (`--source`), or
  - exports all sources (`--all`).
- Default output directory is `./ROOTreviews` (not `./reviews`), unless overridden by `--out`.
- Markdown renderer includes reasoning/report/verdict sections per point and sorts by parsed `mined_at`.

### 3) Canon analysis (`canon_alignment_report.py`)
- Scrolls vectors + selected payload fields from `meta_reflections`.
- Handles plain or named vectors; for named vectors it takes the first vector value encountered.
- Normalizes concept tokens (lowercase/whitespace collapse + comma splitting).
- Produces:
  - Concept frequency table ranked by cross-source spread then count.
  - MiniBatchKMeans clusters (`k` configurable) over L2-normalized vectors.
  - Canonical vs provincial cluster grouping by `--canonical-threshold`.
- Writes markdown report to `./reviews/canon_alignment_report.md` by default.

## Coding and Style Patterns Observed
- Single-file CLI scripts using `argparse` and `if __name__ == "__main__":` entrypoints.
- Heavy use of module-level constants for config defaults.
- Qdrant access pattern is mostly `scroll(...)` loops with pagination `offset`.
- Robust-ish file writes for mutable JSON state via temp-file + `os.replace(...)`.
- Type hints are present but no static-type tooling is configured in repo.

## Testing and Validation Reality
- No formal unit/integration test suite exists.
- CI only executes `--help` smoke checks for each script.
- For changes, safest repo-consistent verification is:
  1. run all three `--help` checks,
  2. run targeted script command relevant to modified script against a reachable Qdrant instance.

## Gotchas / Non-obvious Behaviors
- `misfit_report_pull.py` writes to `ROOTreviews/` by default, while README examples often use `reviews/`.
- `misfit_crew.py` requires three external dependencies at runtime (Qdrant + DeepSeek + OpenRouter + local Ollama); smoke checks do not validate live pipeline behavior.
- Dead-point behavior is sticky via `misfit_failures.json`; reprocessing may require manual cleanup of that file.
- `misfit_crew.py` computes "remaining" once near startup; ETA/progress reflect that static baseline, not dynamic recounting.
- `canon_alignment_report.py` return annotation for `cluster_reflections` says 2 values but function actually returns 3 (`labels, centroids, X`); callers already use 3.

## CI/Automation
- Single workflow: `.github/workflows/ci.yml`.
- Trigger: push/PR on `main` + manual dispatch.
- Job does install from `requirements.txt` and CLI `--help` checks only.

## Working Guidelines for Future Agents
- Treat this as a script workspace, not a package with centralized shared utilities.
- Preserve collection names and vector field names unless migration is intentional (`meta_reflections`, `misfit_reports`, `claims_vec`, `summary_vec`).
- When changing runtime-state behavior, account for both `misfit_ledger.json` and `misfit_failures.json` semantics.
- Prefer adding/maintaining smoke-check-compatible CLI behavior (`--help` must stay healthy), since that is what CI enforces today.
