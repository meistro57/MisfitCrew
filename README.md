# MisfitCrew
<img width="328" height="305" alt="image" src="https://github.com/user-attachments/assets/330c2c3a-5b70-4be2-a995-1568e49ed2ba" />

Script-first workspace for mining, critiquing, and reporting over Qdrant collections.

## What this repo does

### Python pipelines
- `misfit_crew.py` mines `meta_reflections`, generates report + verdict, embeds both, and upserts into `misfit_reports`.
- `misfit_report_pull.py` exports `misfit_reports` entries into readable Markdown review files.
- `canon_alignment_report.py` runs corpus-level concept and clustering analysis over `meta_reflections`.

### Go tools
- `receipts/` builds concept synthesis docs from `meta_reflections` + `misfit_reports`.
- `mcp-servers/` builds MCP binaries for Qdrant and Redis.

## Repository layout

- `misfit_crew.py`
- `misfit_report_pull.py`
- `canon_alignment_report.py`
- `receipts/`
- `mcp-servers/`
- `reviews/`, `ROOTreviews/`, `report/`, `report2/` (generated Markdown outputs)
- `misfit_ledger.json` / `misfit_failures.json` (runtime state)

## Requirements

### Python
- Python 3.10+
- Qdrant running and reachable

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Go (for `receipts/` and `mcp-servers/`)
- Go 1.24+

## Environment variables

Create `.env` in repo root as needed:

```env
# Shared
QDRANT_URL=http://localhost:6333
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...
ANALYSIS_API_KEY=...

# Provider/model controls for misfit_crew.py
ANALYSIS_PROVIDER=DeepSeek
DEEPSEEK_CHAT_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner

EMBED_PROVIDER=OpenRouter
OPENROUTER_EMBED_URL=https://openrouter.ai/api/v1/embeddings
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=MisfitCrew
EMBED_MODEL=google/gemini-embedding-001

CRITIC_PROVIDER=Ollama
OLLAMA_GEN_URL=http://localhost:11434/api/generate
CRITIC_MODEL=gemma4:latest
CRITIC_CHAT_URL=https://openrouter.ai/api/v1/chat/completions

# receipts (gRPC client)
QDRANT_HOST=localhost
SOURCE_TRADITIONS_COUNT=6
REFLECTIONS_COLLECTION=meta_reflections
REPORTS_COLLECTION=misfit_reports
```

If `DEEPSEEK_CHAT_URL` points to OpenRouter, `misfit_crew.py` auto-uses `OPENROUTER_API_KEY` for analysis unless `ANALYSIS_API_KEY` is explicitly set.

## Usage

### 1) Run mining pipeline

```bash
python /home/mark/MisfitCrew/misfit_crew.py
python /home/mark/MisfitCrew/misfit_crew.py --limit 10 --sleep 1 --max-attempts 3
python /home/mark/MisfitCrew/misfit_crew.py --workers 4 --sleep 0.5 --limit 100
```

`--workers` controls concurrent wells (default `1`). `--limit` is global across all workers, and `--sleep` is applied between submissions.

### 2) Pull report(s) from `misfit_reports`

```bash
python /home/mark/MisfitCrew/misfit_report_pull.py --list
python /home/mark/MisfitCrew/misfit_report_pull.py --source <source_file.pdf>
python /home/mark/MisfitCrew/misfit_report_pull.py --all --out /home/mark/MisfitCrew/reviews
```

### 3) Generate canon alignment report

```bash
python /home/mark/MisfitCrew/canon_alignment_report.py
python /home/mark/MisfitCrew/canon_alignment_report.py --k 40 --sample 10000
python /home/mark/MisfitCrew/canon_alignment_report.py --collection meta_reflections --out /home/mark/MisfitCrew/reviews/canon_alignment_report.md
```

### 4) Build receipts

```bash
cd /home/mark/MisfitCrew/receipts
./rebuild.sh
./receipts --help
```

### 5) Build MCP servers

```bash
cd /home/mark/MisfitCrew/mcp-servers
make all
```

### 6) Run full all-reports batch

```bash
cd /home/mark/MisfitCrew
./run_all_reports.sh
```

## Smoke checks

```bash
python /home/mark/MisfitCrew/misfit_crew.py --help
python /home/mark/MisfitCrew/misfit_report_pull.py --help
python /home/mark/MisfitCrew/canon_alignment_report.py --help
```

## Notes

- No formal `tests/` suite is currently present for Python; validation is CLI smoke checks and targeted script runs.
- `receipts/cmd/main.go` loads env from `../.env` first, then local `.env`.
- `run_all_reports.sh` now sources and exports repo-root `.env` before running steps, so batch runs use current project keys instead of stale shell exports.
- `canon_alignment_report.py` now uses `QDRANT_TIMEOUT` (seconds, default `120`) plus retry controls (`CANON_SCROLL_RETRIES`, `CANON_SCROLL_BATCH`) to reduce scroll timeout failures on large corpora.
- `.gitignore` excludes local env, venv, cache files, logs/backups, and `Zone.Identifier` artifacts.
