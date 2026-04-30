# MisfitCrew

Script-first Python workspace for mining, critiquing, and reporting over Qdrant collections.

## What this repo does

- `misfit_crew.py` mines `meta_reflections`, generates report + verdict, embeds both, and upserts into `misfit_reports`.
- `misfit_report_pull.py` exports `misfit_reports` entries into readable Markdown review files.
- `canon_alignment_report.py` runs corpus-level concept and clustering analysis over `meta_reflections`.

## Repository layout

- `misfit_crew.py`
- `misfit_report_pull.py`
- `canon_alignment_report.py`
- `reviews/` and `ROOTreviews/` (generated Markdown outputs)
- `misfit_ledger.json` / `misfit_failures.json` (runtime state)

## Requirements

- Python 3.10+
- Qdrant running and reachable
- Dependencies:

```bash
pip install qdrant-client httpx python-dotenv numpy scikit-learn
```

## Environment variables

Create `.env` in repo root as needed:

```env
QDRANT_URL=http://localhost:6333
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...

# Provider/model controls for misfit_crew.py
ANALYSIS_PROVIDER=DeepSeek
DEEPSEEK_CHAT_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner

EMBED_PROVIDER=OpenRouter
OPENROUTER_EMBED_URL=https://openrouter.ai/api/v1/embeddings
EMBED_MODEL=google/gemini-embedding-001

CRITIC_PROVIDER=Ollama
OLLAMA_GEN_URL=http://localhost:11434/api/generate
CRITIC_MODEL=gemma4:latest
```

## Usage

### 1) Run mining pipeline

```bash
python /home/mark/MisfitCrew/misfit_crew.py
python /home/mark/MisfitCrew/misfit_crew.py --limit 10 --sleep 1 --max-attempts 3
```

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

## Smoke checks

```bash
python /home/mark/MisfitCrew/misfit_crew.py --help
python /home/mark/MisfitCrew/misfit_report_pull.py --help
python /home/mark/MisfitCrew/canon_alignment_report.py --help
```

## Notes

- No formal `tests/` suite is currently present; validation is CLI smoke checks and targeted script runs.
- `.gitignore` now excludes local env, venv, cache files, logs/backups, and Zone.Identifier artifacts.
