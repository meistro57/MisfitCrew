# receipts

**The "Look at this!!" engine.**

Takes a concept and generates a plain-English multi-tradition synthesis document showing that people from different cultures, eras, and backgrounds independently arrived at the same idea — with receipts.

Part of the MisfitCrew pipeline, downstream of `meta_reflections` and `misfit_reports`.

---

## What it does

1. Auto-detects embedding model from `meta_reflections` vector size (if available), then embeds your concept query via OpenRouter
2. Searches `meta_reflections` for top matching chunks across distinct source traditions
3. Searches `misfit_reports` for Hardware Glitch receipts on the same material
4. Sends everything to DeepSeek to write a synthesis doc
5. Writes Markdown to `./output/<concept>_<timestamp>.md`

Output format:
- YAML front matter with run metadata for downstream automation
- **The Core Claim**
- **The Receipts**
- **Why The Convergence Matters**
- **The Hardware Glitch**
- **The Operative Question**
- **Source Ledger** table with similarity, tone, and concepts
- **Evidence Pack** appendix with source IDs, summaries, claims, concepts, and echoes
- **Hardware Glitch Receipts** appendix with critic verdicts and report excerpts

---

## Install

```bash
cd ~/MisfitCrew/receipts
go mod tidy
./rebuild.sh
./receipts --help
```

---

## Usage

```bash
# Show top corpus concepts
./receipts --list-concepts

# Generate a receipts doc
./receipts --concept "belief creates reality"

# More source traditions, fast model
./receipts --concept "consciousness survives death" --sources 8 --model deepseek-chat

# Alias for sources count
./receipts --concept "consciousness survives death" --source-traditions 10

# Custom output dir
./receipts --concept "non-linear time" --out ~/MisfitCrew/receipts/output

# Leaner Markdown without raw source appendices
./receipts --concept "non-linear time" --raw-receipts=false

# Version
./receipts --version
```

---

## Environment (.env or shell)

`receipts` loads env in this order:
1. `../.env` (repo root)
2. `./.env` (receipts directory)
3. existing shell environment

```env
QDRANT_HOST=localhost
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...          # only needed when DEEPSEEK_CHAT_URL points directly at DeepSeek
DEEPSEEK_CHAT_URL=https://openrouter.ai/api/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner # auto-maps on OpenRouter to deepseek/deepseek-r1
EMBED_MODEL=google/gemini-embedding-001 # fallback when collection model cannot be inferred
SOURCE_TRADITIONS_COUNT=6              # default when --sources is not passed
REFLECTIONS_COLLECTION=meta_reflections
REPORTS_COLLECTION=misfit_reports
```

Notes:
- Qdrant connection is gRPC at `QDRANT_HOST:6334`.
- `OPENROUTER_API_KEY` is required for embeddings and synthesis when `DEEPSEEK_CHAT_URL` is an OpenRouter endpoint.

---

## Pipeline position

```
mb_chunks
    ↓ reflect_loop_fixed.py
meta_reflections
    ↓ misfit_crew.py
misfit_reports
    ↓
receipts
    ↓
output/<concept>_<timestamp>.md
```

---

## Operational notes

- `--list-concepts` scrolls all of `meta_reflections` and ranks concepts by cross-source spread.
- `deepseek-reasoner` asks for confirmation before long synthesis runs.
- On OpenRouter URLs, `deepseek-reasoner` and `deepseek-chat` map to `deepseek/deepseek-r1` and `deepseek/deepseek-chat`.
- Embedding model is auto-selected from `claims_vec` size when recognized (3072 → `google/gemini-embedding-001`, 1536 → `openai/text-embedding-3-small`).
- `misfit_reports` is optional; receipts still runs if no glitch matches are found.
- Markdown output now includes a machine-readable YAML header, a source ledger, and raw evidence appendices by default. Use `--raw-receipts=false` when you only want the polished synthesis plus ledger.
- Source name display mapping lives in `friendlySourceName()` in `receipts/cmd/main.go`.
