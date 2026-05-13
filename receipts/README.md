# receipts

**The "Look at this!!" engine.**

Takes a concept and generates a plain-English multi-tradition synthesis document showing that people from completely different cultures, centuries, and backgrounds all independently arrived at the same idea — with receipts.

Part of the MisfitCrew pipeline. Sits downstream of `misfit_crew.py` and `meta_reflections`.

---

## What it does

1. Auto-detects embedding model from `meta_reflections` vector size (if available), then embeds your concept query via OpenRouter
2. Searches `meta_reflections` for the best matching chunks across distinct source traditions
3. Searches `misfit_reports` for Hardware Glitch receipts on the same material
4. Sends everything to DeepSeek R1 to write a plain-English synthesis doc
5. Outputs Markdown to `./output/<concept>_<timestamp>.md`

Output format:
- **The Big Idea** — plain English, 2-3 sentences
- **The Receipts** — one paragraph per source tradition with the specific claim
- **Why This Is Wild** — why the cross-tradition convergence is remarkable
- **The Weird Part** — the Hardware Glitch, the place it contradicts itself
- **What It Means For You** — practical takeaway

---

## Install

```bash
cd ~/MisfitCrew/receipts
go mod tidy
go build -o receipts ./cmd
go test ./...
```

---

## Usage

```bash
# See what concepts are available
./receipts --list-concepts

# Generate a receipts doc
./receipts --concept "belief creates reality"

# More sources, fast mode (no R1 chain-of-thought)
./receipts --concept "consciousness survives death" --sources 8 --model deepseek-chat

# Alias for sources count
./receipts --concept "consciousness survives death" --source-traditions 10

# Custom output dir
./receipts --concept "non-linear time" --out ~/MisfitCrew/receipts/output
```

---

## Environment (.env or shell)

```env
QDRANT_URL=http://localhost:6333
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...          # only needed if pointing directly at DeepSeek
DEEPSEEK_CHAT_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-reasoner
EMBED_MODEL=google/gemini-embedding-001 # fallback when collection model can't be inferred
SOURCE_TRADITIONS_COUNT=6              # default when --sources is not passed
REFLECTIONS_COLLECTION=meta_reflections
REPORTS_COLLECTION=misfit_reports
```

---

## Pipeline position

```
mb_chunks
    ↓ reflect_loop_fixed.py
meta_reflections
    ↓ misfit_crew.py
misfit_reports
    ↓
receipts  ← YOU ARE HERE
    ↓
output/<concept>.md  →  QMU forum / ROOT ACCESS / Discord / blog
```

---

## Notes

- `--list-concepts` scrolls the entire `meta_reflections` collection and ranks concepts by cross-source spread. Takes ~30 seconds on 3k reflections.
- R1 synthesis takes 3-5 minutes. Use `--model deepseek-chat` for fast drafts.
- Embedding model is auto-selected from the reflections collection `claims_vec` size when recognized (3072 → `google/gemini-embedding-001`, 1536 → `openai/text-embedding-3-small`).
- `misfit_reports` is optional — if empty, receipts still runs, just without the Hardware Glitch section populated from the critic layer.
- Source names are mapped to human-readable labels in `friendlySourceName()` — add new sources there as you ingest them.
