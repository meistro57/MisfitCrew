#!/usr/bin/env bash
# filename: run_all_reports.sh
# MisfitCrew — fire all reports into one place
# Runs: ROOTreviews, canon alignment, and receipts for a concept list
# Output lands in ~/MisfitCrew/ALL_REPORTS/

set -euo pipefail

CREW_DIR="$HOME/MisfitCrew"
OUT_DIR="$CREW_DIR/ALL_REPORTS"
RECEIPTS_BIN="$CREW_DIR/receipts/receipts"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; AMBER='\033[0;33m'; CYAN='\033[0;36m'; RESET='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓${RESET} $*"; }
warn() { echo -e "${AMBER}  ⚠${RESET} $*"; }

# ── parse flags ───────────────────────────────────────────────────────────────
SKIP_REVIEWS=false
SKIP_CANON=false
SKIP_RECEIPTS=false
CONCEPTS_FILE=""   # optional: path to a file with one concept per line
CANON_K=30

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --skip-reviews      Skip ROOTreviews generation
  --skip-canon        Skip canon alignment report
  --skip-receipts     Skip receipts generation
  --concepts FILE     Path to a file with one concept per line
                      (default: uses the built-in concept list below)
  --canon-k N         Number of clusters for canon report (default 30)
  -h, --help          Show this help

Output: $OUT_DIR/
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-reviews)   SKIP_REVIEWS=true  ;;
    --skip-canon)     SKIP_CANON=true    ;;
    --skip-receipts)  SKIP_RECEIPTS=true ;;
    --concepts)       CONCEPTS_FILE="$2"; shift ;;
    --canon-k)        CANON_K="$2";       shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown flag: $1"; usage; exit 1 ;;
  esac
  shift
done

# ── default concept list ──────────────────────────────────────────────────────
# Edit this list freely — one concept per line
DEFAULT_CONCEPTS=(
  "consciousness"
  "non-duality"
  "death as transition"
  "belief creates reality"
  "time is non-linear"
  "the body as a temporary vehicle"
  "reincarnation"
  "free will"
  "the nature of the soul"
  "light as the primary substance of reality"
  "simulation theory"
  "synchronicity"
)

# ── setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR/reviews"
mkdir -p "$OUT_DIR/receipts"
mkdir -p "$OUT_DIR/canon"

cd "$CREW_DIR"

# activate venv if present
if [[ -d "$CREW_DIR/.venv" ]]; then
  source "$CREW_DIR/.venv/bin/activate"
elif [[ -d "$CREW_DIR/venv" ]]; then
  source "$CREW_DIR/venv/bin/activate"
fi

if [[ -f "$CREW_DIR/.env" ]]; then
  set +u
  set -a
  source "$CREW_DIR/.env"
  set +a
  set -u
fi

echo ""
log "MisfitCrew — All Reports Run  |  $TIMESTAMP"
log "Output root: $OUT_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROOTreviews — one markdown per source book
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_REVIEWS" == false ]]; then
  log "Step 1/3 — Pulling ROOTreviews (all sources → $OUT_DIR/reviews/) ..."
  python misfit_report_pull.py --all --out "$OUT_DIR/reviews"
  COUNT=$(ls "$OUT_DIR/reviews/"*.md 2>/dev/null | wc -l)
  ok "ROOTreviews done — $COUNT files written"
else
  warn "Skipping ROOTreviews"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Canon alignment report
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_CANON" == false ]]; then
  CANON_OUT="$OUT_DIR/canon/canon_alignment_report_${TIMESTAMP}.md"
  log "Step 2/3 — Canon alignment report (k=$CANON_K) → $CANON_OUT ..."
  python canon_alignment_report.py \
    --k "$CANON_K" \
    --out "$CANON_OUT"
  ok "Canon alignment report done"
else
  warn "Skipping canon alignment report"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. Receipts — one synthesis doc per concept
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_RECEIPTS" == false ]]; then
  log "Step 3/3 — Generating receipts → $OUT_DIR/receipts/ ..."

  # build concept array
  if [[ -n "$CONCEPTS_FILE" ]]; then
    mapfile -t CONCEPTS < "$CONCEPTS_FILE"
    log "  Loaded $(( ${#CONCEPTS[@]} )) concepts from $CONCEPTS_FILE"
  else
    CONCEPTS=("${DEFAULT_CONCEPTS[@]}")
    log "  Using built-in concept list (${#CONCEPTS[@]} concepts)"
  fi

  if [[ ! -x "$RECEIPTS_BIN" ]]; then
    warn "receipts binary not found at $RECEIPTS_BIN — attempting rebuild..."
    pushd "$CREW_DIR/receipts" > /dev/null
    ./rebuild.sh
    popd > /dev/null
  fi

  RECEIPTS_OK=0
  RECEIPTS_FAIL=0

  for concept in "${CONCEPTS[@]}"; do
    [[ -z "$concept" || "$concept" == \#* ]] && continue   # skip blanks/comments
    log "  Generating receipt: \"$concept\" ..."
    if "$RECEIPTS_BIN" \
        --concept "$concept" \
        --out "$OUT_DIR/receipts" \
        2>&1; then
      ok "  \"$concept\" done"
      (( RECEIPTS_OK++ )) || true
    else
      warn "  \"$concept\" FAILED (check output above)"
      (( RECEIPTS_FAIL++ )) || true
    fi
    # small pause between R1 calls to be kind to rate limits
    sleep 2
  done

  ok "Receipts done — $RECEIPTS_OK ok, $RECEIPTS_FAIL failed"
else
  warn "Skipping receipts"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  All reports complete — $TIMESTAMP${RESET}"
echo -e "${GREEN}═══════════════════════════════════════════════════${RESET}"
echo ""
echo "  Reviews:  $OUT_DIR/reviews/"
echo "  Canon:    $OUT_DIR/canon/"
echo "  Receipts: $OUT_DIR/receipts/"
echo ""
ls -lh "$OUT_DIR/reviews/" 2>/dev/null | tail -5 || true
