#!/usr/bin/env python3
"""
misfit_crew.py — R1 mines meta_reflections; Gemma4 critiques. Results land in misfit_reports.

Changes from previous version:
- Mined-ID set lives in memory, built ONCE at startup by scrolling misfit_reports.
  No more reading the JSON ledger on every iteration.
- Source-side scroll with client-side exclusion (fast) instead of HasIdCondition
  filter with a giant must_not list (slow; degrades as N grows).
- Proper failure tracking: misfit_failures.json holds per-ID error history with
  attempt counts. After MAX_ATTEMPTS retries a chunk is marked dead and skipped.
- Retry with exponential backoff on DeepSeek API (rate limits and transient 5xx).
- Progress counter and ETA matching reflect.py's format.
- JSON ledger (misfit_ledger.json) kept as append-only audit log only — NOT used
  as the read path for exclusion decisions.
- --limit, --sleep, --max-attempts flags added.
- Switched embedding to google/gemini-embedding-001 via OpenRouter (3072-dim).
- misfit_reports now uses dual named vectors: claims_vec + summary_vec (Cosine).

Pipeline per well:
  1. pull next un-mined reflection from meta_reflections
  2. DeepSeek R1 → reasoning + report ("Hardware Glitches and Ontological Shock")
  3. Gemma4 critic → verdict on the report
  4. Gemini embed report → claims_vec, verdict → summary_vec via OpenRouter
  5. upsert to misfit_reports with matching point_id (lineage preserved)
  6. append to audit ledger
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# --- CONFIG ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

SOURCE_COLLECTION = "meta_reflections"
DEST_COLLECTION = "misfit_reports"
LEDGER_FILE = "misfit_ledger.json"
FAILURES_FILE = "misfit_failures.json"

CRITIC_MODEL = "gemma4:latest"
EMBED_MODEL = "google/gemini-embedding-001"
EMBED_DIM = 3072

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_SLEEP_BETWEEN = 1.0
DEFAULT_SCROLL_BATCH = 128

# Backoff schedule for DeepSeek retries (seconds): 2, 8, 30
BACKOFF_SCHEDULE = [2, 8, 30]

client = QdrantClient(QDRANT_URL)

STOP = False


def handle_sigint(signum, frame):
    global STOP
    if STOP:
        print("\n[!] second ctrl-c; exiting hard")
        sys.exit(130)
    STOP = True
    print("\n[!] ctrl-c caught, finishing current well then stopping...")


# ------------------------------------------------------------- qdrant setup --

def ensure_collections():
    existing = {c.name for c in client.get_collections().collections}
    if DEST_COLLECTION not in existing:
        print(f"🏗️  Initializing {DEST_COLLECTION} (dual 3072-dim named vectors, Cosine)...")
        client.create_collection(
            collection_name=DEST_COLLECTION,
            vectors_config={
                "claims_vec": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
                "summary_vec": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
            },
        )


def load_mined_ids() -> set[str]:
    """Scroll misfit_reports ONCE at startup; return set of all existing point IDs."""
    mined: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=DEST_COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            mined.add(str(p.id))
        if offset is None:
            break
    return mined


def count_source() -> int:
    return client.count(collection_name=SOURCE_COLLECTION, exact=True).count


# --------------------------------------------------------- failure tracking --

def load_failures() -> dict:
    if os.path.exists(FAILURES_FILE):
        try:
            with open(FAILURES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_failures(failures: dict) -> None:
    tmp = FAILURES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(failures, f, indent=2)
    os.replace(tmp, FAILURES_FILE)


def record_failure(failures: dict, point_id: str, error: str) -> int:
    """Record a failure; returns the new attempt count."""
    entry = failures.get(point_id, {"attempts": 0, "errors": []})
    entry["attempts"] = entry.get("attempts", 0) + 1
    entry["last_error"] = error
    entry["last_attempt"] = datetime.now().isoformat()
    errs = entry.get("errors", [])
    errs.append({"error": error, "at": entry["last_attempt"]})
    entry["errors"] = errs[-5:]  # keep last 5 only
    failures[point_id] = entry
    save_failures(failures)
    return entry["attempts"]


def clear_failure(failures: dict, point_id: str) -> None:
    if point_id in failures:
        del failures[point_id]
        save_failures(failures)


# --------------------------------------------------------- source iteration --

def iter_source_unmined(mined: set[str], dead: set[str]):
    """Scroll meta_reflections in pages, yield points that aren't mined or dead."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=DEFAULT_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pid = str(p.id)
            if pid in mined or pid in dead:
                continue
            yield p
        if offset is None:
            break


# ----------------------------------------------------------- model adapters --

HGOS_CRITIC_PROMPT = """IMPORTANT — WHAT YOU ARE EVALUATING

You are evaluating whether the REPORTER'S CLAIMS ABOUT THE SOURCE MATERIAL are correct. You are NOT evaluating whether the report is well-written, whether its jargon is sophisticated, whether it reads like a piece of academic prose, or whether the framework it uses is rigorous.

The reporter's job was to identify Hardware Glitches (logical errors, fallacies, conceptual breakdowns) and Ontological Shock (paradigm-disrupting implications) in the source claims. Your job is to test whether the reporter SUCCEEDED at that.

Do not write meta-critique. Do not say "this is an analysis of an analysis." Do not praise the report's sophistication or rigor. Do not credential the report's academic style.

Instead, ask:
- Did the reporter correctly identify hardware glitches in the source claims, or did they miss real ones / invent fake ones?
- Did the reporter correctly identify ontological shocks in the source claims, or did they miss real ones / overstate weak ones?
- Are the reporter's claims about the source MATERIALLY CORRECT? When the reporter says "Claim X commits a category error," is that actually true of Claim X, or is the reporter wrong?

When you find errors, name them concretely. When the reporter is correct, confirm specifically what they got right. When the reporter is partially correct, identify exactly which parts are sound and which aren't.

You may be terse. You may be blunt. Praise is reserved for genuine analytical accuracy, not for style.

Evaluate the report below using the exact structure:

I. Logical Consistency
- Logical Consistency: Are the reporter's CLAIMS ABOUT THE SOURCE internally coherent? Does the reporter avoid fallacies in their own reasoning about the source? When the reporter accuses the source of a fallacy, is the accusation correct?
- Give concrete examples from the report and source-claim handling.

II. Reality Engineering Validity
- Reality Engineering Validity: Do the reporter's claims about the source identify real mechanisms vs. real metaphors? Are the reporter's hardware-glitch callouts genuinely glitches in the source material, or are they pseudo-glitches the reporter invented? Are the ontological-shock callouts genuinely paradigm-disrupting, or are they everyday claims dressed up as shock?
- Name specific source-claim assessments that are correct, incorrect, or mixed.

III. Final Verdict
- State whether the reporter mostly succeeded or failed at accurately evaluating the source material.

JSON rubric (include this exact key schema at the end):
{
  "logical_consistency_score": 0.0,
  "re_validity_score": 0.0,
  "drift_score": 0.0,
  "notes": "brief rationale tied to source-claim accuracy"
}

Scoring guidance:
- logical_consistency_score: 0.0 (invalid/contradictory reasoning about source claims) to 1.0 (consistently sound reasoning about source claims)
- re_validity_score: 0.0 (misidentifies mechanisms, glitches, or shocks) to 1.0 (accurately distinguishes mechanisms/metaphors and valid glitch/shock callouts)
- drift_score: 0.0 (cleanly evaluates report claims about source) to 1.0 (drifts into meta-critique of writing style)
- drift_score calibration rule: if your verdict stays on source-claim accuracy and avoids writing/style meta-commentary, drift_score must be between 0.0 and 0.2. Set drift_score above 0.3 only if you actually drifted into style/academic/meta framing.

IMPORTANT REPEAT: do not frame this as "analysis of an analysis." Do not evaluate writing quality or sophistication. Evaluate source-claim accuracy.

REPORT TO CRITIQUE:
"""


async def openrouter_embed(text: str) -> Optional[list[float]]:
    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def deepseek_analyze(payload: dict, api_key: str) -> tuple[str, str]:
    """Returns (reasoning, report). Retries with backoff on 429 / 5xx."""
    claims = "\n- ".join(payload.get("claims", []) or [])
    concepts = ", ".join(payload.get("concepts", []) or [])
    echoes = ", ".join(payload.get("echoes", []) or [])

    prompt = (
        f"### RAW DATA\n"
        f"CLAIMS: {claims}\n"
        f"CONCEPTS: {concepts}\n"
        f"ECHOES: {echoes}\n\n"
        f"Analyze for Hardware Glitches and Ontological Shock."
    )

    last_exc = None
    for attempt, wait in enumerate([0] + BACKOFF_SCHEDULE):
        if wait:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=180.0) as ds:
                resp = await ds.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-reasoner",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if resp.status_code in (429,) or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable {resp.status_code}",
                        request=resp.request, response=resp,
                    )
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                return (
                    msg.get("reasoning_content", "") or "",
                    msg.get("content", "") or "",
                )
        except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
            last_exc = e
            continue
    raise RuntimeError(f"DeepSeek failed after retries: {last_exc}")


async def gemma_critique(report: str) -> str:
    prompt = f"{HGOS_CRITIC_PROMPT}\n{report}"
    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(
            OLLAMA_GEN_URL,
            json={"model": CRITIC_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "⚠️ Critic offline.")


# --------------------------------------------------------- publish + ledger --

async def publish(point_id: str, source: str, reasoning: str, report: str, verdict: str) -> bool:
    print(f"📡 Vectorizing with {EMBED_MODEL} via OpenRouter...")
    claims_vector = await openrouter_embed(report)
    summary_vector = await openrouter_embed(verdict)
    if not claims_vector or not summary_vector:
        return False
    client.upsert(
        collection_name=DEST_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "claims_vec": claims_vector,
                    "summary_vec": summary_vector,
                },
                payload={
                    "source_file": source,
                    "reasoning": reasoning,
                    "report": report,
                    "verdict": verdict,
                    "mined_at": datetime.now().isoformat(),
                },
            )
        ],
    )
    return True


def append_ledger(point_id: str, source: str, reasoning: str, report: str, verdict: str) -> None:
    """Audit log only. Append-only. Never read as source of truth for exclusion."""
    entry = {
        "id": point_id,
        "source": source,
        "reasoning": reasoning,
        "report": report,
        "verdict": verdict,
        "timestamp": datetime.now().isoformat(),
    }
    # append-by-rewrite is fine at this scale; swap for JSONL if you ever go big
    ledger = []
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r") as f:
                ledger = json.load(f)
        except (json.JSONDecodeError, OSError):
            ledger = []
    ledger.append(entry)
    tmp = LEDGER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, LEDGER_FILE)


# ------------------------------------------------------------------ runner ---

async def process_one(target, failures: dict, max_attempts: int, api_key: str) -> str:
    """Returns 'ok', 'fail', or 'dead'."""
    pid = str(target.id)
    source = target.payload.get("source_file", "unknown")
    print(f"\n📍 Digging Well: {pid} | {source}")

    try:
        reasoning, report = await deepseek_analyze(target.payload, api_key)
        if not report.strip():
            raise RuntimeError("DeepSeek returned empty report")
        verdict = await gemma_critique(report)
        ok = await publish(pid, source, reasoning, report, verdict)
        if not ok:
            raise RuntimeError("embedding failed — report not published")
        append_ledger(pid, source, reasoning, report, verdict)
        clear_failure(failures, pid)
        print("✅ Well capped and vector-indexed.")
        return "ok"
    except Exception as e:
        attempts = record_failure(failures, pid, f"{type(e).__name__}: {e}")
        if attempts >= max_attempts:
            print(f"☠️  Well declared dead after {attempts} attempts: {e}")
            return "dead"
        print(f"⚠️  Well failed (attempt {attempts}/{max_attempts}): {e}")
        return "fail"


async def run_conductor(args):
    global STOP

    if not DEEPSEEK_API_KEY:
        print("[fatal] DEEPSEEK_API_KEY not set in env or .env file")
        return 1

    print(f"🚀 Misfit Crew: Starting Full-Cycle Archaeology Loop...")
    print(f"[config] qdrant={QDRANT_URL} max_attempts={args.max_attempts} sleep={args.sleep}s")

    ensure_collections()

    print("[init] scanning misfit_reports for already-mined IDs...")
    mined = load_mined_ids()
    print(f"[init] {len(mined)} wells already capped")

    failures = load_failures()
    dead = {pid for pid, entry in failures.items()
            if entry.get("attempts", 0) >= args.max_attempts}
    if dead:
        print(f"[init] {len(dead)} wells marked dead (skipped); clear misfit_failures.json to retry")

    total_source = count_source()
    remaining = total_source - len(mined) - len(dead)
    print(f"[plan]  {total_source} reflections, {remaining} eligible to mine")

    if args.limit > 0:
        remaining = min(remaining, args.limit)
        print(f"[plan]  --limit {args.limit} applied")

    processed = 0
    successes = 0
    fails = 0
    deaths = 0
    t0 = time.time()

    for target in iter_source_unmined(mined, dead):
        if STOP:
            break

        result = await process_one(target, failures, args.max_attempts, DEEPSEEK_API_KEY)
        processed += 1
        if result == "ok":
            successes += 1
            mined.add(str(target.id))
        elif result == "dead":
            deaths += 1
            dead.add(str(target.id))
        else:
            fails += 1

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (remaining - processed) / rate if rate > 0 else 0
        print(f"[{processed}/{remaining}] ok={successes} fail={fails} dead={deaths}  "
              f"({rate*60:.1f}/min, eta {eta/60:.0f}m)")

        if args.limit > 0 and processed >= args.limit:
            print("[plan] --limit reached")
            break

        await asyncio.sleep(args.sleep)

    elapsed = time.time() - t0
    print(f"\n[done] {processed} wells processed in {elapsed/60:.1f}m")
    print(f"       {successes} ok, {fails} transient fails, {deaths} declared dead")
    if fails or deaths:
        print(f"       see {FAILURES_FILE} for details")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Misfit Crew — R1 mines, Gemma critiques")
    parser.add_argument("--limit", type=int, default=0, help="process at most N wells")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN,
                        help=f"seconds between wells (default {DEFAULT_SLEEP_BETWEEN})")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help=f"retries per well before declaring it dead (default {DEFAULT_MAX_ATTEMPTS})")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)
    sys.exit(asyncio.run(run_conductor(args)))


if __name__ == "__main__":
    main()