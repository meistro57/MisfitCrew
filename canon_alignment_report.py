#!/usr/bin/env python3
"""
canon_alignment_report.py

Analyze meta_reflections to find what aligns across the whole canon.
Produces a Markdown report with three sections:

  1. **Concept Frequency** — which concept tokens appear most often, with
     their cross-source spread (how many distinct source_files mention them).

  2. **Shared Conceptual Regions** — embedding-space clustering of reflection
     vectors. For each cluster, the report shows its dominant concepts, its
     exemplar reflections (closest to centroid), and its source distribution.

  3. **Canonical vs Provincial** — clusters contributed to by many sources
     are "canonical" (shared across the corpus). Clusters dominated by 1-2
     sources are "provincial" (idiosyncratic to those sources).

Usage:
    python canon_alignment_report.py
        -> writes ./reviews/canon_alignment_report.md

    python canon_alignment_report.py --k 40
        -> use 40 clusters instead of default 30

    python canon_alignment_report.py --sample 10000
        -> sample 10k reflections instead of using all (faster)

    python canon_alignment_report.py --collection meta_reflections
        -> override source collection

Requires:
    pip install qdrant-client scikit-learn numpy

Environment:
    QDRANT_URL  (default: http://localhost:6333)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------- config ----

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = "meta_reflections"
DEFAULT_K = 30
DEFAULT_SAMPLE = 0  # 0 = use all
DEFAULT_OUTPUT = Path("./reviews/canon_alignment_report.md")
SCROLL_BATCH = 200
TOP_CONCEPTS = 80          # top-N concepts in the frequency table
EXEMPLARS_PER_CLUSTER = 5  # closest-to-centroid reflections per cluster
CONCEPTS_PER_CLUSTER = 10  # top concept tokens per cluster
CANONICAL_THRESHOLD = 5    # source_files-per-cluster to be "canonical" (default; overridable via --canonical-threshold)


# ---------------------------------------------------------------- load ------

def load_all_reflections(
    client: QdrantClient, collection: str, sample: int
) -> list[dict]:
    """
    Pull reflections with their vectors and the payload fields we need.
    Returns a list of dicts: {id, vector, source_file, concepts, summary, page}.
    """
    reflections: list[dict] = []
    offset = None
    pulled = 0
    fields = ["source_file", "concepts", "summary", "page", "tone"]
    print(f"Loading reflections from {collection!r}...")
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            with_payload=fields,
            with_vectors=True,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            vec = p.vector
            if isinstance(vec, dict):
                # named vectors — grab the first
                vec = next(iter(vec.values()))
            if vec is None:
                continue
            reflections.append({
                "id": str(p.id),
                "vector": np.asarray(vec, dtype=np.float32),
                "source_file": payload.get("source_file", "<unknown>"),
                "concepts": payload.get("concepts", []) or [],
                "summary": payload.get("summary", ""),
                "page": payload.get("page", 0),
                "tone": payload.get("tone", ""),
            })
        pulled += len(points)
        if pulled % 2000 == 0 or offset is None:
            print(f"  ...{pulled} loaded")
        if offset is None:
            break
        if sample and pulled >= sample:
            break
    if sample and len(reflections) > sample:
        idx = np.random.RandomState(42).choice(
            len(reflections), size=sample, replace=False
        )
        reflections = [reflections[i] for i in sorted(idx)]
        print(f"  sampled down to {len(reflections)}")
    print(f"Loaded {len(reflections)} reflections.")
    return reflections


# --------------------------------------------------- normalize concepts -----

def normalize_concept(c: str) -> str:
    """Light normalization: strip, lowercase, collapse whitespace."""
    if not isinstance(c, str):
        return ""
    return " ".join(c.strip().lower().split())


def concept_tokens(concepts_field) -> list[str]:
    """
    Gemma sometimes emits concepts as a list of strings, sometimes as a single
    string with items separated by spaces or commas. Handle both.
    """
    out: list[str] = []
    if isinstance(concepts_field, list):
        for c in concepts_field:
            if not isinstance(c, str):
                continue
            # if the list item looks like a single long string with multiple
            # concepts separated by commas, split it
            parts = [p.strip() for p in c.split(",") if p.strip()]
            if len(parts) > 1:
                out.extend(normalize_concept(p) for p in parts)
            else:
                out.append(normalize_concept(c))
    elif isinstance(concepts_field, str):
        parts = [p.strip() for p in concepts_field.split(",") if p.strip()]
        out.extend(normalize_concept(p) for p in parts)
    return [c for c in out if c and len(c) > 1]


# ----------------------------------------------------------- pass 1 ---------

def concept_frequency_report(
    reflections: list[dict], top_n: int
) -> list[dict]:
    """
    Build a concept-level report: for each normalized concept, how often
    does it appear, and in how many distinct source_files?
    """
    concept_count: Counter[str] = Counter()
    concept_sources: dict[str, set[str]] = defaultdict(set)
    for r in reflections:
        toks = concept_tokens(r["concepts"])
        for tok in toks:
            concept_count[tok] += 1
            concept_sources[tok].add(r["source_file"])

    rows = []
    for concept, count in concept_count.most_common():
        sources = concept_sources[concept]
        rows.append({
            "concept": concept,
            "count": count,
            "n_sources": len(sources),
            "sources": sources,
        })

    # rank by cross-source spread first, then by raw count
    rows.sort(key=lambda r: (-r["n_sources"], -r["count"]))
    return rows[:top_n]


# ----------------------------------------------------------- pass 2 ---------

def cluster_reflections(
    reflections: list[dict], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run MiniBatchKMeans on reflection vectors. Returns (labels, centroids)."""
    print(f"Clustering {len(reflections)} reflections into {k} clusters...")
    X = np.stack([r["vector"] for r in reflections])
    # normalize for cosine-on-kmeans
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        batch_size=1024,
    )
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    return labels, centroids, X


def summarize_cluster(
    cluster_id: int,
    reflections: list[dict],
    labels: np.ndarray,
    centroids: np.ndarray,
    X: np.ndarray,
) -> dict:
    """Extract the human-readable summary for one cluster."""
    mask = labels == cluster_id
    members_idx = np.where(mask)[0]
    members = [reflections[i] for i in members_idx]

    # exemplars: closest reflections to centroid
    centroid = centroids[cluster_id]
    sims = X[members_idx] @ centroid
    order = np.argsort(-sims)
    exemplar_idx = members_idx[order[:EXEMPLARS_PER_CLUSTER]]
    exemplars = [reflections[i] for i in exemplar_idx]
    exemplar_scores = sims[order[:EXEMPLARS_PER_CLUSTER]]

    # concept vocabulary
    concept_count: Counter[str] = Counter()
    for m in members:
        for tok in concept_tokens(m["concepts"]):
            concept_count[tok] += 1
    top_concepts = concept_count.most_common(CONCEPTS_PER_CLUSTER)

    # source distribution
    src_count: Counter[str] = Counter()
    for m in members:
        src_count[m["source_file"]] += 1
    top_sources = src_count.most_common(10)

    return {
        "id": cluster_id,
        "size": len(members),
        "top_concepts": top_concepts,
        "top_sources": top_sources,
        "n_sources": len(src_count),
        "exemplars": list(zip(exemplars, exemplar_scores.tolist())),
    }


# --------------------------------------------------------------- render -----

def render_report(
    collection: str,
    total: int,
    k: int,
    freq_rows: list[dict],
    clusters: list[dict],
    n_source_files: int,
    canonical_threshold: int = CANONICAL_THRESHOLD,
) -> str:
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Canon Alignment Report")
    lines.append("")
    lines.append(
        f"*Generated {ts} from `{collection}` ({total:,} reflections, "
        f"{n_source_files} distinct source files, {k} clusters).*"
    )
    lines.append("")
    lines.append(
        "This report surfaces what aligns across the whole canon. Three views: "
        "concept frequency across sources, embedding-space clustering of "
        "reflection vectors, and canonical-vs-provincial classification of "
        "clusters based on how many source files contribute to each."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- section 1: concept frequency
    lines.append("## 1. Shared Vocabulary")
    lines.append("")
    lines.append(
        "Concept tokens extracted by Gemma during the reflection pass, "
        "normalized and counted across the corpus. **Cross-source spread** "
        "(how many distinct source_files mention the concept) is the "
        "primary signal; raw count is secondary. A concept mentioned in 15 "
        "source files is structurally canonical in a way a concept "
        "mentioned 500 times in one source is not."
    )
    lines.append("")
    lines.append("| Rank | Concept | Sources | Count |")
    lines.append("| ---- | ------- | ------: | ----: |")
    for i, row in enumerate(freq_rows, 1):
        lines.append(
            f"| {i} | {row['concept']} | {row['n_sources']} | {row['count']} |"
        )
    lines.append("")
    lines.append(
        "_Note: concept-token matching is literal string-based. Different "
        "vocabularies for the same idea (e.g. 'logos', 'nous', 'cosmic "
        "intelligence') will not merge here — see Section 2 for the "
        "embedding-based view that catches cross-vocabulary convergence._"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- section 2: clusters
    lines.append("## 2. Shared Conceptual Regions")
    lines.append("")
    lines.append(
        "Reflection vectors clustered in embedding space. Each cluster "
        "represents a region of concept-space the canon concentrates in. "
        "For each cluster: its size, its dominant concept vocabulary, its "
        "source distribution, and a handful of exemplar reflections "
        "(closest to the cluster centroid)."
    )
    lines.append("")

    # group clusters by canonical/provincial
    canonical = [c for c in clusters if c["n_sources"] >= canonical_threshold]
    provincial = [c for c in clusters if c["n_sources"] < canonical_threshold]
    # within each group, sort by size desc
    canonical.sort(key=lambda c: -c["size"])
    provincial.sort(key=lambda c: -c["size"])

    def write_cluster(c: dict, marker: str) -> None:
        lines.append(
            f"### {marker} Cluster {c['id']} — "
            f"{c['size']} reflections, {c['n_sources']} sources"
        )
        lines.append("")
        lines.append("**Top concepts:**")
        lines.append("")
        for tok, n in c["top_concepts"]:
            lines.append(f"- {tok} ({n})")
        lines.append("")
        lines.append("**Source distribution (top 10):**")
        lines.append("")
        lines.append("| Source | Count |")
        lines.append("| ------ | ----: |")
        for src, n in c["top_sources"]:
            lines.append(f"| {src} | {n} |")
        lines.append("")
        lines.append("**Exemplar reflections (closest to centroid):**")
        lines.append("")
        for refl, score in c["exemplars"]:
            summary = (refl["summary"] or "").strip().replace("\n", " ")
            src = refl["source_file"]
            page = refl["page"]
            page_str = f", p.{page}" if page else ""
            lines.append(
                f"- *(score {score:.3f})* **{src}{page_str}** — {summary}"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("### Canonical clusters")
    lines.append("")
    lines.append(
        f"Clusters contributed to by {canonical_threshold}+ distinct source "
        "files. These are the regions of concept-space the canon as a whole "
        "agrees is worth talking about."
    )
    lines.append("")
    for c in canonical:
        write_cluster(c, "🌐")

    lines.append("### Provincial clusters")
    lines.append("")
    lines.append(
        f"Clusters contributed to by fewer than {canonical_threshold} "
        "sources. These are idiosyncratic regions — specific to one or a "
        "few traditions rather than canonical to the whole corpus. Often "
        "the most interesting for tracking what a tradition uniquely "
        "contributes."
    )
    lines.append("")
    for c in provincial:
        write_cluster(c, "📍")

    return "\n".join(lines)


# ----------------------------------------------------------------- main -----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="number of clusters (default 30)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help="sample N reflections for speed; 0 = use all")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--top-concepts", type=int, default=TOP_CONCEPTS)
    ap.add_argument("--canonical-threshold", type=int, default=CANONICAL_THRESHOLD,
                    help=f"min distinct source_files for a cluster to be "
                         f"'canonical' (default {CANONICAL_THRESHOLD}); "
                         f"with a 64-source corpus try 15-20")
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL)

    reflections = load_all_reflections(client, args.collection, args.sample)
    if not reflections:
        print("No reflections loaded.")
        return 1

    n_source_files = len({r["source_file"] for r in reflections})

    # pass 1
    print("Counting concept frequencies...")
    freq_rows = concept_frequency_report(reflections, args.top_concepts)

    # pass 2
    labels, centroids, X = cluster_reflections(reflections, args.k)
    clusters = []
    for cid in range(args.k):
        c = summarize_cluster(cid, reflections, labels, centroids, X)
        clusters.append(c)

    # render
    text = render_report(
        args.collection,
        len(reflections),
        args.k,
        freq_rows,
        clusters,
        n_source_files,
        canonical_threshold=args.canonical_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    print(f"  {len(reflections):,} reflections")
    print(f"  {n_source_files} distinct source files")
    print(f"  {args.k} clusters "
          f"({sum(1 for c in clusters if c['n_sources'] >= args.canonical_threshold)} "
          f"canonical at >={args.canonical_threshold} sources, "
          f"{sum(1 for c in clusters if c['n_sources'] < args.canonical_threshold)} "
          f"provincial)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
