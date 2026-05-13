# adjuster — The Broom

Follows vectoreologist through `vectoreology_findings` and acts on each finding exactly once. Idempotent by design — findings already processed are skipped via the `processed_by_adjuster` flag written back to Qdrant.

---

## What it does

Reads every point in `vectoreology_findings`. For each unswept finding:

| Finding type | Action on `meta_reflections` |
|---|---|
| `cluster` | Tags member points with `cluster_name`, `in_cluster: true` |
| `bridge` | Tags member points with `is_bridge: true`, `bridge_partners` |
| `moat` | Tags isolated points with `needs_review: true` |
| `anomaly` | Tags with `priority_critique: true` |
| `source_contradiction` | Tags with `priority_critique: true`, `has_contradiction: true` |

After acting, writes back to the finding:
```json
{
  "processed_by_adjuster": true,
  "adjuster_action": "tagged 47 meta_reflections as cluster \"Non-Linear Simultaneity\"",
  "adjuster_at": "2026-05-01T...",
  "adjuster_version": "dev"
}
```

---

## Install

```bash
cd ~/MisfitCrew/adjuster
ln -s ~/MisfitCrew/.env .env
go mod tidy
go build -o adjuster ./cmd
```

---

## Usage

```bash
# run once
./adjuster

# dry run — see what it would do without writing
./adjuster --dry-run

# watch mode — sweep every 90 seconds alongside vectoreologist
./adjuster --watch 90

# custom collections
./adjuster --findings vectoreology_findings --reflections meta_reflections
```

---

## Watch mode with vectoreologist

Run both together and the broom follows the archaeologist:

```bash
# terminal 1
./vectoreologist --collection meta_reflections --watch 90 --semantic-labels

# terminal 2
./adjuster --watch 90
```

Or Lewis fires both.

---

## What gets enriched in meta_reflections

After adjuster runs, `meta_reflections` points gain:

```json
{
  "cluster_name": "Non-Linear Simultaneity",
  "in_cluster": true,
  "is_bridge": true,
  "bridge_partners": ["Consciousness-Mediated Reality", "Channeling Methodology"],
  "needs_review": false,
  "priority_critique": false,
  "has_contradiction": false
}
```

Receipts can now filter by `cluster_name` for smarter source selection.
MisfitCrew can prioritize `priority_critique: true` points.
Lewis can query "show me all bridge points" instantly.

---

## Pipeline position

```
vectoreology_findings  ←  vectoreologist (the archaeologist)
        ↓
    adjuster           ←  YOU ARE HERE (the broom)
        ↓
meta_reflections       ←  enriched with cluster/bridge/anomaly tags
        ↓
receipts + misfit_crew ←  smarter, topology-aware
```
