# B4. LOG / EUCLIDEAN / BOOLEAN `edge_aggregate` semirings

> **Status (2026-06-02): parked (engineering backlog) — genuinely open
> (verified against live code).** Not a commitment — gated on the
> **Trigger** below. This is the **canonical entry** for the request (also
> surfaced consumer-side as the residual mesh item E in
> [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md)). Provenance:
> migrated from the retired top-level `BACKLOG.md` (B-numbering preserved);
> ledger context in [`internal-backlog.md`](internal-backlog.md).

`semiring_ell_edge_aggregate` supports REAL, TROPICAL_MAX_PLUS,
TROPICAL_MIN_PLUS; LOG / EUCLIDEAN raise `NotImplementedError` with a
specific message (`src/nitrix/semiring/ell_edge.py:143`); BOOLEAN is
excluded (most `edge_fn`s produce floats).

**Trigger.** A consumer with a concrete use case where one composes
naturally — most likely LOG for attention-style softmax aggregation, though
the `ell_row_softmax`-then-REAL composition is usually cleaner than a fused
logsumexp at the edge.

**Notes.** The per-edge message shape is user-controlled, so the algebra's
identity-action on the message must be user-respected (same contract as the
existing semirings). LOG: `logsumexp` over the neighbour axis on
`log(edge_fn_output)`-type messages. EUCLIDEAN: `sqrt(sum_p edge_fn**2)`-shaped.

## Cross-references

- [`internal-backlog.md`](internal-backlog.md) — the engineering-backlog
  ledger.
- [`ilex-pipeline-substrate.md`](ilex-pipeline-substrate.md) — the
  consumer-side surfacing (residual mesh item E; this doc is canonical).
- `src/nitrix/semiring/ell_edge.py:143` — the `NotImplementedError` site;
  `docs/design/semiring-protocols.md`.
