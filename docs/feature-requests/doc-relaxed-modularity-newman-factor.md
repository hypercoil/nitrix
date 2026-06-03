# Doc-fix: `relaxed_modularity` does not reduce to Newman modularity (½ factor)

> **Status (2026-06-03): open — documentation-correctness fix (one line, no
> behaviour change).** Provenance: surfaced matching `relaxed_modularity` to
> `networkx.algorithms.community.modularity` for a `nitrix-perf-bench` case;
> ledger context in [`perf-bench-feedback.md`](perf-bench-feedback.md).

`src/nitrix/graph/community.py:245` (the `relaxed_modularity` docstring) states
that *"For a one-hot ``C`` this reduces to the standard Newman modularity."*
The actual reduction carries a **½** factor: for a hard one-hot partition,
`relaxed_modularity(A, C, exclude_diag=False)` equals the canonical Newman
quality score **divided by 2**, not the score itself.

**Root cause (the double-count is corrected twice).** The dense path is
`Q = (B * CCᵀ).sum() / 2` with `B = (A - γ·kkᵀ/2m) / 2m`. The literature Newman
modularity is `Q_N = (1/2m) Σ_ij (A_ij - k_i k_j/2m) δ(c_i,c_j)` — the `1/2m`
prefactor *already* accounts for the undirected double-count of the ordered
sum. nitrix applies `1/2m` **and** an explicit `Q/2` (the `directed=False`
branch), so it halves the literature value. Harmless for a differentiable loss
(a constant scale leaves the gradient direction and the argmax partition
unchanged — which is presumably why it was never caught), but the docstring's
claim of reducing to *the standard* Newman modularity is inaccurate.

**Verified (fp64, this checkout).** Across 6 seeds × {(16,2),(32,4),(64,5),
(50,3)} (n,k) × γ ∈ {0.5, 1.0, 1.7}, `relaxed_modularity(A, C,
exclude_diag=False)` == `networkx ...modularity(G, weight='weight',
resolution=γ) / 2` to **rel < 1e-9** in every configuration. Without the `/2`
the gap is a clean factor of 2.

A second, separate gap: the **default** `exclude_diag=True` additionally drops
the within-community diagonal term `Σ_i B_ii` (negative for a self-loop-free
graph), so the default is neither `Q_N` nor `Q_N/2` — it is the off-diagonal
restriction. The docstring's "reduces to the standard Newman modularity" reads
as describing the default call; it holds (up to the ½) only for
`exclude_diag=False`.

**Fix (docstring only, two clarifications).** On
`src/nitrix/graph/community.py:245`:
1. state the factor — *"reduces to **half** the standard Newman modularity
   (`Q_Newman / 2`); the undirected double-count is corrected by both the
   `1/2m` normalisation and the `directed=False` `Q/2`"*; and
2. scope it to `exclude_diag=False` (the default `exclude_diag=True` drops the
   within-community diagonal, so it equals neither).

Optionally (behaviour change, separate decision) drop the redundant `Q/2` when
`normalise_modularity=True` so the score matches the literature `Q_N` exactly;
this is *not* proposed here because it would silently rescale every existing
modularity loss. The benchmark uses the verified `Q_N / 2` bridge, so coverage
does not depend on the fix.

## Cross-references

- [`perf-bench-feedback.md`](perf-bench-feedback.md) — the perf-bench-surfaced
  doc-drift ledger.
- [`doc-gaussian-kernel-gamma.md`](doc-gaussian-kernel-gamma.md) — the sibling
  ½-factor docstring drift (also surfaced by a perf-bench reference match).
- `src/nitrix/graph/community.py:245` (`relaxed_modularity`); nitrix-perf-bench
  `relaxed_modularity` case (the `networkx.modularity` baseline is the bridged
  reference).
