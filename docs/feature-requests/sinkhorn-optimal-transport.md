# Sinkhorn / optimal transport — `nitrix.transport`

> **Status (2026-07-08): SHIPPED (`nitrix.transport`).** `sinkhorn` (entropic
> OT plan + dual potentials), `wasserstein_distance` (the plan's transport cost),
> `barycentric_map` (the OT pushforward map). The new top-level subpackage clears
> the §14 bar via the promised substrate-composition story: each Sinkhorn
> half-step is a log-domain softmin against the cost, computed as a **`LOG`
> semiring matmul** (`semiring_matmul`) — no new kernel, inheriting the semiring's
> streaming reduction; the loop is fixed-iteration (differentiable) and stays in
> the log domain (stable for small `epsilon`). Verified: marginals recovered,
> plan non-negative, entropic cost approaches the exact linear-assignment OT as
> `epsilon → 0` (<5% at `epsilon=0.002`), self-transport ~0; jit/grad clean.
> Transport *distances* are numerical primitives; a transport *loss* stays
> downstream. Provenance: `docs/feature-requests catalogue §12.4`.

**What.** Entropic optimal transport via Sinkhorn iteration — the flagship
use case for the `LOG` semiring.

**Proposed surface.**

```python
def sinkhorn(cost, mu, nu, *, eps, n_iter): ...        # entropic OT plan
def wasserstein_distance(...): ...                      # derived distance
def barycentric_map(plan, x_source, y_target): ...      # pushforward map
```

**Composition.** Sinkhorn is the log-domain matmul against a cost matrix
plus alternating row/column normalisation, converging to the entropic OT
plan — exactly the streaming-kernel substrate (`semiring.LOG`,
`semiring_matmul`; KeOps's original demo). The implicit-differentiation
story for the OT plan is well-established (Cuturi 2013; Feydy 2020).

**Likely consumer.** Shape matching (mesh correspondence), distribution
alignment in fMRI ICA, transport-based segmentation, domain-adaptation
losses for medical-image pretraining.

**Effort.** M. Reuses the already-declared `LOG` semiring path.

**Live-code status.** No `nitrix.transport` subpackage. The `LOG` semiring
and `semiring_matmul` it would build on are shipped
(`semiring/__init__`: `LOG`, `semiring_matmul`, `reference_semiring_matmul`).

## Cross-references

- `docs/feature-requests catalogue §12.4` — origin entry; `§13` — acceptance protocol;
  `§14` — new-subpackage bar.
- `src/nitrix/semiring/algebras.py` — the `LOG` algebra.
- [`docs/design/streaming-kernel.md`](../design/streaming-kernel.md) — the
  substrate it lowers to.
