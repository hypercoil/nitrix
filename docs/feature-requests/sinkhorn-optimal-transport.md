# Sinkhorn / optimal transport — `nitrix.transport`

> **Status (2026-06-02): not started.** Brainstorm candidate; promotion
> gated by the §13 acceptance protocol. A new top-level subpackage — §14
> requires a clear substrate-composition story before one is added; the LOG
> semiring supplies it (see below). Provenance: `docs/feature-requests catalogue §12.4`.

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
