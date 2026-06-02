# nitrix-perf-bench feedback — ledger & index

> **This doc is now the doc-drift *ledger + index*.** Each finding has been
> atomised into its own tracking doc (one doc per fix, to reduce
> duplicate-issue risk); this file keeps the framing and the index. See
> [`README.md`](README.md) for the directory-wide index.

Documentation / definition drift and consumer-facing gaps surfaced while
building benchmark cases in `nitrix-perf-bench` (the perf migration of the op
matrix; see DESIGN there). Each atomised entry cites file:line and the
measurement that surfaced it so the fix is mechanical. Perf *numbers* live in
the perf-bench `COVERAGE_DEFICIT` report; these entries are for
**correctness-of-documentation** findings only — they are doc fixes, not
primitive proposals.

## Open (atomised)

Surfaced 2026-06-02 while building perf-bench cases; verified against
`scipy 1.17.1` + from-scratch fp64 references on this checkout.

| Finding | Doc | Site | Priority |
|---|---|---|---|
| `lomb_scargle_periodogram` normalisation claim is wrong | [doc-lomb-scargle-normalisation](doc-lomb-scargle-normalisation.md) | `signal/lomb_scargle.py:154` | high-value |
| module docstring says "Cholesky"; code uses `eigh` + pseudo-inverse | [doc-lomb-scargle-eigh-factorisation](doc-lomb-scargle-eigh-factorisation.md) | `signal/lomb_scargle.py:43–49` | normal |
| `lomb_scargle_interpolate` silently runs eigh on CPU (cuSolver-broken stacks) | [doc-lomb-scargle-cpu-eigh-caveat](doc-lomb-scargle-cpu-eigh-caveat.md) | `linalg/_solver.py:147` | normal |
| `tsconv` documented as "convolution" but is cross-correlation | [doc-tsconv-cross-correlation](doc-tsconv-cross-correlation.md) | `signal/tsconv.py:45` | low (clarity) |
| `lomb_scargle_interpolate` intended-use (spectral bridge, not durable imputation) | [doc-lomb-scargle-interpolate-intended-use](doc-lomb-scargle-interpolate-intended-use.md) | `signal/lomb_scargle.py:~264–359` | normal |
| `tangent_project_spd` missing from op_matrix inventory (public op; blocks perf-bench coverage credit) | [doc-op-matrix-missing-tangent](doc-op-matrix-missing-tangent.md) | `docs/op_matrix.json` (`ops`) | normal |

## Resolved

_(none yet; reference the resolving nitrix commit on fix.)_
