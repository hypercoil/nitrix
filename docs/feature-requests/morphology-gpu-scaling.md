# Morphology GPU scaling: connectivity lags cupyx; EDT wants a scale-aware dispatch

> **Status (2026-06-10): perf characterisation, profile/algorithm-gated.**
> Surfaced by `nitrix-perf-bench` benching the morphology ops across scale on an
> L4 (nitrix-jax vs cupyx, `jax-cuda12`; see that repo's morphology cases +
> `reports/REGISTRATION_SCALING.md` methodology). Two items, one primary.

## 1. (Primary) `connected_components` / `largest_connected_component` scale poorly vs cupyx

The pointer-jumping label propagation (`morphology/_label.py`: a `lax.while_loop`
of a neighbour-max hop + an `L = L[L-1]` pointer-jump per pass, O(log d) passes
for diameter d) is correct, jit-able, and a real improvement over an O(d) flood
-- but **it is far slower than `cupyx.scipy.ndimage.label` at brain scale, and
the gap widens steeply**:

| op | 48³ | 96³ | 128³ | 160³ |
|---|---|---|---|---|
| `connected_components` nitrix-GPU | 1.30 ms | 1.72 ms | 5.75 ms | **15.7 ms** |
| cupyx `label` | 0.64 ms | 0.48 ms | 0.62 ms | **0.85 ms** |
| → cupyx ahead | ~2× | ~3.6× | ~9× | **~18×** |
| `largest_connected_component` nitrix-GPU | 1.37 ms | 2.17 ms | 7.11 ms | **17.7 ms** |
| cupyx (label+argmax) | 0.72 ms | 0.79 ms | 0.92 ms | **1.14 ms** |
| → cupyx ahead | ~1.9× | ~2.7× | ~7.7× | **~15×** |

nitrix's steady **grows ~12-13× over 48³→160³** while cupyx stays **~flat**
(~0.6→0.85 ms, near volume-independent in this range). So the per-pass cost
(each pass is an O(N) neighbour-max + pointer-jump over the whole label image,
× the pass count) dominates and does not amortise the way cupyx's labeller does.
The implementation note (`connected-components.md`) characterised the algorithm
against an O(d) flood; **this characterises it against the GPU state of the art
(cupyx), where there is a large and widening gap** -- a kernel/algorithm
candidate (a more work-efficient GPU connected-components, e.g. a
Playne-Equivalence / block-based label-equivalence scheme, or a Pallas kernel).

## 2. (Secondary, lower-priority) `distance_transform_edt`: a scale-aware semiring↔F-H dispatch

`distance_transform_edt` is a thin alias for `distance_transform(metric=
'euclidean')` -- the separable **min-plus semiring** EDT (it searches over *all*
parabolas, not the Felzenszwalb-Huttenlocher lower-envelope). That is shallow +
high-FLOP, so on a parallel GPU it **wins at small scale (depth-bound) and loses
at large scale (FLOP-bound)** vs the F-H-class refs -- the **known, deliberate
semiring trade-off** (the EDT family's original motivation; this is *not* a
regression):

| `distance_transform_edt` | 48³ | 96³ | 128³ | 160³ |
|---|---|---|---|---|
| nitrix-GPU (semiring) | **0.15 ms** | 0.37 ms | 0.48 ms | 1.99 ms |
| cupyx (F-H) | 0.36 ms | 0.31 ms | 0.50 ms | **0.89 ms** |
| → nitrix vs cupyx | **2.4× ahead** | ~par | ~par | **2.2× behind** |

The crossover is ~96³. **A scale-aware dispatch** -- keep the semiring brute
force below the crossover (where it wins on the parallel GPU) and fall back to
an F-H / lower-envelope path above it -- would keep the win at *both* ends
instead of conceding large grids. Lower priority than (1) because the small-grid
semiring win is the deliberate design point and the large-grid loss is bounded
(~2×, not ~18×).

## How it is measured

perf-bench cases `connected_components`, `largest_connected_component`,
`distance_transform_edt` (vs scipy co-oracle + cupyx GPU bar), with a brain-scale
size tier (96³–160³) and `tools/scaling_report.py` (the scale-gaming defence:
the curve + the cost law, so a small-size win can't hide a large-size loss).

## Cross-references

- `docs/feature-requests/connected-components.md` -- the implementation FR
  (shipped; this is its perf characterisation vs cupyx).
- `src/nitrix/morphology/_label.py` (pointer-jumping label prop);
  `src/nitrix/morphology/_mm.py` (`distance_transform` / `distance_transform_edt`
  -- the semiring engine).
