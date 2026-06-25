# Two-phase `histogram_match` (fit reference landmarks once, apply to many)

> **Status (2026-06-25): request (nimox-estimators E2 → nitrix).** A small
> API-shape refinement to the shipped `nitrix.bias.histogram_match`. The
> single-pair `histogram_match(source, reference)` re-derives the **reference**
> landmarks on every call. nimox's new `HistogramMatch` estimator
> (`nimox.estimators`, the immutable sklearn-`fit`/`transform` façade) wants the
> Nyúl-Udupa "fit a standard scale once, apply to many subjects" workflow — so
> it currently has to carry the **whole reference volume** as fitted state and
> recompute its landmarks per `transform`. Exposing the landmark fit/apply split
> publicly shrinks that to ~9 floats and removes the redundant recompute. No new
> math — the internals already exist (`_landmarks_for` / `_match_points`).

## The ask

Split the public surface into the two phases the algorithm already has
internally:

```
histogram_match_fit(reference, *, reference_weight=None, n_match_points=7,
                    n_histogram_levels=1024, threshold_at_mean=True)
    -> landmarks            # the standard-scale landmark intensities (~9 floats)

histogram_match_apply(source, landmarks, *, source_weight=None,
                      n_match_points=7, n_histogram_levels=1024,
                      threshold_at_mean=True)
    -> matched              # source remapped onto the reference CDF
```

Keep the existing `histogram_match(source, reference, ...)` as the convenience
that composes them (`apply(source, fit(reference))`) — byte-unchanged, so no
caller breaks.

## Why (the nimox consumer)

`nimox.estimators.HistogramMatch.fit(reference)` should capture the reference
*distribution*, and `transform(source)` apply it to a new subject — the whole
point of the fit/transform split. With only the single-pair kernel public, the
faithful delegation is to **store the reference image** and call
`histogram_match(source, self.reference, ...)` each `transform`, which:

- carries a whole reference volume in the fitted PyTree (vs ~9 landmark floats),
  inflating `eqx.tree_serialise_leaves` payloads and `vmap`-fit batches; and
- recomputes the reference landmarks on every `transform` (the reference is
  fixed — pure redundant work, `O(n_histogram_levels)` per call).

A landmark-precompute split lets `FittedHistogramMatch` carry just the
landmark array and delegate `transform` to `histogram_match_apply`.

## Notes / scope

- The landmark array's length must agree between fit and apply (the
  `threshold_at_mean`-with-no-weight "+mean landmark" case adds one); the public
  fit should return the resolved landmark vector so apply is unambiguous, and
  apply should validate the length (the same invariant the single-pair path
  already enforces internally).
- Pure refinement: fp-parity with the current `histogram_match` is the
  acceptance bar (the composed convenience must be byte-identical).
- Differentiability is unchanged (the landmark search stays non-differentiable;
  the apply step's `jnp.interp` gradient flows as today).

## Acceptance

- `histogram_match_fit` / `histogram_match_apply` public, with
  `histogram_match(source, reference)` == `apply(source, fit(reference))`
  byte-for-byte.
- nimox `FittedHistogramMatch` re-points its `transform` at
  `histogram_match_apply`, carrying only the landmark vector (drops the stored
  reference volume); its E2 tests stay green.

## Cross-references

- nimox `docs/feature-requests/nimox-estimators.md` §13 Q5 (the origin) and §6
  (the `HistogramMatch` row of the delegation map).
- `nitrix.bias.histogram_match` (`_histogram_match.py` — `_landmarks_for` /
  `_match_points` are the internals to surface).
