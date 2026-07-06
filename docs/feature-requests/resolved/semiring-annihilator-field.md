# B8. Store the `(*)`-annihilator explicitly on `Semiring`

> **Status (2026-06-02): SHIPPED.** `Semiring` now carries an
> `annihilator` field (`None` for EUCLIDEAN; `= identity` for the other
> built-ins) and `ell_mask` accepts `semiring=`, reading
> `semiring.annihilator` and raising when it is `None`. The legacy
> `ell_mask(identity=...)` form is retained but emits a
> `DeprecationWarning`. See `IMPLEMENTATION_PLAN.md §10.3` (2026-06-02
> entry) for the shipped-deviation record. Effort was **S**,
> backward-compatible as planned.
> Provenance: migrated from the retired top-level `BACKLOG.md` (B-numbering
> preserved); ledger context in [`internal-backlog.md`](../internal-backlog.md).

`Semiring.identity` is the **monoid (additive) identity** (`monoid.init`).
ELL padding and `sparse.ell_mask` (medial-wall / grey-matter masking)
actually need the **`(*)`-annihilator** — the `z` with `z (*) b =
monoid_identity` for all `b` — so a masked edge is a no-op. For every
built-in *except* EUCLIDEAN the two coincide (REAL `0`, LOG/TROPICAL_MAX
`-inf`, TROPICAL_MIN `+inf`, BOOLEAN `False`), which is why `identity`
currently doubles as the masking value. EUCLIDEAN's `(a−b)**2` has **no**
annihilator yet `identity == 0.0`, so masking by that value silently injects
`B[idx]**2` instead of vanishing. (Verified: `Semiring` in
`src/nitrix/semiring/_types.py` carries `identity`, no `annihilator` field.)

**Consideration.** Add `annihilator: Optional[...]` to `Semiring` (`None`
for EUCLIDEAN). Then `ell_mask` takes a `semiring=` and pulls
`semiring.annihilator`, raising a clear error when `None`, instead of
overloading `identity` and relying on the caller to know the distinction.

**Trigger.** A second masking consumer, a user-defined semiring whose monoid
identity and annihilator differ, or a confusion report. Until then
`ell_mask(ell, valid, *, identity=...)` takes an explicit value and the
docstring + `tests/test_ell_masking_semirings.py` document the distinction
(incl. the EUCLIDEAN exception).

**Effort.** S — one field + a guarded `ell_mask(semiring=...)` overload;
backward-compatible with the explicit-`identity` form.

## Cross-references

- [`internal-backlog.md`](../internal-backlog.md) — the engineering-backlog
  ledger.
- `docs/design/semiring-protocols.md` (the "identity is the monoid identity,
  not the annihilator" learning); `src/nitrix/semiring/_types.py`.
