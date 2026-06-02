# Spherical harmonic transforms — `nitrix.geometry.sphere.harmonics`

> **Status (2026-06-02): not started.** Brainstorm candidate; promotion
> gated by the §13 acceptance protocol. Provenance:
> `SPEC_UPDATE_v0.3.md §12.9`.

**What.** Classical spherical-harmonic synthesis and analysis at arbitrary
band-limits — real and complex SHs, forward/inverse via Driscoll–Healy
quadrature.

**Proposed surface.**

```python
def sht_forward(f, *, band_limit): ...            # spatial -> SH coefficients
def sht_inverse(coeffs, *, n_lat, n_lon): ...      # SH coefficients -> spatial
def sht_rotation_matrix(R, band_limit): ...        # Wigner-D rotation in SH basis
```

**Composition.** Extends the existing `nitrix.geometry.sphere` substrate
(currently icosphere + `spherical_conv` + `spherical_geodesic_distance`)
with the parameterised-grid SH transforms.

**Likely consumer.** Surface-based CNNs at non-icosphere sampling,
SH-equivariant networks (sphere-domain transformers), fibre-orientation
distribution modelling in dMRI.

**Effort.** M.

**Live-code status.** No `sht_*` symbols. `geometry/__init__` ships
`spherical_conv`, `spherical_geodesic_distance`,
`cartesian_to_latlong` / `latlong_to_cartesian` — the coordinate plumbing
the quadrature grid would build on.

## Cross-references

- `SPEC_UPDATE_v0.3.md §12.9` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/geometry/sphere.py` — the substrate this extends.
- [`docs/design/sphere-grid.md`](../design/sphere-grid.md) — the
  parameterised-grid topology (the equirectangular sampling SHT needs).
