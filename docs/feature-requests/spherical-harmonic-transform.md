# Spherical harmonic transforms — `nitrix.geometry.sphere.harmonics`

> **Status (2026-07-07): SHIPPED — analysis/synthesis (`nitrix.geometry`).**
> `sht_forward` / `sht_inverse` / `sht_grid`: exact spherical-harmonic transform
> on a **Gauss–Legendre** grid (`L+1` GL colatitudes × `2L+1` equiangular
> longitudes), FFT over longitude + a fully-normalised associated-Legendre matmul
> over colatitude (the grid + Legendre table precomputed host-side from the static
> band-limit, so the data path is pure FFT + contraction). Orthonormal
> Condon–Shortley `Y_lm`; verified against `scipy.special.sph_harm_y` (unit
> coefficient per harmonic), round-trip + Parseval exact to machine precision;
> jit/grad/batch clean. **Real spherical harmonics** (`real_sht_forward` /
> `real_sht_inverse` — the redundancy-free real-SH basis, e.g. dMRI FOD) and the
> **Driscoll–Healy equiangular grid** (`grid='driscoll_healy'` — the uniform
> sampling SH-equivariant spherical CNNs use) are now shipped too (2026-07-08),
> both round-tripping to machine precision. **Wigner-D coefficient rotation**
> (`sht_rotation_matrix` / `sht_rotate` — rotate SH coefficients by an SO(3)
> rotation, the small-d block as a matrix exponential of the angular-momentum
> generator, no recurrence; gimbal-lock handled) is now shipped too (2026-07-08),
> validated against direct field rotation (~1e-13), unitarity, and the group
> homomorphism `D(R2 R1) = D(R2) D(R1)`. **The suite is complete.** Provenance:
> `docs/feature-requests catalogue §12.9`.

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

- `docs/feature-requests catalogue §12.9` — origin entry; `§13` — acceptance protocol.
- `src/nitrix/geometry/sphere.py` — the substrate this extends.
- [`docs/design/sphere-grid.md`](../design/sphere-grid.md) — the
  parameterised-grid topology (the equirectangular sampling SHT needs).
