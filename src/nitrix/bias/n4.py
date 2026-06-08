# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
N4 bias-field correction (Tustison 2010), GPU-accelerated in pure JAX.

N4 estimates and removes the smooth, low-frequency multiplicative
intensity inhomogeneity ("bias field") that corrupts MRI.  It is the
improved ("N4ITK") successor to N3 (Sled 1998); the improvements over N3
are (a) a B-spline *approximation* of the field instead of N3's smoothing
spline and (b) a multi-resolution fitting hierarchy.  This module follows
the ITK / ANTs ``N4BiasFieldCorrectionImageFilter`` algorithm and default
parameters, reformulated for the accelerator:

- The per-iteration histogram **sharpening** is the N3 Wiener
  deconvolution (``_sharpen.sharpen_histogram``) -- 1-D FFTs, cheap.
- The field **smoothing** is the Lee--Wolberg--Shin multilevel B-spline
  approximation, specialised to the regular image grid so it becomes
  separable dense per-axis contractions (``_bspline``) -- no gather, no
  scatter, lowers to XLA ``dot`` / tensor cores.
- The **iteration** is a ``lax.while_loop`` with ITK's coefficient-of-
  variation convergence test; the **multi-resolution** hierarchy is a
  static outer loop doubling the B-spline mesh per level.

``n4_bias_field_correction`` is, deliberately, *exactly N4* -- the MBA fit,
validated to ITK/ANTs parity (``docs/design/bias-field.md``).  Higher-
accuracy field estimators (least-squares, P-splines) are a *different*
algorithm and live behind ``nitrix.bias.bias_field_correction(method=...)``,
not here; conflating them with the N4 name would mislabel non-N4 output.

Why no Pallas: the heavy ops are dense small-matrix contractions and 1-D
FFTs, which XLA already schedules near-optimally on GPU/TPU.  Per the house
"benchmark-first, don't build Pallas speculatively" policy
(docs/feature-requests/internal-backlog.md B6/B7), N4 ships pure-JAX.

Differentiability: the B-spline smoothing is differentiable; the histogram
sharpening is not (piecewise-constant binning).  Per the consumer ask,
efficient end-to-end differentiability is a plus, not a requirement.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from jaxtyping import Array, Float

from ._core import apply_bias_field_correction

__all__ = ['n4_bias_field_correction']


def n4_bias_field_correction(
    image: Float[Array, '... *spatial'],
    *,
    mask: Optional[Float[Array, '... *spatial']] = None,
    spline_order: int = 3,
    n_control_points: Union[int, Sequence[int]] = 4,
    n_fitting_levels: int = 4,
    max_iterations: Union[int, Sequence[int]] = 50,
    convergence_threshold: float = 1e-3,
    n_histogram_bins: int = 200,
    fwhm: float = 0.15,
    wiener_noise: float = 0.01,
    spatial_rank: Optional[int] = None,
    return_bias_field: bool = False,
    eps: float = 1e-8,
) -> Union[
    Float[Array, '... *spatial'],
    Tuple[Float[Array, '... *spatial'], Float[Array, '... *spatial']],
]:
    """N4 (Tustison) bias-field correction.

    Estimates the smooth multiplicative bias field corrupting ``image`` and
    returns the corrected image (``image / bias``).  Follows the ITK / ANTs
    ``N4BiasFieldCorrectionImageFilter`` algorithm and defaults; see the
    module docstring and ``docs/design/bias-field.md`` for the accelerator
    reformulation and the parity validation.

    This function is N4 specifically (the Lee--Wolberg--Shin MBA field
    fit).  For the higher-accuracy least-squares / P-spline estimators, use
    ``nitrix.bias.bias_field_correction(method=...)`` -- they are a
    different algorithm and are kept off the N4 name on purpose.

    The entire input (modulo leading batch dims) is treated as one volume.
    For a batch of volumes -- distinct leading dims with ``spatial_rank``
    set -- each volume is corrected independently (its own histogram,
    field, and convergence) via an internal ``vmap``.

    Parameters
    ----------
    image
        Positive intensities, ``(..., *spatial)``.  Non-positive voxels are
        clamped for the log transform; keep them out of the mask.
    mask
        Region over which the field is estimated (the histogram and B-spline
        fit are confined to it; the field is still reconstructed -- and the
        correction applied -- everywhere).  ``None`` (default) uses
        ``image > 0``.  Float masks act as per-voxel confidence weights.
    spline_order
        B-spline order for the field.  Default ``3`` (cubic; the N4 default
        and the parity-validated path).
    n_control_points
        Control points per spatial axis at the **coarsest** fitting level
        (ITK ``NumberOfControlPoints``).  ``int`` -- isotropic; sequence --
        per-axis.  Must be ``>= spline_order + 1``.  The mesh doubles each
        level.  Default ``4`` (-> mesh size 1 at the coarsest level).
    n_fitting_levels
        Number of multi-resolution levels; the B-spline mesh doubles per
        level.  Default ``4``.
    max_iterations
        Maximum sharpen/fit iterations per level.  ``int`` -- same cap every
        level; sequence -- per-level (length ``n_fitting_levels``).  Default
        ``50`` (ITK default is ``(50, 50, 50, 50)``).
    convergence_threshold
        Stop a level early when the coefficient of variation of the field
        update drops below this.  Default ``1e-3`` (ITK default).
    n_histogram_bins, fwhm, wiener_noise
        Histogram-sharpening parameters; see ``sharpen_histogram``.  Defaults
        ``200`` / ``0.15`` / ``0.01`` (ITK defaults).
    spatial_rank
        Number of trailing dims that are spatial.  ``None`` infers it from
        ``n_control_points`` (if a sequence) or treats every dim as spatial
        (single volume).  Set it to mark leading batch dims.
    return_bias_field
        If ``True``, also return the estimated multiplicative bias field
        (same shape as ``image``), e.g. for QC or to apply to another image.
    eps
        Numerical floor for the log transform and the B-spline denominators.

    Returns
    -------
    ``corrected`` (default), or ``(corrected, bias_field)`` if
    ``return_bias_field`` -- both with the same shape as ``image``.

    Notes
    -----
    The B-spline field smoothing is differentiable; the histogram sharpening
    is not (piecewise-constant binning).  The op JITs cleanly (static FFT
    and control-lattice shapes; ``lax.while_loop`` iteration).
    """
    return apply_bias_field_correction(
        image,
        mask=mask,
        fit_method='mba',
        spline_order=spline_order,
        n_control_points=n_control_points,
        n_fitting_levels=n_fitting_levels,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        n_histogram_bins=n_histogram_bins,
        fwhm=fwhm,
        wiener_noise=wiener_noise,
        ridge=0.0,
        penalty=0.0,
        penalty_order=2,
        spatial_rank=spatial_rank,
        return_bias_field=return_bias_field,
        eps=eps,
    )
