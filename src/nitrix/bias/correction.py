# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified bias-field-correction dispatcher.

``bias_field_correction(image, *, method=...)`` selects the field-fitting
algorithm.  All methods share the N3/N4 iterative loop (log-transform ->
histogram sharpen -> residual -> B-spline smooth -> accumulate, with the
ITK coefficient-of-variation convergence test); they differ only in how the
per-iteration residual field is smoothed:

- ``'n4'``            -- the Lee--Wolberg--Shin MBA, i.e. exactly N4 /
  ANTs.  The **parity** path: reproduces ITK ``N4BiasFieldCorrection`` to
  ~1e-4.  Use this for compatibility with legacy / published pipelines.
  (Identical to ``n4_bias_field_correction``.)
- ``'least_squares'`` -- a *regularised* weighted least-squares B-spline
  fit.  Unbiased (no MBA scaling bias), with the ``ridge`` term supplying
  the noise-robustness that MBA gets for free from its ``w^2`` local
  averaging.  Competitive with N4: better on some phantoms (lower scaled
  RMSE / residual CV), modestly worse on others with many tissue
  boundaries, where MBA's local averaging is more robust.  The optimal
  ``ridge`` is data-dependent (principled GCV/REML selection is a noted
  extension).
- ``'psplines'``      -- penalised least-squares (Eilers--Marx P-splines):
  adds a difference-penalty roughness term on top of ``ridge``, giving an
  extra smoothness knob (``penalty``) independent of grid resolution.

All three use the same multi-resolution coarse-to-fine schedule -- it is
load-bearing, not optional: a single fine-grid fit captures non-bias
structure from the residual and fails (see ``docs/design/bias-field.md``).
The crucial lesson is that the *regularisation* (``ridge`` for LS /
P-splines, the implicit ``w^2`` averaging for MBA) is what makes the fit
denoise the noisy per-iteration residual; an unregularised LS fit
overfits it.  Only ``'n4'`` is ITK-parity; ``'least_squares'`` /
``'psplines'`` are the "best available" estimators for internal use where
ANTs bit-compatibility is not required.  See ``docs/design/bias-field.md``
(parity vs correctness; the bias-variance finding; Tier C/D extensions).
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

from jaxtyping import Array, Float

from ._core import apply_bias_field_correction

__all__ = ['bias_field_correction']

CorrectionMethod = Literal['n4', 'least_squares', 'psplines']

_METHOD_TO_FIT = {
    'n4': 'mba',
    'least_squares': 'least_squares',
    'psplines': 'psplines',
}


def bias_field_correction(
    image: Float[Array, '... *spatial'],
    *,
    method: CorrectionMethod = 'n4',
    mask: Optional[Float[Array, '... *spatial']] = None,
    spline_order: int = 3,
    n_control_points: Union[int, Sequence[int]] = 4,
    n_fitting_levels: int = 4,
    max_iterations: Union[int, Sequence[int]] = 50,
    convergence_threshold: float = 1e-3,
    n_histogram_bins: int = 200,
    fwhm: float = 0.15,
    wiener_noise: float = 0.01,
    ridge: float = 1e-1,
    penalty: float = 1e-1,
    penalty_order: int = 2,
    spatial_rank: Optional[int] = None,
    return_bias_field: bool = False,
    eps: float = 1e-8,
) -> Union[
    Float[Array, '... *spatial'],
    Tuple[Float[Array, '... *spatial'], Float[Array, '... *spatial']],
]:
    """Bias-field correction with a selectable field-fitting algorithm.

    Estimates the smooth multiplicative bias field corrupting ``image`` and
    returns the corrected image (``image / bias``).  All methods share the
    N3/N4 iteration; ``method`` selects the B-spline field estimator.

    Parameters
    ----------
    image
        Positive intensities, ``(..., *spatial)``.  Leading dims (with
        ``spatial_rank`` set) are a batch of independently corrected volumes.
    method
        Field-fitting algorithm:

        - ``'n4'`` (default) -- Lee--Wolberg--Shin MBA; exactly N4 / ANTs
          (ITK parity).  Equivalent to ``n4_bias_field_correction``.
        - ``'least_squares'`` -- regularised weighted least-squares
          B-spline fit; unbiased, denoised by ``ridge``.  Competitive with
          N4 (data-dependent which wins).
        - ``'psplines'`` -- penalised least-squares (P-splines); ``ridge``
          plus a difference-roughness ``penalty`` for extra smoothing.

        All methods use the multi-resolution coarse-to-fine schedule
        (``n_fitting_levels``); it is required for stability.
    mask
        Region over which the field is estimated; ``None`` uses
        ``image > 0``.  Float masks are per-voxel confidence weights.
    spline_order, n_control_points, n_fitting_levels, max_iterations,
    convergence_threshold, n_histogram_bins, fwhm, wiener_noise, spatial_rank,
    return_bias_field, eps
        As in ``n4_bias_field_correction``.
    ridge
        Tikhonov regularisation (relative to the mean Gram diagonal) for the
        ``'least_squares'`` / ``'psplines'`` normal equations.  Default
        ``1e-1`` -- this is the **denoising** strength on the noisy
        per-iteration residual (the LS analogue of MBA's ``w^2`` averaging),
        *not* a mere numerical stabiliser; too small (e.g. ``1e-4``)
        overfits residual noise and degrades the field.  Ignored by ``'n4'``.
    penalty, penalty_order
        P-spline roughness penalty weight (relative to the mean Gram
        diagonal; larger -> smoother) and difference order (default ``2``,
        penalising curvature).  Only used by ``'psplines'``.

    Returns
    -------
    ``corrected`` (default), or ``(corrected, bias_field)`` if
    ``return_bias_field``.
    """
    if method not in _METHOD_TO_FIT:
        raise ValueError(
            f'method={method!r}; expected one of {tuple(_METHOD_TO_FIT)!r}.'
        )
    return apply_bias_field_correction(
        image,
        mask=mask,
        fit_method=_METHOD_TO_FIT[method],  # type: ignore[arg-type]
        spline_order=spline_order,
        n_control_points=n_control_points,
        n_fitting_levels=n_fitting_levels,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        n_histogram_bins=n_histogram_bins,
        fwhm=fwhm,
        wiener_noise=wiener_noise,
        ridge=ridge,
        penalty=penalty,
        penalty_order=penalty_order,
        spatial_rank=spatial_rank,
        return_bias_field=return_bias_field,
        eps=eps,
    )
