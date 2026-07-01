# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified bias-field-correction dispatcher.

:func:`bias_field_correction` selects the field-fitting algorithm through
its ``method`` argument.  All methods share the N3/N4 iterative loop
(log-transform, histogram sharpen, residual, B-spline smooth, accumulate,
with the ITK coefficient-of-variation convergence test); they differ only
in how the per-iteration residual field is smoothed:

- ``'n4'`` -- the Lee--Wolberg--Shin multilevel B-spline approximation
  (MBA), i.e. exactly N4 / ANTs.  The parity path: reproduces the ITK
  ``N4BiasFieldCorrection`` routine to roughly :math:`10^{-4}`.  Use this
  for compatibility with legacy or published pipelines.  Identical to
  :func:`n4_bias_field_correction`.
- ``'least_squares'`` -- a regularised weighted least-squares B-spline fit.
  Unbiased (no MBA scaling bias), with the ``ridge`` term supplying the
  noise-robustness that MBA gets for free from its :math:`w^2` local
  averaging.  Competitive with N4: better on some phantoms (lower scaled
  RMSE / residual coefficient of variation), modestly worse on others with
  many tissue boundaries, where MBA's local averaging is more robust.  The
  optimal ``ridge`` is data-dependent.
- ``'psplines'`` -- penalised least-squares (Eilers--Marx P-splines): adds
  a difference-penalty roughness term on top of ``ridge``, giving an extra
  smoothness knob (``penalty``) independent of grid resolution.

All three use the same multi-resolution coarse-to-fine schedule; it is
load-bearing, not optional: a single fine-grid fit captures non-bias
structure from the residual and fails.  The crucial point is that the
regularisation (``ridge`` for least-squares and P-splines, the implicit
:math:`w^2` averaging for MBA) is what makes the fit denoise the noisy
per-iteration residual; an unregularised least-squares fit overfits it.
Only ``'n4'`` is ITK-parity; ``'least_squares'`` and ``'psplines'`` are the
best-available estimators for use where ANTs bit-compatibility is not
required.
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
    image : Float[Array, '... *spatial']
        Positive intensities over the spatial grid.  Leading dimensions
        (with ``spatial_rank`` set) are a batch of independently corrected
        volumes.
    method : {'n4', 'least_squares', 'psplines'}, optional
        Field-fitting algorithm (default ``'n4'``):

        - ``'n4'`` -- Lee--Wolberg--Shin MBA; exactly N4 / ANTs (ITK
          parity).  Equivalent to :func:`n4_bias_field_correction`.
        - ``'least_squares'`` -- regularised weighted least-squares B-spline
          fit; unbiased, denoised by ``ridge``.  Competitive with N4 (which
          wins is data-dependent).
        - ``'psplines'`` -- penalised least-squares (P-splines); ``ridge``
          plus a difference-roughness ``penalty`` for extra smoothing.

        All methods use the multi-resolution coarse-to-fine schedule
        (``n_fitting_levels``); it is required for stability.
    mask : Float[Array, '... *spatial'], optional
        Region over which the field is estimated; ``None`` uses
        ``image > 0``.  Float masks are per-voxel confidence weights.
    spline_order : int, optional
        Degree of the B-spline basis (default ``3``, cubic).  As in
        :func:`n4_bias_field_correction`.
    n_control_points : int or sequence of int, optional
        Number of B-spline control points at the coarsest fitting level, per
        spatial axis or shared (default ``4``).  As in
        :func:`n4_bias_field_correction`.
    n_fitting_levels : int, optional
        Number of coarse-to-fine multi-resolution levels; the control-point
        mesh doubles each level (default ``4``).  As in
        :func:`n4_bias_field_correction`.
    max_iterations : int or sequence of int, optional
        Maximum sharpening iterations, per level or shared across levels
        (default ``50``).  As in :func:`n4_bias_field_correction`.
    convergence_threshold : float, optional
        Coefficient-of-variation threshold on the per-iteration field ratio
        at which a level terminates (default ``1e-3``).  As in
        :func:`n4_bias_field_correction`.
    n_histogram_bins : int, optional
        Number of bins for the intensity histogram used in the deconvolution
        sharpening step (default ``200``).  As in
        :func:`n4_bias_field_correction`.
    fwhm : float, optional
        Full width at half maximum of the Gaussian point-spread function
        assumed by the histogram sharpening (default ``0.15``).  As in
        :func:`n4_bias_field_correction`.
    wiener_noise : float, optional
        Wiener-deconvolution noise level for the histogram sharpening
        (default ``0.01``).  As in :func:`n4_bias_field_correction`.
    ridge : float, optional
        Tikhonov regularisation (relative to the mean Gram diagonal) for the
        ``'least_squares'`` / ``'psplines'`` normal equations.  Default
        ``1e-1``; this is the denoising strength on the noisy per-iteration
        residual (the least-squares analogue of MBA's :math:`w^2`
        averaging), not a mere numerical stabiliser; too small (e.g.
        ``1e-4``) overfits residual noise and degrades the field.  Ignored by
        ``'n4'``.
    penalty : float, optional
        P-spline roughness penalty weight (relative to the mean Gram
        diagonal; larger gives a smoother field).  Default ``1e-1``.  Only
        used by ``'psplines'``.
    penalty_order : int, optional
        Order of the finite-difference roughness penalty (default ``2``,
        penalising curvature).  Only used by ``'psplines'``.
    spatial_rank : int, optional
        Number of trailing axes of ``image`` that are spatial; the remainder
        are batch dimensions.  ``None`` infers it from ``n_control_points``
        (if a sequence) or treats every dim as spatial (single volume).  As
        in :func:`n4_bias_field_correction`.
    return_bias_field : bool, optional
        If ``True``, also return the estimated multiplicative bias field
        (default ``False``).  As in :func:`n4_bias_field_correction`.
    eps : float, optional
        Small positive floor applied before the log transform to guard
        against non-positive intensities (default ``1e-8``).  As in
        :func:`n4_bias_field_correction`.

    Returns
    -------
    corrected : Float[Array, '... *spatial']
        The bias-corrected image (``image / bias_field``), with the same
        shape as ``image``.  Returned alone when ``return_bias_field`` is
        ``False``.
    bias_field : Float[Array, '... *spatial']
        The estimated multiplicative bias field, same shape as ``image``.
        Returned as the second element of the tuple only when
        ``return_bias_field`` is ``True``.
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
