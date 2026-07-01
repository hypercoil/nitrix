# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Mass-univariate generalised **linear mixed** models (GLMM) via PQL.

:func:`glmm_fit` fits, per element (voxel / vertex / fixel), a random-intercept
GLMM under a non-Gaussian :class:`~nitrix.stats._family.Family`:

.. code-block:: text

    g(E[y | b]) = X beta + b[group],   b_j ~ N(0, sigma_b^2),   j = 1 .. q

by **penalised quasi-likelihood** (PQL; Breslow & Clayton, 1993): an outer IRLS
loop forms the working response :math:`z` and weights :math:`W` from the current
fit, then a single variance-component (Fellner-Schall) step updates the
random-effect precision :math:`\lambda = \phi / \sigma_b^2` -- the "penalised GLM
= variance-components REML" identity, here under a GLM family.  This is the
estimator ``mgcv::gam(family=, s(g, bs="re"))`` / ``glmmPQL`` (with the documented
PQL attenuation for binary / low-count responses; a Laplace approximation is the
more accurate follow-up).

Performance-preserving dispatch
-------------------------------

A random effect widens the per-voxel system by :math:`q` = #levels, so the dense
penalised solve is :math:`O((p + q)^3)` per voxel.  That is **cheap for few-level
factors** (site / scanner / batch, :math:`q \sim 10\text{--}50`) but a latency
cliff for **many-level** factors (random intercept per *subject*,
:math:`q \sim 100\text{--}1000`).  :func:`glmm_fit` therefore dispatches on the
level count:

- **few-level** (``q <= few_level_max``): the dense GAMM path --
  :func:`~nitrix.stats.gam.gam_fit` with a
  :func:`~nitrix.stats.basis.re_smooth` block and the GLM family.  This *is* the
  optimal solver in this regime (no cheaper exact route), and reuses the shipped
  GAM machinery verbatim.
- **many-level** (``q > few_level_max``): a **structured** PQL that never forms
  the :math:`(p + q) \times (p + q)` system.  Because a single grouping factor
  makes the random-effect block of the normal equations **diagonal across
  groups**, the Schur complement onto the :math:`p`-dimensional fixed block costs
  :math:`O(N p^2 + q)` per voxel (the block-Woodbury structure, weighted by the
  IRLS :math:`W` and wrapped in the PQL loop) -- linear in the level count, and
  free of cuSOLVER.

The two paths run the *same* PQL iteration (same working response, same
Fellner-Schall update, same iteration budget), so they agree to the iterative
tolerance; the dispatch only changes the linear-algebra cost, never the answer.

Scope.  Beyond the **scalar random intercept** ``(1 | g)`` (binary outcomes /
lesion counts per subject with random intercepts), :func:`glmm_fit` also fits
non-Gaussian random **slopes** via the ``z`` / ``structure`` arguments: diagonal
``(x || g)`` as independent :func:`~nitrix.stats.basis.re_smooth` blocks through
:func:`~nitrix.stats.gam.gam_fit`, and correlated ``(1 + x | g)`` via the
block-Woodbury REML wrapped in the PQL loop (the penalised-IRLS =
iteratively-reweighted-REML identity).  The Gaussian-family slope fit is the same
REML estimator as :func:`~nitrix.stats.lme.lme_fit` (block-Woodbury), to the
iterative (optimiser) tolerance.  Random slopes are also served under the
**Laplace** marginal likelihood (``method='laplace'``, the :math:`r`-dimensional
conditional-mode integral + :math:`r \times r` determinant correction), which
corrects the PQL attenuation for binary / low-count slopes.

References
----------
- Breslow, N. E. & Clayton, D. G. (1993). Approximate inference in generalised
  linear mixed models.  Journal of the American Statistical Association, 88,
  9-25.  :doi:`10.2307/2290687`
- Wood, S. N. & Fasiolo, M. (2017). A generalised Fellner-Schall method for
  smoothing parameter optimisation with application to Tweedie location, scale
  and shape models.  Biometrics, 73, 1071-1081.  :doi:`10.1111/biom.12666`
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .._family import GAUSSIAN, Family, resolve_family
from ..lme._blup import PredictLevel, _conditional_eta
from ._agq import _glmm_agq_slope
from ._base import _AGQ_MAX_NODES, GLMMResult
from ._laplace import _glmm_laplace, _glmm_laplace_slope
from ._pql import (
    _glmm_few_level,
    _glmm_many_level,
    _glmm_slope_diagonal,
    _glmm_slope_structured,
)

__all__ = ['GLMMResult', 'glmm_fit', 'glmm_predict']

GLMMStructure = Literal['unstructured', 'diagonal']
GLMMMethod = Literal['pql', 'laplace', 'agq']


def glmm_fit(
    Y: Float[Array, 'V N'],
    X: Float[Array, 'N p'],
    *,
    group: Int[Array, 'N'],
    n_groups: Optional[int] = None,
    z: Optional[Float[Array, 'N r']] = None,
    structure: GLMMStructure = 'unstructured',
    family: Union[str, Family] = GAUSSIAN,
    method: GLMMMethod = 'pql',
    few_level_max: int = 64,
    n_outer: int = 20,
    n_inner: int = 10,
    n_mode: int = 20,
    n_quad: int = 5,
    ridge: float = 1e-8,
    lam_floor: float = 1e-6,
    lam_ceil: float = 1e8,
    damping: float = 1e-6,
    block: Optional[int] = None,
) -> GLMMResult:
    r"""Mass-univariate random-intercept GLMM via PQL, dispatched on level count.

    Fits, per element, :math:`g(E[y \mid b]) = X \beta + b_{\text{group}}` with
    :math:`b_j \sim N(0, \sigma_b^2)` under the GLM ``family``, by penalised
    quasi-likelihood (working response -> one variance-component step -> repeat).
    The fixed design ``X`` is used as given (include your own intercept column);
    ``group`` is the ``(N,)`` integer grouping factor (levels
    :math:`0 \ldots q-1`).

    Dispatch selects the linear algebra by the level count :math:`q`, without
    changing the fitted answer:

    - ``q <= few_level_max`` -- **few-level**: the dense GAMM path
      (:func:`~nitrix.stats.gam.gam_fit` + :func:`~nitrix.stats.basis.re_smooth`).
      Optimal here; the per-voxel cost :math:`O((p + q)^3)` is tiny.
    - ``q > few_level_max`` -- **many-level**: a structured Schur-complement PQL
      costing :math:`O(N p^2 + q)` per voxel (the block-Woodbury structure,
      weighted and wrapped in the IRLS loop), avoiding the dense
      :math:`(p + q)`-wide solve.

    Both paths run the identical PQL iteration, so they agree to the iterative
    tolerance -- the dispatch changes only the linear-algebra cost.

    Parameters
    ----------
    Y
        ``(V, N)`` responses (one row per element, one column per observation).
    X
        ``(N, p)`` fixed-effect design (shared across elements; carries its own
        intercept).
    group
        ``(N,)`` integer grouping factor (random intercept per level).
    n_groups
        Optional **static** level count :math:`q` (levels are
        :math:`0 \ldots q-1`).  When ``None`` (default) it is derived eagerly as
        ``int(max(group)) + 1``, which concretises ``group`` and so makes
        :func:`glmm_fit` untraceable under ``jax.jit``.  Pass the count explicitly
        (a Python ``int``) to trace the whole fit under ``jit`` -- e.g. to fuse it
        into a larger program -- for every family / structure / method.
    z
        Optional ``(N, r)`` random-effect design for a random **slope** (e.g.
        ``[1, x]`` -> random intercept + random slope of ``x``).  ``None``
        (default) is the scalar random intercept ``(1 | g)``.  Mirrors the ``z=``
        argument of :func:`~nitrix.stats.lme.lme_fit`; the Gaussian-family slope
        GLMM is the same REML fit as ``lme_fit(z=, structure=)`` (to optimiser
        tolerance).
    structure
        Random-effect covariance for a slope (``z`` given): ``'unstructured'``
        (full :math:`r \times r` :math:`G`, the correlated ``(1 + x | g)``) or
        ``'diagonal'`` (independent variance components, the uncorrelated
        ``(x || g)``).  Ignored when ``z is None``.
    family
        GLM family (``'binomial'`` / ``'poisson'`` / ``'gamma'`` /
        ``'negbinomial'`` / ``'gaussian'`` or a
        :class:`~nitrix.stats._family.Family`).  Gaussian is accepted (it reduces
        to the LME) but :func:`~nitrix.stats.lme.lme_fit` /
        :func:`~nitrix.stats.lme.reml_fit` are the direct route for a Gaussian
        mixed model.
    method
        ``'pql'`` (default) -- penalised quasi-likelihood (cheap; documented
        small-cluster attenuation for binary / low-count responses).
        ``'laplace'`` -- the Laplace-approximate marginal likelihood (per-group
        conditional-mode integral + curvature correction): more accurate for
        binary / low-count GLMMs (it corrects the PQL bias), at the cost of an
        inner mode-finding loop.  Fits a scalar random intercept, or a random
        **slope** when ``z`` is given (the :math:`r`-dimensional mode +
        :math:`r \times r` determinant correction, ``structure`` selecting a full
        / diagonal :math:`G`); the ``tier`` is ``'laplace'``.
        ``'agq'`` -- adaptive Gauss-Hermite quadrature (**random slope only**,
        ``z`` required): the marginal random-effect integral by ``n_quad``-point
        tensor Gauss-Hermite, centred / scaled at each group's mode and curvature.
        ``n_quad = 1`` is exactly Laplace; more nodes integrate the density
        directly, converging to the exact marginal (the ``lme4`` ``nAGQ`` accuracy
        tier, the gold standard for small / low-count clusters).  ``tier`` is
        ``'agq'``.
    few_level_max
        Dispatch threshold on the number of levels :math:`q` (default ``64`` --
        the regime where the dense solve stops being trivially cheap).  ``'pql'``
        only.
    n_outer
        PQL outer (Fellner-Schall) iteration budget.
    n_inner
        PQL inner (IRLS) iteration budget.
    n_mode
        Per-group conditional-mode Newton budget for ``method='laplace'`` /
        ``'agq'``.
    n_quad
        Gauss-Hermite nodes **per dimension** for ``method='agq'`` (default ``5``;
        the integral uses :math:`\text{n\_quad}^{r}` tensor nodes).
    ridge
        Normal-equation ridge stabiliser added to the fixed-effect system.
    lam_floor
        Lower clamp on the smoothing (precision) parameter
        :math:`\lambda = \phi / \sigma_b^2`.
    lam_ceil
        Upper clamp on the smoothing (precision) parameter.
    damping
        Levenberg-style damping factor for the outer Newton optimiser over the
        variance components, keeping the variance-parameter step stable.
    block
        Optional element-block size bounding peak memory (elements are processed
        in blocks of this many rows of ``Y``).  ``None`` processes all elements at
        once.

    Returns
    -------
    GLMMResult
        A :class:`~nitrix.stats.glmm._base.GLMMResult` carrying ``beta_hat``,
        per-level ``blups``, ``re_var`` (:math:`\sigma_b^2`, or the
        :math:`r`-vector / :math:`r \times r` :math:`G` for a slope),
        ``dispersion``, ``deviance``, ``edf_total``, and the ``tier`` that ran
        (``'few'`` / ``'many'`` / ``'slope'`` / ``'laplace'`` / ``'agq'``).

    Notes
    -----
    PQL carries the documented small-cluster bias for binary / low-count
    responses (it under-estimates the variance component; the bias vanishes as
    the per-group information grows).  Random *slopes* (``z`` + ``structure``) are
    fit by PQL -- diagonal via :func:`~nitrix.stats.gam.gam_fit` blocks,
    correlated via the joint-Schur + REML-EM solver (``tier='slope'``) -- by
    Laplace (``method='laplace'``), or by adaptive Gauss-Hermite quadrature
    (``method='agq'``, ``tier='agq'``), the accuracy ladder for the slope-variance
    attenuation: AGQ with ``n_quad`` nodes converges to the exact marginal
    (``n_quad = 1`` is Laplace).
    """
    family = resolve_family(family)
    n = X.shape[0]
    group = jnp.asarray(group)
    if Y.shape[-1] != n:
        raise ValueError(
            f'glmm_fit: Y.shape[-1]={Y.shape[-1]} must match N={n}.'
        )
    if group.shape[0] != n:
        raise ValueError(
            f'glmm_fit: group has {group.shape[0]} labels; expected N={n}.'
        )
    # The level count is a *static* shape driver (segment sums, output shapes,
    # the few/many dispatch).  Deriving it from the data with int(jnp.max(...))
    # concretises a tracer, so the eager-only default makes glmm_fit untraceable
    # under jax.jit (P7); pass n_groups explicitly -- the count the caller
    # already knows -- to fuse the whole fit (mirrors lme_fit / reml_fit, where
    # the level structure is likewise caller-supplied).
    # Resolve the static level count exactly as before: int(jnp.max(group))
    # concretises a tracer, so without an explicit n_groups this (correctly)
    # raises under jit -- the P7 contract (pass n_groups to trace).
    if n_groups is None:
        n_groups = int(jnp.max(group)) + 1
    # Validate the label range on the host when `group` is concrete (the eager
    # case, or a constant captured by an outer jit -- np.asarray reads the
    # constant without building a tracer). one_hot / segment_sum / b[group] all
    # silently zero out-of-range or negative indices, so a bad label would drop
    # that observation from the group structure (a wrong-but-finite fit). A
    # genuinely traced `group` is skipped (the caller owns the contract).
    if not isinstance(group, jax.core.Tracer):
        g_host = np.asarray(group)
        g_min, g_max = int(g_host.min()), int(g_host.max())
        if g_min < 0 or g_max >= n_groups:
            raise ValueError(
                f'glmm_fit: group labels must lie in [0, n_groups={n_groups}); '
                f'got min={g_min}, max={g_max}. Pass contiguous 0-based labels '
                'so no observation silently drops out of the group structure.'
            )

    # The Laplace / AGQ marginal likelihoods evaluate the family at a fixed
    # dispersion of 1; for a free-dispersion family the residual scale is not a
    # parameter of that marginal, so the optimiser silently folds the missing
    # scale into the random-effect covariance G (a wrong fit). Only the PQL path
    # estimates the dispersion.  Reject the mis-specified combination outright.
    if method in ('laplace', 'agq') and not family.has_fixed_dispersion:
        raise ValueError(
            f'glmm_fit: method={method!r} marginalises with a fixed dispersion '
            f'of 1, which mis-specifies the free-dispersion family '
            f'{family.name!r} (the residual scale would be absorbed into the '
            f"random-effect covariance). Use method='pql' (which estimates the "
            'dispersion), or for a Gaussian response use lme_fit / reml_fit '
            '(exact REML).'
        )

    if z is not None:
        z = jnp.asarray(z, dtype=X.dtype)
        if z.ndim != 2 or z.shape[0] != n:
            raise ValueError(
                f'glmm_fit: z must be (N, r) with N={n}; got shape '
                f'{tuple(z.shape)}.'
            )
        if structure not in ('unstructured', 'diagonal'):
            raise ValueError(
                f"glmm_fit: structure={structure!r}; expected 'unstructured' "
                "or 'diagonal'."
            )
        diagonal = structure == 'diagonal'
        if method == 'laplace':
            return _glmm_laplace_slope(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_mode,
                damping,
                diagonal,
                block,
            )
        if method == 'agq':
            r = z.shape[-1]
            n_nodes = n_quad**r
            if n_nodes > _AGQ_MAX_NODES:
                raise ValueError(
                    f'glmm_fit: AGQ tensor grid has n_quad**r = {n_quad}**{r} '
                    f'= {n_nodes} nodes (> {_AGQ_MAX_NODES}); the per-element '
                    f'graph grows as n_quad**r and is differentiated through the '
                    f'mode scan, so a large r is a compile / memory cliff.  Use a '
                    f"smaller n_quad, method='laplace' (= AGQ with n_quad=1), or "
                    f'a lower-dimensional random effect.'
                )
            return _glmm_agq_slope(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_mode,
                damping,
                diagonal,
                n_quad,
                block,
            )
        if method != 'pql':
            raise ValueError(
                f"glmm_fit: method={method!r}; expected 'pql', 'laplace' or "
                "'agq'."
            )
        if diagonal:
            return _glmm_slope_diagonal(
                Y,
                X,
                group,
                n_groups,
                z,
                family,
                n_outer,
                n_inner,
                ridge,
                lam_floor,
                lam_ceil,
                block,
            )
        return _glmm_slope_structured(
            Y,
            X,
            group,
            n_groups,
            z,
            family,
            n_outer,
            n_inner,
            ridge,
            block,
        )

    if method == 'laplace':
        return _glmm_laplace(
            Y, X, group, n_groups, family, n_outer, n_mode, damping, block
        )
    if method == 'agq':
        raise NotImplementedError(
            "glmm_fit: adaptive Gauss-Hermite (method='agq') currently requires "
            'a random slope -- pass z= (use z=ones((N, 1)) for a scalar '
            "intercept), or use method='laplace'."
        )
    if method != 'pql':
        raise ValueError(
            f"glmm_fit: method={method!r}; expected 'pql' or 'laplace'."
        )

    args = (
        Y,
        X,
        group,
        n_groups,
        family,
        n_outer,
        n_inner,
        ridge,
        lam_floor,
        lam_ceil,
        block,
    )
    if n_groups <= few_level_max:
        return _glmm_few_level(*args)
    return _glmm_many_level(*args)


def glmm_predict(
    result: GLMMResult,
    X: Float[Array, 'N p'],
    *,
    z: Optional[Float[Array, 'N r']] = None,
    group: Optional[Int[Array, 'N']] = None,
    level: PredictLevel = 'population',
    type: Literal['response', 'link'] = 'response',
) -> Float[Array, 'V N']:
    r"""Per-element GLMM prediction on a (new) design ``X``.

    The mixed-model apply half: the linear predictor :math:`\eta = X \hat\beta`
    (``level='population'``, the marginal mean) optionally plus the
    subject-specific random-effect contribution :math:`Z b_{\text{group}}`
    (``level='conditional'``, using the BLUPs a
    :class:`~nitrix.stats.glmm._base.GLMMResult` always retains), then mapped
    through the link.  ``type='link'`` returns :math:`\eta`; ``type='response'``
    (default) returns ``family.linkinv(eta)`` (the mean response -- a probability
    / rate / count).

    Parameters
    ----------
    result
        A :class:`GLMMResult` from :func:`glmm_fit`.
    X
        ``(N, p)`` fixed-effect design (same columns as the fit).
    z
        ``(N, r)`` random-effect design for the new rows (a random-slope fit);
        ``None`` for a scalar intercept.
    group
        ``(N,)`` integer group labels indexing the fitted levels ``0..q-1``;
        an out-of-range / ``None`` group falls back to the marginal mean.
    level
        ``'population'`` (marginal) or ``'conditional'`` (subject-specific).
    type
        ``'response'`` (the mean) or ``'link'`` (the linear predictor).

    Returns
    -------
    ``(V, N)`` predictions.  Differentiable w.r.t. ``X`` / ``z``.
    """
    eta = _conditional_eta(
        result.beta_hat,
        jnp.asarray(X, result.beta_hat.dtype),
        level=level,
        blups=result.blups,
        z=z,
        group=group,
    )
    if type == 'link':
        return eta
    if type == 'response':
        return result.family.linkinv(eta)
    raise ValueError(
        f"glmm_predict: type={type!r}; expected 'response' or 'link'."
    )
