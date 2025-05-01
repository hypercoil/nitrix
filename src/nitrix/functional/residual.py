# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise tensor block via least squares.
"""

from typing import Any, Callable, Literal

import jax.numpy as jnp

from .._internal import (
    NestedDocParse,
    Tensor,
    broadcast_ignoring,
    tensor_dimensions,
    vmap_over_outer,
)


def document_linreg(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    regress_warning = """
    :::{.callout-warning}
    When testing an old ``torch``-based implementation of this operation,
    we have found in some cases that the least-squares fit returned was
    incorrect for reasons that are not  clear. (Incorrect results were
    returned by
    ``torch.linalg.lstsq``, although correct results were returned if
    ``torch.linalg.pinv`` was used instead.) Verify that results are
    reasonable when using this operation.

    It is not clear whether the same is true for ``jax``. Caution is
    advised.
    :::

    :::{.callout-note}
    The [conditional covariance](nitrix.functional.covariance.conditionalcov.html)
    or [conditional correlation](nitrix.functional.covariance.conditionalcorr.html)
    may be used instead where appropriate.
    :::"""

    tensor_dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $N$ | Batch size  | Optional |
    |$C_X$| {desc_C_X} ||
    |$C_Y$| {desc_C_Y} ||
    |$obs$| {desc_obs} | Order controlled by `rowvar` |
    | $*$ | Any number of intervening dimensions ||
    """.format(
        desc_C_X=(
            'Number of explanatory variables or data channels. For example, '
            'if `X` is a confound time series tensor, then $C_X$ is the '
            'number of confound time series.'
        ),
        desc_C_Y=(
            'Number of data channels or variables. For example, if `Y` is a '
            'BOLD time series tensor, then $C_Y$ is the number of spatial '
            'loci.'
        ),
        desc_obs=(
            'Number of observations per data channel. For example, if `X` '
            'and `Y` are time series tensors, then $obs$ is the number of '
            'time points.'
        ),
    )
    regress_dim = tensor_dimensions(tensor_dim_spec)

    regress_param_spec = """
    rowvar : bool
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then
        this relationship is transposed.
    l2 : float
        L2 regularisation parameter. If non-zero, the least-squares solution
        will be regularised by adding a penalty term to the cost function.
    return_mode : Literal[`residual`, `projection`]
        Indicates whether the residual or projection tensor should be
        returned. The `projection` tensor is the projection of `Y` onto the
        span of `X` (i.e., the least-squares solution).
    proj_atol : float
        Absolute tolerance for determining whether the projection tensor is
        numerically close to `Y`. See `jax.numpy.isclose` for details.
    proj_rtol : float
        Relative tolerance for determining whether the projection tensor is
        numerically close to `Y`. See `jax.numpy.isclose` for details."""
    fmt = NestedDocParse(
        regress_warning=regress_warning,
        regress_dim=regress_dim,
        regress_param_spec=regress_param_spec,
    )
    f.__doc__ = (f.__doc__ or '').format_map(fmt)
    return f


@document_linreg
def residualise(
    Y: Tensor,
    X: Tensor,
    rowvar: bool = True,
    l2: float = 0.0,
    return_mode: Literal['residual', 'projection'] = 'residual',
    proj_atol: float = 1e-8,
    proj_rtol: float = 1e-5,
) -> Tensor:
    """
    Residualise a tensor block via ordinary linear least squares.
    \
    {regress_warning}
    \
    {regress_dim}

    Parameters
    ----------
    Y : ($N$, $*$, $C_Y$, $obs$) or ($N$, $*$, $obs$, $C_Y$) tensor
        Tensor to be residualised or orthogonalised with respect to `X`. The
        vector of observations in each channel is projected into a subspace
        orthogonal to the span of `X`.
    X : ($N$, $*$, $C_X$, $obs$) or ($N$, $*$, $obs$, $C_X$) tensor
        Tensor containing explanatory variables. Any variance in `Y` that can
        be explained by variables in `X` will be removed from `Y`.
    {regress_param_spec}

    Returns
    -------
    ($N$, $*$, $C_Y$, $obs$) or ($N$, $*$, $obs$, $C_Y$) tensor
        The residual or projection tensor, depending on the `return_mode`
        parameter.
    """
    if rowvar:
        X_in = X.swapaxes(-1, -2)
        Y_in = Y.swapaxes(-1, -2)
    else:
        X_in, Y_in = X, Y
    if l2 > 0.0:
        X_reg = jnp.eye(X_in.shape[-1]) * l2
        Y_reg = jnp.zeros((X_in.shape[-1], Y_in.shape[-1]))
        X_in, X_reg = broadcast_ignoring(X_in, X_reg, -2)
        Y_in, Y_reg = broadcast_ignoring(Y_in, Y_reg, -2)
        X_in = jnp.concatenate((X_in, X_reg), axis=-2)
        Y_in = jnp.concatenate((Y_in, Y_reg), axis=-2)

    X_in, Y_in = broadcast_ignoring(X_in, Y_in, -1)
    fit = vmap_over_outer(jnp.linalg.lstsq, 2)
    betas = fit((X_in, Y_in))[0]
    if rowvar:
        proj = betas.swapaxes(-1, -2) @ X
    else:
        proj = X @ betas
    # Handle numerical error in projection
    close_params = {'atol': proj_atol, 'rtol': proj_rtol}
    if return_mode == 'residual':
        return jnp.where(jnp.isclose(proj, Y, **close_params), 0, Y - proj)
    elif return_mode == 'projection':
        return jnp.where(jnp.isclose(proj, Y, **close_params), Y, proj)
    else:
        raise ValueError(f'Invalid return_mode: {return_mode}')
