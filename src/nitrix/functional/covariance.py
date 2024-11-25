# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable estimation of covariance and derived measures.

Functional connectivity is a measure of the statistical relationship between
(localised) time series signals. It is typically operationalised as some
value derived from the covariance between the time series, most often the
correlation coefficient.

::: {.callout-warning}
## Untested use cases
Several use cases are not yet tested, including:

- Off-diagonal weighted covariance. This use case can occur, for example, in
  the estimation of lagged covariance (often a Toeplitz matrix weight) in time
  series analysis.
- Complex-valued partial correlation. A reference implementation is not yet
  available for this use case.
- General correctness of computations involving complex numbers and weights
  beyond the most basic operations.
:::
"""

from typing import Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .._internal import (
    NestedDocParse,
    Tensor,
    _conform_bform_weight,
    tensor_dimensions,
)


def document_covariance(func: callable) -> callable:
    param_spec = """
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    weight : $obs$ or ($obs$, $obs$) tensor or None (default None)
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal
        coupling for a time series covariance.
    l2 : nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of
        the covariance matrix. This can be set to a positive value to obtain
        an intermediate for estimating the regularised inverse covariance."""
    unary_param_spec = """
    X : ($N$, $*$, $C$, $obs$) or ($N$, $*$, $obs$, $C$) tensor
        Tensor containing a sample of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice
        along the penultimate axis corresponds to a data channel or more
        generally a variable."""
    binary_param_spec = """
    X : ($N$, $*$, $C$, $obs$) or ($N$, $*$, $obs$, $C$) tensor
        Tensor containing a sample of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice
        along the penultimate axis corresponds to a data channel or more
        generally a variable.
    Y : ($N$, $*$, $C_Y$, $obs$) or ($N$, $*$, $obs$, $C_Y$) tensor
        Tensor containing a sample of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice
        along the penultimate axis corresponds to a data channel or more
        generally a variable."""
    conditional_param_spec = """
    X : ($N$, $*$, $C$, $obs$) or ($N$, $*$, $obs$, $C$) tensor
        Tensor containing samples of multivariate observations, with those
        variables whose influence we wish to control for removed and separated
        out into tensor Y. Each slice along the last axis corresponds to an
        observation, and each slice along the penultimate axis corresponds to
        a data channel or more generally a variable.
    Y : ($N$, $*$, $C_Y$, $obs$) or ($N$, $*$, $obs$, $C_Y$) tensor
        Tensor containing samples of multivariate observations, limited to
        nuisance or confound variables whose influence we wish to control for.
        Each slice along the last axis corresponds to an observation, and each
        slice along the penultimate axis corresponds to a data channel or more
        generally a variable."""
    inverting_param_spec = """
    require_nonsingular : bool
        Indicates that the covariance must be nonsingular. If this is False,
        then the Moore-Penrose pseudoinverse is computed instead of the
        inverse."""
    cov_param_warning = """
    :::{.callout-important}
    ## Danger
    The ``l2`` parameter has no effect on this function. It is included
    only for conformance with the ``cov`` function.
    :::"""
    desc_obs = (
        'Number of observations per data channel. For example, if `X` '
        'is a time series tensor, then $obs$ could be the number of time '
        'points.'
    )
    desc_C = (
        'Number of variables or data channels to be correlated. For example, '
        'if `X` is a BOLD time series tensor, then this is the number of '
        'spatial loci.'
    )
    desc_C_Y = (
        'Number of variables or data channels in the second tensor to be '
        'correlated. For example, if `Y` is a BOLD time series tensor, then '
        'this is the number of spatial loci.'
    )
    desc_cC_Y = (
        'Number of variables or data channels to be conditioned on. For '
        'example, if `Y` is a confound time series tensor, then $C_Y$ is the '
        'number of confound time series.'
    )
    dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $N$ | Batch size  | Optional |
    | $C$ | {desc_C}   ||{extra_channels}
    |$obs$| {desc_obs} | Order controlled by `rowvar` |
    |$*$  | Any number of intervening dimensions ||
    """
    unary_dim_spec = dim_spec.format(
        desc_C=desc_C,
        desc_obs=desc_obs,
        extra_channels='',
    )
    binary_dim_spec = dim_spec.format(
        desc_C=desc_C,
        desc_obs=desc_obs,
        extra_channels=f'\n    |$C_Y$| {desc_C_Y}||',
    )
    conditional_dim_spec = dim_spec.format(
        desc_C=desc_C,
        desc_obs=desc_obs,
        extra_channels=f'\n    |$C_Y$| {desc_cC_Y}||',
    )
    unary_dim_spec = tensor_dimensions(unary_dim_spec)
    binary_dim_spec = tensor_dimensions(binary_dim_spec)
    conditional_dim_spec = tensor_dimensions(conditional_dim_spec)

    corr_long_description = r"""
    The correlation is obtained via normalisation of the covariance. Given a
    covariance matrix
    $\hat{\Sigma} \in \mathbb{R}^{n \times n}$, each
    entry of the correlation matrix
    $R \in \mathbb{R}^{n \times n}$
    is defined according to

    $R_{ij} = \frac{\hat{\Sigma}_{ij}}{\sqrt{\hat{\Sigma}_{ii}} \sqrt{\hat{\Sigma}_{jj}}}$"""

    conditional_cov_long_description = r"""

    The conditional covariance is the covariance of one set of variables X
    conditioned on another set of variables Y. The conditional covariance is
    computed from the covariance $\Sigma_{{XX}}$ of X,
    the covariance $\Sigma_{{YY}}$ of Y, and the
    covariance $\Sigma_{{XY}}$ between X and Y.
    It is defined as the Schur complement of $\Sigma_{{YY}}$:

    $\Sigma_{X|Y} = \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{{-1}} \Sigma_{XY}^\intercal$

    The conditional covariance is equivalent to the covariance of the first set
    of variables after residualising them with respect to the second set of
    variables (plus an intercept term). This can be interpreted as the
    covariance of variables of interest (the first set) after controlling for
    the effects of confounds or nuisance variables (the second set)."""

    fmt = NestedDocParse(
        param_spec=param_spec,
        unary_param_spec=unary_param_spec,
        binary_param_spec=binary_param_spec,
        conditional_param_spec=conditional_param_spec,
        inverting_param_spec=inverting_param_spec,
        cov_param_warning=cov_param_warning,
        unary_dim_spec=unary_dim_spec,
        binary_dim_spec=binary_dim_spec,
        conditional_dim_spec=conditional_dim_spec,
        corr_long_description=corr_long_description,
        conditional_cov_long_description=conditional_cov_long_description,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_covariance
def cov(
    X: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weight: Optional[Tensor] = None,
    l2: float = 0,
) -> Tensor:
    """
    Empirical covariance of variables in a tensor batch.
    \
    {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : $(N, *, C, C)$ tensor
        Empirical covariance matrix of the variables in the input tensor.

    See also
    --------
    [pairedcov](nitrix.functional.covariance.pairedcov.html)
        Covariance among variables in 2 tensors
    [corr](nitrix.functional.covariance.corr.html)
        Normalised covariance matrix (Parametric correlation matrix)
    [precision](nitrix.functional.covariance.precision.html)
        Inverse covariance (precision) matrix
    [partialcorr](nitrix.functional.covariance.partialcorr.html)
        Partial correlation matrix
    """
    X = _prepare_input(X, rowvar)
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), weight)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - avg
    sigma = _cov_inner(X0, X0, weight, w_type, fact)
    if l2 > 0:
        sigma = sigma + l2 * jnp.eye(X.shape[-2])
    return sigma


@document_covariance
def corr(
    X: Tensor,
    **params,
) -> Tensor:
    """
    Parametric correlation of variables in a tensor batch.
    \
    {corr_long_description}
    \
    {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {param_spec}

    Returns
    -------
    R : $(N, *, C, C)$ tensor
        Parametric correlation matrix of the variables in the input tensor.

    See also
    --------
    [cov](nitrix.functional.covariance.cov.html)
        Empirical covariance matrix
    [partialcorr](nitrix.functional.covariance.partialcorr.html)
        Partial correlation matrix
    [conditionalcorr](nitrix.functional.covariance.conditionalcorr.html)
        Conditional correlation matrix
    """
    sigma = cov(X, **params)
    fact = corrnorm(sigma)
    return sigma / fact


@document_covariance
def partialcov(
    X: Tensor,
    require_nonsingular: bool = True,
    **params,
) -> Tensor:
    """
    Partial covariance of variables in a tensor batch.

    The partial covariance is obtained by conditioning the covariance of each
    pair of variables on all other observed variables. It can be interpreted
    as a measurement of the direct relationship between each variable pair.
    The partial covariance is computed via inversion of the covariance matrix,
    followed by negation of off-diagonal entries.
    \
    {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : $(N, *, C, C)$ tensor
        Partial covariance matrix of the variables in the input tensor.

    See also
    --------
    [cov](nitrix.functional.covariance.cov.html)
        Empirical covariance matrix
    [precision](nitrix.functional.covariance.precision.html)
        Inverse covariance (precision) matrix
    [partialcorr](nitrix.functional.covariance.partialcorr.html)
        Partial correlation matrix
    [conditionalcov](nitrix.functional.covariance.conditionalcov.html)
        Conditional covariance matrix
    """
    omega = precision(X, require_nonsingular=require_nonsingular, **params)
    omega = omega * (2 * jnp.eye(omega.shape[-1]) - 1)
    return omega


@document_covariance
def partialcorr(
    X: Tensor,
    require_nonsingular: bool = True,
    **params,
) -> Tensor:
    """
    Partial correlation coefficient of variables in a tensor batch.

    The partial correlation is obtained by conditioning the covariance of each
    pair of variables on all other observed variables. It can be interpreted
    as a measurement of the direct relationship between each variable pair.
    The partial correlation is efficiently computed via successive inversion
    and normalisation of the covariance matrix, accompanied by negation of
    off-diagonal entries.
    \
    {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    R : $(N, *, C, C)$ tensor
        Matrix of partial correlation coefficients of the variables in the
        input tensor.

    See also
    --------
    [corr](nitrix.functional.covariance.corr.html)
        Normalised covariance matrix (Parametric correlation matrix)
    [partialcov](nitrix.functional.covariance.partialcov.html)
        Partial covariance matrix
    [conditionalcorr](nitrix.functional.covariance.conditionalcorr.html)
        Conditional correlation matrix
    """
    omega = partialcov(X, require_nonsingular=require_nonsingular, **params)
    fact = corrnorm(omega)
    return omega / fact


@document_covariance
def pairedcov(
    X: Tensor,
    Y: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weight: Optional[Tensor] = None,
    l2: float = 0,
) -> Tensor:
    """
    Empirical covariance between two sets of variables.

    {cov_param_warning}
    \
    {binary_dim_spec}

    Parameters
    ----------\
    {binary_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : $(N, *, C, C_Y)$ tensor
        Covariance matrix of the variables in the input tensors.

    See also
    --------
    [cov](nitrix.functional.covariance.cov.html)
        Empirical covariance matrix
    [pairedcorr](nitrix.functional.covariance.pairedcorr.html)
        Correlation among variables in 2 tensors
    [conditionalcov](nitrix.functional.covariance.conditionalcov.html)
        Conditional covariance matrix
    """
    X = _prepare_input(X, rowvar)
    Y = _prepare_input(Y, rowvar)
    weight, w_type, w_sum, (Xavg, Yavg) = _prepare_weight_and_avg(
        (X, Y), weight
    )
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - Xavg
    Y0 = Y - Yavg
    return _cov_inner(X0, Y0, weight, w_type, fact)


@document_covariance
def pairedcorr(
    X: Tensor,
    Y: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    **params,
) -> Tensor:
    """
    Empirical parametric correlation between variables in two tensor batches.
    \
    {corr_long_description}

    {cov_param_warning}
    \
    {binary_dim_spec}

    Parameters
    ----------\
    {binary_param_spec}\
    {param_spec}

    Returns
    -------
    R : $(N, *, C, C_Y)$ tensor
        Paired parametric correlation matrix of the variables in the input
        tensor.
    """
    inddof = ddof
    if inddof is None:
        inddof = 1 - bias
    varX = X.var(-1, keepdims=True, ddof=inddof)
    varY = Y.var(-1, keepdims=True, ddof=inddof)
    fact = jax.lax.sqrt(varX @ varY.swapaxes(-2, -1))
    return (
        pairedcov(X, Y, rowvar=rowvar, bias=bias, ddof=ddof, **params) / fact
    )


@document_covariance
def conditionalcov(
    X: Tensor,
    Y: Tensor,
    require_nonsingular: bool = True,
    **params,
) -> Tensor:
    """
    Conditional covariance of variables in a tensor batch.
    \
    {conditional_cov_long_description}
    \
    {conditional_dim_spec}

    Parameters
    ----------\
    {conditional_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : $(N, *, C, C)$ tensor
        Conditional empirical covariance matrix of the variables in input
        tensor X conditioned on the variables in input tensor Y.

    See also
    --------
    [conditionalcorr](nitrix.functional.covariance.conditionalcorr.html)
        Conditional correlation matrix
    [partialcov](nitrix.functional.covariance.partialcov.html)
        Partial covariance matrix
    """
    A = cov(X, **params)
    B = pairedcov(X, Y, **params)
    C_inv = precision(Y, require_nonsingular=require_nonsingular, **params)
    return A - B @ C_inv @ B.swapaxes(-1, -2)


@document_covariance
def conditionalcorr(
    X: Tensor,
    Y: Tensor,
    require_nonsingular: bool = True,
    **params,
) -> Tensor:
    """
    Conditional correlation of variables in a tensor batch.
    \
    {conditional_cov_long_description}
    \
    {corr_long_description}
    \
    {conditional_dim_spec}

    Parameters
    ----------\
    {conditional_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : $(N, *, C, C)$ tensor
        Conditional empirical covariance matrix of the variables in input
        tensor X conditioned on the variables in input tensor Y.

    See also
    --------
    [conditionalcov](nitrix.functional.covariance.conditionalcov.html)
        Conditional covariance matrix
    [partialcorr](nitrix.functional.covariance.partialcorr.html)
        Partial correlation matrix
    """
    sigma = conditionalcov(
        X, Y, require_nonsingular=require_nonsingular, **params
    )
    fact = corrnorm(sigma)
    return sigma / fact


@document_covariance
def precision(
    X: Tensor,
    require_nonsingular: bool = True,
    **params,
) -> Tensor:
    """
    Empirical precision of variables in a tensor batch.

    The precision matrix is the inverse of the covariance matrix.

    ::: {{.callout-note}}
    The precision matrix is not defined for singular covariance matrices.
    If the number of input observations is less than the number of
    variables, the covariance matrix can be regularised to ensure it is
    non-singular. This is done by setting the ``l2`` parameter to a
    positive value. Alternatively, the ``require_nonsingular`` parameter
    can be set to `False` to use the Moore-Penrose pseudoinverse of the
    covariance matrix, but this setting is not recommended for most
    applications. Unless there is a specific reason to do this, regularise
    the computation instead.
    :::
    \
    {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    omega : $(N, *, C, C)$ tensor
        Precision matrix of the variables in input tensor X.

    See also
    --------
    [cov](nitrix.functional.covariance.cov.html)
        Empirical covariance matrix
    [partialcov](nitrix.functional.covariance.partialcov.html)
        Partial covariance matrix
    [pairedcorr](nitrix.functional.covariance.pairedcorr.html)
        Correlation among variables in 2 tensors
    """
    sigma = cov(X, **params)
    if require_nonsingular:
        return jnp.linalg.inv(sigma)
    else:
        return jnp.linalg.pinv(sigma)


def corrnorm(A: Tensor) -> Tensor:
    """
    Normalisation term for the correlation coefficient.

    Parameters
    ----------
    A : Tensor
        Batch of covariance or unnormalised correlation matrices.

    Returns
    -------
    normfact: Tensor
        Normalisation term for each element of the input tensor. Dividing by
        this will yield the normalised correlation.
    """
    d = jnp.diagonal(A, axis1=-2, axis2=-1)
    fact = -jax.lax.sqrt(d)[..., None]
    return fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps


def covariance(*pparams, **params):
    """Alias for [`cov`](nitrix.functional.covariance.cov.qmd)."""
    return cov(*pparams, **params)


def correlation(*pparams, **params):
    """Alias for [`corr`](nitrix.functional.covariance.corr.qmd)."""
    return corr(*pparams, **params)


def corrcoef(*pparams, **params):
    """Alias for [`corr`](nitrix.functional.covariance.corr.qmd)."""
    return corr(*pparams, **params)


def pcorr(*pparams, **params):
    """
    Alias for [`partialcorr`](nitrix.functional.covariance.partialcorr.qmd).
    """
    return partialcorr(*pparams, **params)


def ccov(*pparams, **params):
    """
    Alias for
    [`conditionalcov`](nitrix.functional.covariance.conditionalcov.qmd).
    """
    return conditionalcov(*pparams, **params)


def ccorr(*pparams, **params):
    """
    Alias for
    [`conditionalcorr`](nitrix.functional.covariance.conditionalcorr.qmd).
    """
    return conditionalcorr(*pparams, **params)


def _prepare_input(X: Tensor, rowvar: bool = True) -> Tensor:
    """
    Ensure that the input is conformant with the transposition expected by the
    covariance function.
    """
    X = jnp.atleast_2d(X)
    if (not rowvar) and (X.shape[-2] != 1):
        X = X.swapaxes(-1, -2)
    return X


def _prepare_weight_and_avg(
    vars: Sequence[Tensor],
    weight: Optional[Tensor] = None,
) -> Tuple[
    Optional[Tensor],
    Optional[Literal['vector', 'matrix']],
    Union[float, int],
    Sequence[Tensor],
]:
    """
    Set the weights for the covariance computation based on user input and
    determine the sum of weights for normalisation. If weights are not
    provided, the sum is simply the count of data observations. Compute the
    first moment or its weighted analogue.
    """
    if weight is not None:
        if weight.ndim == 1 or weight.shape[-1] != weight.shape[-2]:
            w_type = 'vector'
            weight = _conform_bform_weight(weight)
            w_sum = weight.sum(-1, keepdims=True)
            avg = [(V * (weight / w_sum)).sum(-1, keepdims=True) for V in vars]
        else:
            w_type = 'matrix'
            w_sum = weight.sum((-1, -2), keepdims=True)
            # TODO
            # We'll need to ensure that this is correct
            # for the nondiagonal case. The tests still don't.
            avg = [(V @ (weight / w_sum)).sum(-1, keepdims=True) for V in vars]
    else:
        w_type = None
        w_sum = vars[0].shape[-1]
        avg = [V.mean(-1, keepdims=True) for V in vars]
    return weight, w_type, w_sum, avg


def _prepare_denomfact(
    w_sum: Union[float, int],
    w_type: Optional[Literal['vector', 'matrix']] = 'matrix',
    ddof: Optional[int] = None,
    bias: bool = False,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Determine the factor we should divide by to obtain the (un)biased
    covariance from the sum over observations.
    """
    if ddof is None:
        ddof = int(not bias)
    if weight is None:
        fact = w_sum - ddof
    elif ddof == 0:
        fact = w_sum
    elif w_type == 'vector':
        fact = w_sum - ddof * (weight**2).sum(-1, keepdims=True) / w_sum
    else:
        # TODO
        # I don't have the intuition here yet: should this be
        # weight * weight or weight @ weight ? Or something else?
        # This affects only the nondiagonal case.
        if not _is_diagonal(weight):
            # TODO: Danger! This can be competely bypassed and the function
            # executed with a non-diagonal weight matrix. This fails silently
            # and produces incorrect results.
            # This will happen if the function is JIT compiled with a diagonal
            # weight matrix and then invoked with a nondiagonal one.
            raise NotImplementedError(
                'Nondiagonal weight matrices are not yet supported.'
            )
        fact = (
            w_sum
            - ddof
            * (
                # weight.sum(-1, keepdims=True) ** 2
                weight @ weight.swapaxes(-1, -2)
            ).sum((-1, -2), keepdims=True)
            / w_sum
        )
    return fact


def _cov_inner(
    X: Tensor,
    Y: Tensor,
    weight: Optional[Tensor],
    w_type: Optional[Literal['vector', 'matrix']],
    fact: Union[float, int, Tensor],
) -> Tensor:
    Y = Y.conj()
    if weight is None:
        sigma = X @ Y.swapaxes(-1, -2) / fact
    elif w_type == 'vector':
        sigma = (X * weight) @ Y.swapaxes(-1, -2) / fact
    else:
        sigma = X @ weight @ Y.swapaxes(-1, -2) / fact
    return sigma


def _is_diagonal(X: Tensor) -> bool:
    batch_dims = X.shape[:-2]
    return (
        X.reshape(*batch_dims, -1)[..., :-1]
        .reshape(
            *batch_dims,
            X.shape[-2] - 1,
            X.shape[-1] + 1,
        )[..., 1:]
        .sum()
        == 0
    )
