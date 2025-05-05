# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Computations on tensors structured by Euclidean, spherical, point-set, and
graph geometries.

These computations include utilities for computing centres of mass, geodesics,
convolutions, and derived quantities.
"""

import warnings
from functools import partial
from typing import Any, Optional, Sequence

import jax.numpy as jnp

from .._internal import Tensor
from .._internal.docutil import (
    DocTemplateFormat,
    form_docstring,
    tensor_dimensions,
)


@form_docstring
def document_cmass_regular_grid() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    |$k_i$| {desc_k_i} ||
    | $*$ | Any number of prefix, intervening, and suffix dimensions ||
    | $d$ | Dimension of the embedding space ||
    """.format(
        desc_k_i=(
            'Dimension corresponding to the $i$-th axis of the grid over '
            'which the input tensor dataset is defined.'
        ),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@form_docstring
def document_cmass_coor() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix, intervening, and suffix dimensions ||
    | $W$ | {desc_W} | {desc_W_notes} |
    | $L$ | {desc_L} | {desc_L_notes} |
    | $D$ | {desc_D} | {desc_D_notes} |
    """.format(
        desc_W=('Number of weight distributions over the coordinates.'),
        desc_W_notes=(
            'For example, if the input tensor represents a set of weights '
            'over regions of an atlas, then $W$ is the number of regions.'
        ),
        desc_L=('Number of coordinates.'),
        desc_L_notes=(
            'For example, if the input tensor represents a set of weights '
            'over regions of an atlas, then $L$ is the number of voxels or '
            'mesh vertices.'
        ),
        desc_D='Dimension of the embedding space.',
        desc_D_notes=(
            'For example, if the input tensor represents a set of weights '
            'over a mesh embedded in 3-dimensional space, then $D$ is 3.'
        ),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@form_docstring
def document_sphere_coordinate_change() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $2$ | {desc_latlong} ||
    | $3$ | {desc_normal} ||
    """.format(
        desc_latlong=('Latitude and longitude coordinates.'),
        desc_normal=('Normal vector coordinates.'),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    normal_coor_spec = """
    coor : ($*$, 3) tensor
        Tensor containing 3-tuple coordinates indicating x, y, and z values of
        each point on a sphere whose centre is the origin."""
    latlong_coor_spec = """
    coor : ($*$, 2) tensor
        Tensor containing 2-tuple coordinates indicating the latitude and
        longitude of each point.
    """
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
        normal_coor_spec=normal_coor_spec,
        latlong_coor_spec=latlong_coor_spec,
    )
    return fmt


@form_docstring
def document_spherical_geodesic() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $N_X$ | {desc_N_X} ||
    | $N_Y$ | {desc_N_Y} ||
    | $3$ | {desc_D} | {desc_D_notes} |
    """.format(
        desc_N_X=('Number of coordinates in X.'),
        desc_N_Y=('Number of coordinates in Y.'),
        desc_D=('Dimension of the embedding space.'),
        desc_D_notes=('This must be 3.'),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@form_docstring
def document_spatial_conv() -> DocTemplateFormat:
    tensor_dim_spec = """

    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $N$ | {desc_N} ||
    | $C$ | {desc_C} ||
    | $D$ | {desc_D} ||
    """.format(
        desc_N=('Number of coordinates.'),
        desc_C=('Number of data channels.'),
        desc_D=('Dimension of the embedding space.'),
    )
    tensor_dim_spec = tensor_dimensions(tensor_dim_spec)
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
    )
    return fmt


@document_cmass_regular_grid
def cmass_regular_grid(
    X: Tensor,
    axes: Optional[Sequence[int]] = None,
    na_rm: bool = False,
) -> Tensor:
    r"""
    Differentiably compute a weight's centre of mass, interpreting the values
    of the input tensor as a regular grid on the tensor coordinates.

    This can, for example, be used to regularise the weight so that its centre
    of mass is close to a provided coordinate.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    X : ($*$, $k_1$, $*$, $k_2$, $*$, ..., $*$, $k_n$, $*$) tensor
        Tensor containing the weights whose centres of mass are to be
        computed.
    axes : iterable or None (default None)
        Axes of the input tensor that together define each slice of the tensor
        within which a single centre-of-mass vector is computed. If this is
        set to None, then the centre of mass is computed across all axes. If
        this is [-3, -2, -1], then the centre of mass is computed separately
        for each 3-dimensional slice spanned by the last three axes of the
        tensor.

        The value of this parameter configures the expected shape of the input
        tensor by specifying the axes $k_i$ over which the centre of mass is
        computed.
    na_rm : float or False (default False)
        If any single slice of the input tensor has zero mass, then the centre
        of mass within that slice is undefined and populated with NaN. The
        `na_rm` parameter specified how such undefined values are handled. If
        this is False, then NaN values are left intact; if this is a float,
        then NaN values are replaced by the specified float.

    Returns
    -------
    cmass : ($*$, $d$, $*$) tensor
        Centre of mass vectors for each slice from the input tensor. The
        coordinates are ordered according to the specification in ``axes``.

    See also
    --------
    [cmass_coor](nitrix.functional.geom.cmass_coor.html)
        Centre of mass of a weight at specified coordinates
    """
    dim = X.shape
    ndim = X.ndim
    all_axes = list(range(ndim))
    axes = [all_axes[ax] for ax in axes] if axes is not None else all_axes
    axes = tuple(axes)
    out_dim = [s for ax, s in enumerate(dim) if all_axes[ax] not in axes]
    out_dim += [len(axes)]
    out = jnp.zeros(out_dim)
    for i, ax in enumerate(axes):
        coor = jnp.arange(1, X.shape[ax] + 1)
        while coor.ndim < ndim - all_axes[ax]:
            coor = coor[..., None]
        num = (coor * X).sum(axes)
        denom = X.sum(axes)
        out = out.at[..., i].set(num / denom - 1)
        if na_rm is not False:
            out = jnp.where(denom == 0, na_rm, out)
    return out


def cmass(
    X: Tensor,
    axes: Optional[Sequence[int]] = None,
    na_rm: bool = False,
) -> Tensor:
    warnings.warn(
        'This function is deprecated due to ambiguous naming. Use '
        'cmass_regular_grid instead.',
        DeprecationWarning,
    )
    return cmass_regular_grid(X, axes=axes, na_rm=na_rm)


@document_cmass_coor
def cmass_coor(
    X: Tensor,
    coor: Tensor,
    radius: Optional[float] = None,
) -> Tensor:
    r"""
    Differentiably compute a weight's centre of mass.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    X : ($*$, $W$, $L$) tensor
        Weight whose centre of mass is to be computed.
    coor : ($*$, $D$, $L$) tensor
        Coordinates corresponding to each column (location/voxel) in X.
    radius : float or None (default None)
        If this is not None, then the computed centre of mass is projected
        onto a sphere with the specified radius.

    Returns
    -------
    cmass : ($*$, $D$, $W$) tensor
        Tensor containing the coordinates of the centre of mass of each row of
        input X. Coordinates are ordered as in the second-to-last axis of
        ``coor``.

    See also
    --------
    [cmass_regular_grid](nitrix.functional.geom.cmass_regular_grid.html)
        Centre of mass of a weight whose values are interpreted over a regular
        grid on the tensor coordinates
    """
    num = jnp.einsum('...wl,...dl->...dw', X, coor)
    denom = X.sum(-1)
    if radius is not None:
        cmass_euc = num / denom
        return (
            radius
            * cmass_euc
            / jnp.linalg.norm(
                cmass_euc,
                ord=2,
                axis=-2,
            )
        )
    return num / denom


def cmass_reference_displacement_grid(
    weight: Tensor,
    refs: Tensor,
    axes: Optional[Sequence[int]] = None,
    na_rm: bool = False,
) -> Tensor:
    """
    Displacement of centres of mass from reference points -- grid version.

    See :func:`cmass_regular_grid` for parameter specifications.
    """
    cm = cmass_regular_grid(weight, axes=axes, na_rm=na_rm)
    return cm - refs


def cmass_reference_displacement_coor(
    weight: Tensor,
    refs: Tensor,
    coor: Tensor,
    radius: Optional[float] = None,
) -> Tensor:
    """
    Displacement of centres of mass from reference points -- explicit
    coordinate version.

    See :func:`cmass_coor` for parameter specifications.
    """
    cm = cmass_coor(weight, coor, radius=radius)
    return cm - refs


# TODO: Switch to using kernel.gaussian_kernel instead of this everywhere
def kernel_gaussian(x: Tensor, scale: float = 1) -> Tensor:
    """
    An example of an isotropic kernel. Zero-centered Gaussian kernel with
    specified scale parameter.
    """
    return jnp.exp(-((x / scale) ** 2) / 2) / (scale * (2 * jnp.pi) ** 0.5)


@document_sphere_coordinate_change
def sphere_to_normals(coor: Tensor, r: float = 1) -> Tensor:
    r"""
    Convert spherical coordinates from latitude/longitude format to normal
    vector format. Note this only works for 2-spheres as of now.
    \
    {tensor_dim_spec}

    Parameters
    ----------\
    {normal_coor_spec}
    r : float (default 1)
        Radius of the sphere.

    Returns
    -------\
    {latlong_coor_spec}
    """
    lat, lon = coor.swapaxes(0, -1)
    cos_lat = jnp.cos(lat)
    x = r * cos_lat * jnp.cos(lon)
    y = r * cos_lat * jnp.sin(lon)
    z = r * jnp.sin(lat)
    return jnp.stack((x, y, z), axis=-1)


@document_sphere_coordinate_change
def sphere_to_latlong(coor: Tensor) -> Tensor:
    r"""
    Convert spherical coordinates from normal vector format to latitude/
    longitude format. Note this only works for 2-spheres as of now.
    \
    {tensor_dim_spec}

    Parameters
    ----------\
    {normal_coor_spec}

    Returns
    -------\
    {latlong_coor_spec}
    """
    x, y, z = coor.swapaxes(0, -1)
    lat = jnp.arctan2(z, jnp.sqrt(x**2 + y**2))
    lon = jnp.arctan2(y, x)
    return jnp.stack((lat, lon), axis=-1)


@document_spherical_geodesic
def spherical_geodesic(
    X: Tensor,
    Y: Optional[Tensor] = None,
    r: float = 1,
) -> Tensor:
    r"""
    Geodesic great-circle distance between two sets of spherical coordinates
    formatted as normal vectors.

    This is not a haversine distance, although the result is identical. Please
    ensure that input vectors are expressed as normals and not as latitude/
    longitude pairs. Because this uses a cross-product in the computation, it
    works only with 2-spheres.
    \
    {tensor_dim_spec}

    Parameters
    ----------
    X : Tensor
        Tensor containing coordinates on a sphere formatted as surface-normal
        vectors in Euclidean coordinates. Distances are computed between each
        coordinate in X and each coordinate in Y.
    Y : Tensor or None (default X)
        As X. If a second tensor is not provided, then distances are computed
        between every pair of points in X.
    r : float
        Radius of the sphere. We could just get this from X or Y, but we
        don't.

    Returns
    -------
    dist : Tensor
        Tensor containing pairwise great-circle distances between each
        coordinate in X and each coordinate in Y.
    """
    if Y is None:
        Y = X
    if X.shape[-1] != 3:
        raise ValueError('X must have shape (*, N, 3)')
    if Y.shape[-1] != 3:
        raise ValueError('Y must have shape (*, N, 3)')
    X = X[..., None, :]
    Y = Y[..., None, :, :]
    X, Y = jnp.broadcast_arrays(X, Y)
    crXY = jnp.cross(X, Y, axis=-1)
    num = jnp.sqrt((crXY**2).sum(-1))
    denom = (X * Y).sum(-1)
    dist = jnp.arctan2(num, denom)
    dist = jnp.where(
        dist < 0,
        dist + jnp.pi,
        dist,
    )
    return dist * r


@document_spatial_conv
def spatial_conv(
    data: Tensor,
    coor: Tensor,
    kernel: callable = kernel_gaussian,
    metric: callable = spherical_geodesic,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
) -> Tensor:
    r"""
    Convolve data on a manifold with an isotropic kernel.

    Currently, this works by taking a dataset, a list of coordinates associated
    with each point in the dataset, an isotropic kernel, and a distance metric.
    It proceeds as follows:

    1. Using the provided metric, compute the distance between each pair of
       coordinates.
    2. Evaluate the isotropic kernel at each computed distance. Use this value
       to operationalise the loading weight of every coordinate on every other
       coordinate.
    3. Use the loading weights to perform a matrix product and obtain the
       kernel-convolved dataset.\
    {tensor_dim_spec}

    Parameters
    ----------
    data : ($*$, $N$, $C$) tensor
        Tensor containing data observations, which might be arrayed into
        channels.
    coor : ($*$, $N$, $D$) tensor
        Tensor containing the spatial coordinates associated with each
        observation in `data`.
    kernel : callable
        Function that maps a distance to a weight. Typically, shorter distances
        correspond to larger weights.
    metric : callable
        Function that takes as parameters two :math:`(*, N_i, D)` tensors
        containing coordinates and returns the pairwise distance between each
        pair of tensors. In Euclidean space, this could be the L2 norm; in
        spherical space, this could be the great-circle distance.
    max_bin : int
        Maximum number of points to include in a distance computation. If you
        run out of memory, try decreasing this.
    truncate : float or None (default None)
        Maximum distance at which data points can be convolved together.

    Returns
    -------
    data_conv : ($*$, $C$, $N$) tensor
        The input data convolved with the kernel. Each channel is convolved
        completely separately as of now.
    """
    start = 0
    end = min(start + max_bin, data.shape[-2])
    data_conv = jnp.zeros_like(data)
    while start < data.shape[-2]:
        # TODO: Won't work if the array is more than 2D.
        coor_block = coor[..., start:end, :]
        dist = metric(coor_block, coor)
        weight = kernel(dist)
        if truncate is not None:
            weight = jnp.where(dist > truncate, 0, weight)
        data_conv = data_conv.at[..., start:end, :].set(weight @ data)
        start = end
        end += max_bin
    return data_conv


def spherical_conv(
    data: Tensor,
    coor: Tensor,
    scale: float = 1,
    r: float = 1,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
):
    r"""
    Convolve data on a 2-sphere with an isotropic Gaussian kernel.

    This is implemented in the least clever of all possible ways, but it
    works. Here is a likely more efficient method that requires Lie groups or
    some such thing:
    https://openreview.net/pdf?id=Hkbd5xZRb

    See :func:`spatial_conv` for implementation details.

    :Dimension: **data :** :math:`(*, C, N)`
                    `*` denotes any number of intervening dimensions, `C`
                    denotes the number of data channels, and `N` denotes the
                    number of data observations per channel.
                **coor :** :math:`(*, N, D)`
                    `D` denotes the dimension of the space in which the data
                    are embedded.
                **Output :** :math:`(*, C, N)`

    Parameters
    ----------
    data : Tensor
        Tensor containing data observations, which might be arrayed into
        channels.
    coor : Tensor
        Tensor containing the spatial coordinates associated with each
        observation in `data`.
    scale : float (default 1)
        Scale parameter of the Gaussian kernel.
    r : float (default 1)
        Radius of the sphere.
    max_bin : int
        Maximum number of points to include in a distance computation. If you
        run out of memory, try decreasing this.
    truncate : float or None (default None)
        Maximum distance at which data points can be convolved together.

    Returns
    -------
    data_conv : Tensor
        The input data convolved with the kernel. Each channel is convolved
        completely separately as of now.
    """
    kernel = partial(kernel_gaussian, scale=scale)
    metric = partial(spherical_geodesic, r=r)
    return spatial_conv(
        data=data,
        coor=coor,
        kernel=kernel,
        metric=metric,
        max_bin=max_bin,
        truncate=truncate,
    )


def _euc_dist(X, Y=None):
    """
    Euclidean L2 norm metric.
    """
    X = X[..., None, :]
    Y = Y[..., None, :, :]
    X, Y = jnp.broadcast_arrays(X, Y)
    return jnp.sqrt(((X - Y) ** 2).sum(-1))


def euclidean_conv(
    data: Tensor,
    coor: Tensor,
    scale: float = 1,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
):
    """
    Spatial convolution using the standard L2 metric and a Gaussian kernel.

    See :func:`spatial_conv` for implementation details.
    """
    kernel = partial(kernel_gaussian, scale=scale)
    return spatial_conv(
        data=data,
        coor=coor,
        kernel=kernel,
        metric=_euc_dist,
        max_bin=max_bin,
        truncate=truncate,
    )


def diffuse(
    X: Tensor,
    coor: Tensor,
    norm: Any = 2,
    floor: float = 0,
    radius: Optional[float] = None,
) -> Tensor:
    r"""
    Compute a compactness score for a weight.

    The compactness is defined as

    :math:`\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{AC}{A\mathbf{1}} \right\|_{cols} \right)\mathbf{1}`

    :Dimension: **Input :** :math:`(*, W, L)`
                    ``*`` denotes any number of preceding dimensions, W
                    denotes number of weights (e.g., regions of an atlas),
                    and L denotes number of locations (e.g., voxels).
                **coor :** :math:`(*, D, L)`
                    D denotes the dimension of the embedding space of the
                    locations.

    Parameters
    ----------
    X : Tensor
        Weight for which the compactness score is to be computed.
    coor : Tensor
        Coordinates corresponding to each column (location/voxel) in X.
    norm
        Indicator of the type of norm to use for the distance function.
    floor : float (default 0)
        Any points closer to the centre of mass than the floor are assigned
        a compactness score of 0.
    radius : float or None (default None)
        If this is not None, then the centre of mass and distances are
        computed on a sphere with the specified radius.

    Returns
    -------
    float
        Measure of each weight's compactness about its centre of mass.
    """
    cm = cmass_coor(X, coor, radius=radius)
    if radius is None:
        dist = cm[..., None] - coor[..., None, :]
        dist = jnp.linalg.norm(dist, ord=norm, axis=-3)
    else:
        dist = spherical_geodesic(
            coor.swapaxes(-1, -2), cm.swapaxes(-1, -2), r=radius
        ).swapaxes(-1, -2)
    dist = jnp.maximum(dist - floor, 0)
    return (X * dist).mean(-1)
