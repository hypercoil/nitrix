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
from typing import Any, Callable, Literal, Optional, Sequence

import jax.numpy as jnp
from jax.nn import relu

from .matrix import delete_diagonal
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
    diffuse_penalty = """$\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{AC}{A\mathbf{1}} \right\|_{cols} \right)\mathbf{1}$"""
    fmt = DocTemplateFormat(
        tensor_dim_spec=tensor_dim_spec,
        diffuse_penalty=diffuse_penalty,
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


@form_docstring
def document_modularity() -> DocTemplateFormat:
    girvan_newman_long_desc = r"""
    The Girvan-Newman null model is defined as the expected connection weight
    between each pair of vertices if all edges are cut and the resulting stubs
    then randomly rewired. For the vector of node in-degrees
    :math:`k_i \in \mathbb{R}^I`, vector of node out-degrees
    :math:`k_o \in \mathbb{R}^O`, and total edge weight
    :math:`2m \in \mathbb{R}`, this yields the null model

    :math:`P_{GN} = \frac{1}{2m} k_i k_o^\intercal`

    or, in terms of the adjacency matrix :math:`A \in \mathbb{R}^{I \times O}`

    :math:`P_{GN} = \frac{1}{\mathbf{1}^\intercal A \mathbf{1}} A \mathbf{1} \mathbf{1}^\intercal A`"""
    modularity_matrix_long_desc = r"""
    The modularity matrix is defined as a normalised, weighted difference
    between the adjacency matrix and a suitable null model. For a weight
    :math:`\gamma`, an adjacency matrix :math:`A`, a null model :math:`P`, and
    total edge weight :math:`2m`, the modularity matrix is computed as

    :math:`B = \frac{1}{2m} \left( A - \gamma P \right)`"""
    coaffiliation_long_desc = r"""
    Given community affiliation matrices
    :math:`C^{(i)} \in \mathbb{R}^{I \times C}` for source nodes and
    :math:`C^{(o)} \in \mathbb{R}^{O \times C}` for sink nodes, and given a
    matrix of inter-community coupling coefficients
    :math:`\Omega \in \mathbb{R}^{C \times C}`, the coaffiliation
    :math:`H \in \mathbb{R}^{I \times O}` is computed as

    :math:`H = C^{(i)} \Omega C^{(o)\intercal}`"""
    relaxed_modularity_long_desc = r"""
    This relaxation supports non-deterministic assignments of vertices to
    communities and non-assortative linkages between communities. It reverts
    to standard behaviour when the inputs it is provided are standard.

    The relaxed modularity is defined as the sum of all entries in the
    Hadamard (elementwise) product between the modularity matrix and the
    coaffiliation matrix.

    :math:`Q = \mathbf{1}^\intercal \left( B \circ H \right) \mathbf{1}`"""
    adjacency_dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $I$ | {desc_I} | {note_I} |
    | $O$ | {desc_O} ||
    """.format(
        desc_I=('Number of vertices in the source set.'),
        desc_O=('Number of vertices in the sink set.'),
        note_I=(
            'If the same set of vertices emits and receives edges, then '
            '$I = O$. This is the case for any non-bipartite graph.'
        ),
    )
    coaffiliation_dim_spec = """
    | Dim | Description | Notes |
    |-----|-------------|-------|
    | $*$ | Any number of prefix dimensions ||
    | $I$ | {desc_I} | {note_I} |
    | $O$ | {desc_O} ||
    | $C_i$ | {desc_C_i} ||
    | $C_o$ | {desc_C_o} ||
    """.format(
        desc_I=('Number of vertices in the source set.'),
        desc_O=('Number of vertices in the sink set.'),
        desc_C_i=('Number of communities in the proposed partition for the '
                  'source set.'),
        desc_C_o=('Number of communities in the proposed partition for the '
                  'sink set.'),
        note_I=(
            'If the same set of vertices emits and receives edges, then '
            '$I = O$. This is the case for any non-bipartite graph.'
        ),
    )
    adjacency_dim_spec = tensor_dimensions(adjacency_dim_spec)
    coaffiliation_dim_spec = tensor_dimensions(coaffiliation_dim_spec)
    adjacency_param_spec = """
    A : ($*$, $I$, $O$) tensor
        Block of adjacency matrices for which the quantity of interest is to
        be computed."""
    modularity_matrix_param_spec = """
    gamma : nonnegative float (default 1)
        Resolution parameter for the modularity matrix. A smaller value assigns
        maximum modularity to partitions with large communities, while a larger
        value assigns maximum modularity to partitions with many small
        communities.
    null : callable(A) (default ``girvan_newman_null``)
        Function of ``A`` that returns, for each adjacency matrix in the input
        tensor block, a suitable null model. By default, the
        :doc:`Girvan-Newman null model <hypercoil.functional.graph.girvan_newman_null>`
        is used.
    normalise_modularity : bool (default False)
        Indicates that the resulting matrix should be normalised by the total
        matrix degree. This may not be necessary for many use cases -- for
        instance, where the arg max of a function of the modularity matrix is
        desired.
    sign : ``'+'``, ``'-'``, or None (default ``'+'``)
        Sign of connections to be considered in the modularity.
    **params
        Any additional parameters are passed to the null model."""
    coaffiliation_param_spec = r"""
    C : ($*$, $I$, $C_i$) tensor
        Community affiliation of vertices in the source set. Each slice is a
        matrix $C^{(i)} \in \mathbb{R}^{I \times C_i}$ that encodes the
        uncertainty in each vertex's community assignment. $C^{(i)}_{jk}$
        denotes the probability that vertex j is assigned to community k. If
        this is binary-valued, then it reflects a deterministic assignment.
    C_o : ($*$, $O$, $C_o$) tensor or None (default None)
        Community affiliation of vertices in the sink set. If None, then it is
        assumed that the source and sink sets are the same, and ``C_o`` is set
        equal to ``C_i``.
    L : Tensor or None (default None)
        The inter-community coupling matrix $\Omega$, mapping the
        probability of affiliation between communities. Each entry
        $L_{ij}$ encodes the probability of a vertex in community $i$
        connecting with a vertex in community $j$. If None, then a strictly
        assortative structure is assumed (equivalent to $L$ equals identity),
        under which nodes in the same community preferentially coaffiliate
        while nodes in different communities remain disaffiliated.
    exclude_diag : bool (default True)
        Indicates that self-links are not factored into the coaffiliation.
    normalise_coaffiliation : bool (default False)
        Normalise all community assignment weights to max out at 1."""

    fmt = DocTemplateFormat(
        girvan_newman_long_desc=girvan_newman_long_desc,
        modularity_matrix_long_desc=modularity_matrix_long_desc,
        coaffiliation_long_desc=coaffiliation_long_desc,
        relaxed_modularity_long_desc=relaxed_modularity_long_desc,
        adjacency_dim_spec=adjacency_dim_spec,
        coaffiliation_dim_spec=coaffiliation_dim_spec,
        adjacency_param_spec=adjacency_param_spec,
        modularity_matrix_param_spec=modularity_matrix_param_spec,
        coaffiliation_param_spec=coaffiliation_param_spec,
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
       kernel-convolved dataset.
    \
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


@document_spatial_conv
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
    \
    {tensor_dim_spec}

    Parameters
    ----------
    data : ($*$, $N$, $C$) tensor
        Tensor containing data observations, which might be arrayed into
        channels.
    coor : ($*$, $N$, $D$) tensor
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
    data_conv : ($*$, $C$, $N$) tensor
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


@document_cmass_coor
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
    {diffuse_penalty}
    \
    {cmass_coor_dim_spec}

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


@document_modularity
def girvan_newman_null(A: Tensor) -> Tensor:
    """
    Girvan-Newman null model for a tensor block.
    \
    {girvan_newman_long_desc}
    \
    {adjacency_dim_spec}

    Parameters
    ----------\
    {adjacency_param_spec}

    Returns
    -------
    P : ($*$, $I$, $O$) tensor
        Block comprising Girvan-Newman null matrices corresponding to each
        input adjacency matrix.
    """
    k_i = A.sum(-1, keepdims=True)
    k_o = A.sum(-2, keepdims=True)
    two_m = k_i.sum(-2, keepdims=True)
    return k_i @ k_o / two_m


@document_modularity
def modularity_matrix(
    A: Tensor,
    gamma: float = 1,
    null: Callable = girvan_newman_null,
    normalise_modularity: bool = True,
    sign: Optional[Literal['+', '-']] = '+',
    **params,
) -> Tensor:
    """
    Modularity matrices for a tensor block.
    \
    {modularity_matrix_long_desc}
    \
    {adjacency_dim_spec}

    Parameters
    ----------\
    {adjacency_param_spec}\
    {modularity_matrix_param_spec}

    Returns
    -------
    B : ($*$, $I$, $O$) tensor
        Block comprising modularity matrices corresponding to each input
        adjacency matrix.

    See also
    --------
    relaxed_modularity: Compute the modularity given a community structure.
    """
    if sign == '+':
        A = relu(A)
    elif sign == '-':
        A = -relu(-A)
    mod = A - gamma * null(A, **params)
    if normalise_modularity:
        two_m = A.sum((-2, -1), keepdims=True)
        return mod / two_m
    return mod


@document_modularity
def coaffiliation(
    C: Tensor,
    C_o: Optional[Tensor] = None,
    L: Optional[Tensor] = None,
    exclude_diag: bool = True,
    normalise_coaffiliation: bool = False,
) -> Tensor:
    """
    Coaffiliation of vertices under a community structure.
    \
    {coaffiliation_long_desc}
    \
    {coaffiliation_dim_spec}

    Parameters
    ----------\
    {coaffiliation_param_spec}

    Returns
    -------
    C : ($*$, $I$, $O$) tensor
        Coaffiliation matrix for each input community structure.
    """
    C_i = C
    if C_o is None:
        C_o = C_i
    if normalise_coaffiliation:
        norm_fac_i = jnp.maximum(1, C_i.max((-1, -2), keepdims=True))
        norm_fac_o = jnp.maximum(1, C_o.max((-1, -2), keepdims=True))
        C_i = C_i / norm_fac_i
        C_o = C_o / norm_fac_o
    if L is None:
        C = C_i @ C_o.swapaxes(-1, -2)
    else:
        C = C_i @ L @ C_o.swapaxes(-1, -2)
    if exclude_diag:
        C = delete_diagonal(C)
    return C


@document_modularity
def relaxed_modularity(
    A: Tensor,
    C: Tensor,
    C_o: Optional[Tensor] = None,
    L: Optional[Tensor] = None,
    exclude_diag: bool = True,
    gamma: float = 1,
    null: Callable = girvan_newman_null,
    normalise_modularity: bool = True,
    normalise_coaffiliation: bool = True,
    directed: bool = False,
    sign: Optional[Literal['+', '-']] = '+',
    **params,
) -> Tensor:
    """
    A relaxation of the modularity of a network given a community partition.
    \
    {relaxed_modularity_long_desc}
    \
    {coaffiliation_dim_spec}

    Parameters
    ----------
    {adjacency_param_spec}\
    {coaffiliation_param_spec}
    directed : bool (default False)
        Indicates that the input adjacency matrices should be considered as a
        directed graph.\
    {modularity_matrix_param_spec}

    Returns
    -------
    Q : Tensor
        Modularity of each input adjacency matrix.
    """
    B = modularity_matrix(
        A,
        gamma=gamma,
        null=null,
        normalise_modularity=normalise_modularity,
        sign=sign,
        **params,
    )
    C = coaffiliation(
        C,
        C_o=C_o,
        L=L,
        exclude_diag=exclude_diag,
        normalise_coaffiliation=normalise_coaffiliation,
    )
    Q = (B * C).sum((-2, -1))
    if not directed:
        return Q / 2
    return Q
