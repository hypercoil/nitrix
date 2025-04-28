# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utility functions for tensor manipulation.
"""

from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from numpy.typing import NDArray

Tensor = Union[jax.Array, NDArray[Any]]
PyTree = Any


def _conform_bform_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 1:
        return weight
    elif weight.shape[-2] != 1:
        return weight[..., None, :]
    return weight


def _dim_or_none(x, align, i, ndmax):
    if not align:
        i = i - ndmax
    else:
        i = -i
    proposal = i + x
    if proposal < 0:
        return None
    elif align:
        return -i
    return proposal


def _compose(
    f: Any,
    g: Callable[[Any], Any],
) -> Any:
    return g(f)


def _seq_pad(
    x: Tuple[Any, ...],
    n: int,
    pad: str = 'last',
    pad_value: Any = None,
) -> Tuple[Any, ...]:
    padding = [pad_value for _ in range(n - len(x))]
    if pad == 'last':
        return tuple((*x, *padding))
    elif pad == 'first':
        return tuple((*padding, *x))
    raise ValueError(f'Invalid padding: {pad}')


def atleast_4d(*pparams: Tensor) -> Tensor | Sequence[Tensor]:
    res = []
    for p in pparams:
        if p.ndim == 0:
            result = p.reshape(1, 1, 1, 1)
        elif p.ndim == 1:
            result = p[None, None, None, ...]
        elif p.ndim == 2:
            result = p[None, None, ...]
        elif p.ndim == 3:
            result = p[None, ...]
        else:
            result = p
        res.append(result)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def axis_complement(
    ndim: int,
    axis: Union[int, Sequence[int]],
) -> Tuple[int, ...]:
    """
    Return the complement of the axis or axes for a tensor of dimension ndim.
    """
    if isinstance(axis, int):
        axis = (axis,)
    ax = [True for _ in range(ndim)]
    for a in axis:
        ax[a] = False
    ax = [i for i, a in enumerate(ax) if a]
    return tuple(ax)


# TODO:
# We're repeating a lot of code below so we can guarantee the return type.
# We should figure out if there's a better way to do this with mypy.
def standard_axis_number_strict(axis: int, ndim: int) -> int:
    """
    Convert an axis number to a standard axis number.
    """
    if axis < 0 and axis >= -ndim:
        axis += ndim
    elif axis < -ndim or axis >= ndim:
        raise ValueError(f'Invalid axis: {axis}')
    return axis


def standard_axis_number(axis: int, ndim: int) -> Optional[int]:
    """
    Convert an axis number to a standard axis number.
    """
    if axis < 0 and axis >= -ndim:
        axis += ndim
    elif axis < -ndim or axis >= ndim:
        return None
    return axis


def negative_axis_number_strict(axis: int, ndim: int) -> int:
    """
    Convert a standard axis number to a negative axis number.
    """
    if axis >= 0:
        axis -= ndim
    if axis >= 0 or axis < -ndim:
        raise ValueError(f'Invalid axis: {axis}')
    return axis


def negative_axis_number(axis: int, ndim: int) -> Optional[int]:
    """
    Convert a standard axis number to a negative axis number.
    """
    if axis >= 0:
        axis -= ndim
    if axis >= 0 or axis < -ndim:
        return None
    return axis


def promote_axis(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    """
    Promote an axis or axes to the outermost dimension.
    This operation might not work as expected if the axes are not sorted.
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis: List[int] = [standard_axis_number_strict(ax, ndim) for ax in axis]
    return (*axis, *axis_complement(ndim, axis))


def _demote_axis(
    ndim: int,
    axis: Sequence[int],
) -> Iterator[int]:
    """Helper function for axis demotion."""
    compl = range(len(axis), ndim).__iter__()
    src = range(len(axis)).__iter__()
    for ax in range(ndim):
        if ax in axis:
            yield src.__next__()
        else:
            yield compl.__next__()


def demote_axis(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    if isinstance(axis, int):
        axis = (axis,)
    axis: List[int] = [standard_axis_number_strict(ax, ndim) for ax in axis]
    return tuple(_demote_axis(ndim=ndim, axis=axis))


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_axis(tensor: Tensor, axis: int, n_folds: int) -> Tensor:
    """
    Fold the specified axis into the specified number of folds.
    """
    axis = standard_axis_number_strict(axis, tensor.ndim)
    shape = tensor.shape
    current = shape[axis]
    # fmt: off
    new_shape = (
        shape[:axis] +
        (current // n_folds, n_folds) +
        shape[axis + 1:]
    )
    # fmt: on
    return tensor.reshape(new_shape)


@partial(jax.jit, static_argnames=('axes',))
def unfold_axes(tensor: Tensor, axes: Union[int, Tuple[int, ...]]) -> Tensor:
    """
    Unfold the specified consecutive axes into a single new axis.

    This function will fail if the specified axes are not consecutive.
    """

    def _prod(x, y):
        return x * y

    if isinstance(axes, int):
        return tensor
    shape = tensor.shape
    axes: List[int] = [  # type: ignore
        standard_axis_number_strict(ax, tensor.ndim) for ax in axes
    ]
    current = [shape[ax] for ax in axes]
    prod = reduce(_prod, current)
    # fmt: off
    new_shape = (
        tensor.shape[:axes[0]] +
        (prod,) +
        tensor.shape[axes[-1] + 1:]
    )
    # fmt: on
    return tensor.reshape(new_shape)


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_and_promote(tensor: Tensor, axis: int, n_folds: int) -> Tensor:
    """
    Fold the specified axis into the specified number of folds, and promote
    the new axis across the number of folds to the outermost dimension.
    """
    axis = standard_axis_number_strict(axis, tensor.ndim)
    folded = fold_axis(tensor, axis, n_folds)
    return jnp.transpose(folded, promote_axis(folded.ndim, axis + 1))


@partial(jax.jit, static_argnames=('target_address', 'axes'))
def demote_and_unfold(
    tensor: Tensor,
    target_address: int,
    axes: Union[int, Tuple[int, ...]] | None = None,
):
    if axes is None:
        axes = (target_address - 1, target_address)
    demoted = jnp.transpose(tensor, demote_axis(tensor.ndim, target_address))
    return unfold_axes(demoted, axes)


def broadcast_ignoring(
    x: Tensor,
    y: Tensor,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[Tensor, Tensor]:
    """
    Broadcast two tensors, ignoring the axis or axes specified.

    This can be useful, for instance, when concatenating tensors along
    the ignored axis.
    """

    def _form_reduced_shape(axes, shape, ndim):
        axes = tuple(standard_axis_number(a, ndim) for a in axes)
        shape_reduced = tuple(
            1 if i in axes else shape[i] for i in range(ndim)
        )
        return shape_reduced, axes

    def _form_final_shape(axes_out, axes_in, shape_in, common_shape):
        j = 0
        for i, s in enumerate(common_shape):
            if i not in axes_out:
                yield s
            else:
                ax = axes_in[j]
                if ax is None or ax > len(shape_in):
                    yield 1
                else:
                    yield shape_in[ax]
                j += 1

    if isinstance(axis, int):
        axis = (axis,)
    axis = sorted(axis)
    shape_x, shape_y = x.shape, y.shape
    shape_x_reduced, axes_x = _form_reduced_shape(axis, shape_x, x.ndim)
    shape_y_reduced, axes_y = _form_reduced_shape(axis, shape_y, y.ndim)
    common_shape = jnp.broadcast_shapes(shape_x_reduced, shape_y_reduced)
    axes_out = tuple(standard_axis_number(a, len(common_shape)) for a in axis)
    shape_y = tuple(
        _form_final_shape(
            axes_out=axes_out,
            axes_in=axes_y,
            shape_in=shape_y,
            common_shape=common_shape,
        )
    )
    shape_x = tuple(
        _form_final_shape(
            axes_out=axes_out,
            axes_in=axes_x,
            shape_in=shape_x,
            common_shape=common_shape,
        )
    )
    return jnp.broadcast_to(x, shape_x), jnp.broadcast_to(y, shape_y)


# TODO: use chex to evaluate how often this has to compile when using
#      jit + vmap_over_outer
def apply_vmap_over_outer(
    x: PyTree,
    f: Callable[[Any], Any],
    f_dim: int,
    align_outer: bool = False,
    # structuring_arg: Optional[Union[Callable, int]] = None,
) -> PyTree:
    """
    Apply a function across the outer dimensions of a tensor.

    This is intended to be a QoL feature for handling the common case of
    applying a function across the outer dimensions of a tensor. The goal is
    to eliminate the need for nested calls to ``jax.vmap``.
    """
    if isinstance(f_dim, int):
        f_dim = tree_map(lambda _: f_dim, x)
    if isinstance(align_outer, bool):
        align_outer = tree_map(lambda _: align_outer, x)
    ndim = tree_map(lambda x, f: x.ndim - f - 1, x, f_dim)
    ndmax = tree_reduce(max, ndim)
    # if structuring_arg is None:
    #     output_structure = range(0, ndmax + 1)
    # else:
    #     if isinstance(structuring_arg, int):
    #         output_structure = range(
    #             0, x[structuring_arg].ndim - f_dim[structuring_arg]
    #         )
    #         criterion = align_outer[structuring_arg]
    #     else:
    #         output_structure = range(
    #             0, structuring_arg(x).ndim - structuring_arg(f_dim)
    #         )
    #         criterion = structuring_arg(align_outer)
    #     if criterion:
    #         output_structure = _seq_pad(output_structure, ndmax + 1, 'last')
    #     else:
    #         output_structure = _seq_pad(
    #             output_structure,
    #             ndmax + 1,
    #             'first',
    #         )
    # print(ndim, tuple(range(ndmax + 1)))
    output_structure = range(0, ndmax + 1)
    # print([(
    #    tree_map(
    #         partial(_dim_or_none, i=i, ndmax=ndmax),
    #         ndim,
    #         align_outer
    #     ), i, o)
    #     for i, o in zip(range(0, ndmax + 1), output_structure)
    # ])
    return reduce(
        _compose,
        # lambda x, g: g(x),
        [
            partial(
                vmap,
                in_axes=tree_map(
                    partial(_dim_or_none, i=i, ndmax=ndmax),
                    ndim,
                    align_outer,
                ),
                out_axes=o,
            )
            for i, o in zip(range(0, ndmax + 1), output_structure)
        ],
        f,
    )(*x)


def vmap_over_outer(
    f: Callable[[Any], Any],
    f_dim: int,
    align_outer: bool = False,
    # structuring_arg: Optional[Union[Callable, int]] = None,
) -> Callable[[Any], Any]:
    """
    Transform a function to apply to the outer dimensions of a tensor.
    """
    return partial(
        apply_vmap_over_outer,
        f=f,
        f_dim=f_dim,
        align_outer=align_outer,
        # structuring_arg=structuring_arg,
    )


def argsort(seq, reverse: bool = False):
    # Sources:
    # (1) https://stackoverflow.com/questions/3382352/ ...
    #     equivalent-of-numpy-argsort-in-basic-python
    # (2) http://stackoverflow.com/questions/3071415/ ...
    #     efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def orient_and_conform(
    input: Tensor,
    axis: Union[int, Sequence[int]],
    reference: Optional[Tensor] = None,
    dim: Optional[int] = None,
) -> Tensor:
    """
    Orient an input tensor along a set of axes, and conform its overall
    dimension to equal that of a reference.

    .. warning::

        If both ``reference`` and ``dim`` are provided, then ``dim`` takes
        precedence.

    Parameters
    ----------
    input : tensor
        Input tensor.
    axis : tuple
        Output axes along which the tensor's input dimensions should be
        reoriented. This should be an n-tuple, where n is the number of axes
        in the input tensor. If these axes are not in the same order in the
        input tensor, the input is transposed before being oriented.
    reference : tensor or None
        Reference tensor. The output is unsqueezed so that its total
        dimension equals that of the reference. Either a reference or an
        explicit output dimension (``dim``) must be provided.
    dim : int or None
        Number of tensor axes in the desired output.

    Returns
    -------
    tensor
        Reoriented tensor with singleton axes appended to conform with the
        reference number of axes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if dim is None and reference is None:
        raise ValueError('Must specify either `reference` or `dim`')
    elif dim is None:
        dim = reference.ndim  # type: ignore
    # can't rely on this when we compile with jit
    # TODO: Would there be any benefit to checkify this?
    assert (
        len(axis) == input.ndim
    ), 'Output orientation axis required for each input dimension'
    standard_axes = [standard_axis_number(ax, dim) for ax in axis]
    axis_order = argsort(standard_axes)
    # I think XLA will be smart enough to know when this is a no-op
    input = input.transpose(axis_order)
    standard_axes = set(standard_axes)
    shape = [1] * dim
    for size, ax in zip(input.shape, standard_axes):
        shape[ax] = size
    return input.reshape(*shape)


def promote_to_rank(tensor: Tensor, rank: int) -> Tensor:
    """
    Promote a tensor to a rank-``rank`` tensor by prepending singleton
    dimensions. If the tensor is already rank-``rank``, this is a no-op.
    """
    ndim = tensor.ndim
    if ndim >= rank:
        return tensor
    return tensor.reshape((1,) * (rank - ndim) + tensor.shape)


def extend_to_size(
    tensor: Tensor,
    shape: Sequence[int],
    fill: float = float('nan'),
) -> Tensor:
    """
    Extend a tensor in the positive direction until its size matches the
    specification. Any new entries created via extension are populated with
    ``fill``.
    """
    tensor = promote_to_rank(tensor, len(shape))
    out = jnp.full(shape, fill, dtype=tensor.dtype)
    return out.at[tuple(slice(s) for s in tensor.shape)].set(tensor)


def extend_to_max_size(
    tensors: Sequence[Tensor],
    fill: float = float('nan'),
) -> Tuple[Tensor, ...]:
    """
    Extend all tensors in a sequence until their sizes are equal to the size
    of the largest tensor along each axis. Any new entries created via
    extension are populated with ``fill``.
    """
    ndim_max = max(t.ndim for t in tensors)
    tensors = tuple(promote_to_rank(t, ndim_max) for t in tensors)
    shape_max = tuple(
        max(t.shape[i] for t in tensors) for i in range(ndim_max)
    )
    return tuple(extend_to_size(t, shape_max, fill=fill) for t in tensors)


def complex_decompose(complex: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    Together with
    [complex_recompose](nitrix._internal.util.complex_recompose.html), this
    function provides a uniform interface for switching between complex and
    polar representations of complex numbers. This enables us to change the
    implementation of these operations in one place when we find the most
    efficient and stable method.

    Parameters
    ----------
    complex : Tensor
        Complex-valued tensor.

    Returns
    -------
    ampl : Tensor
        Amplitude of each entry in the input tensor.
    phase : Tensor
        Phase of each entry in the input tensor, in radians.

    See also
    --------
    :func:`complex_recompose`
    """
    ampl = jnp.abs(complex)
    phase = jnp.angle(complex)
    return ampl, phase


def complex_recompose(ampl: Tensor, phase: Tensor) -> Tensor:
    """
    Reconstitute a complex-valued tensor from real-valued tensors denoting its
    amplitude and its phase.

    Together with
    [complex_decompose](nitrix._internal.util.complex_decompose.html), this
    function provides a uniform interface for switching between complex and
    polar representations of complex numbers. This enables us to change the
    implementation of these operations in one place when we find the most
    efficient and stable method.

    Parameters
    ----------
    ampl : Tensor
        Real-valued array storing complex number amplitudes.
    phase : Tensor
        Real-valued array storing complex number phases in radians.

    Returns
    -------
    complex : Tensor
        Complex numbers formed from the specified amplitudes and phases.

    See also
    --------
    :func:`complex_decompose`
    """
    # TODO : consider using the complex exponential function,
    # depending on the gradient properties
    # return ampl * jnp.exp(phase * 1j)
    return ampl * (jnp.cos(phase) + 1j * jnp.sin(phase))


def amplitude_apply(func: Callable) -> Callable:
    """
    Decorator for applying a function to the amplitude component of a complex
    tensor.
    """

    def wrapper(complex: Tensor) -> Tensor:
        ampl, phase = complex_decompose(complex)
        return complex_recompose(func(ampl), phase)

    return wrapper


def conform_mask(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
) -> Tensor:
    """
    Conform a mask or weight for elementwise applying to a tensor.

    There is almost certainly a better way to do this.

    See also
    --------
    :func:`apply_mask`
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = sorted(standard_axis_number(ax, tensor.ndim) for ax in axis)
    mask = orient_and_conform(mask, axis, reference=tensor)
    axis = set(axis)
    tile = [1 if i in axis else e for i, e in enumerate(tensor.shape)]
    # if mask.ndim != tensor.ndim:
    #     breakpoint()
    return jnp.tile(mask, tile)


def apply_mask(
    tensor: Tensor,
    msk: Tensor,
    axis: int,
) -> Tensor:
    """
    Mask a tensor along an axis.

    .. warning::

        This function will only work if the mask is one-dimensional. For
        multi-dimensional masks, use :func:`conform_mask`.

    .. warning::

        Use of this function is strongly discouraged. It is incompatible with
        `jax.jit`.

    See also
    --------
    :func:`conform_mask`
    :func:`mask_tensor`
    """
    tensor = jnp.asarray(tensor)
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1) :]
    msk = jnp.tile(msk, (*shape_pfx, 1))
    return tensor[msk].reshape(*shape_pfx, -1, *shape_sfx)


def mask_tensor(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
    fill_value: Union[float, Tensor] = 0,
):
    mask = conform_mask(tensor=tensor, mask=mask, axis=axis)
    return jnp.where(mask, tensor, fill_value)


def masker(
    mask: Tensor,
    axis: int | Sequence[int],
    output_axis: int | None = None,
) -> Callable[[Tensor], Tensor]:
    """
    Create a JIT-compatible function that applies a mask to a tensor.

    .. warning::

        This function comes with some memory overhead. Specifically, it
        closes over an integer array of the same size as the number of
        ``True`` elements in the mask. When applying a very large mask to
        a tensor, it is important to consider the trade-off between memory
        and potential performance gains of JIT compilation.

    .. warning::

        Just like any JIT-compiled function, the resulting function must be
        recompiled when the shape of the input tensor changes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    # The axis sequence must be either all negative or all nonnegative
    sign = tuple(ax < 0 for ax in axis)
    if any(sign) and not all(sign):
        raise ValueError(
            'Mixed signs in the axis selector can result in undefined '
            'behaviour. Ensure that all axis entries are either negative '
            '(i.e., the tensor is being indexed from the end) or nonnegative '
            '(i.e., the tensor is being indexed from the beginning).'
        )
    axis_order = argsort(axis)
    mask = mask.transpose(axis_order)
    axis = tuple(axis[i] for i in axis_order)

    mask_loc = jnp.where(mask)
    if mask_loc[0].size == 0:
        raise ValueError('Mask is empty')
    assert len(mask_loc) == len(axis)

    @jax.jit
    def apply_mask(tensor: Tensor) -> Tensor:
        _axis = tuple(standard_axis_number(ax, tensor.ndim) for ax in axis)
        indexer = [slice(None)] * tensor.ndim
        for ax, loc in zip(_axis, mask_loc):
            indexer[ax] = loc
        result = tensor[tuple(indexer)]

        # Check if masked axes are consecutive. Numpy's mixed indexing logic
        # follows a different path depending on whether the axes are
        # consecutive or not. See:
        # https://numpy.org/doc/stable/user/basics.indexing.html# ...
        # ... combining-advanced-and-basic-indexing
        # and find the section that contains the phrase:
        # "Two cases of index combination need to be distinguished..."
        masked_axes_are_consecutive = all(
            _axis[i] + 1 == _axis[i + 1] for i in range(len(_axis) - 1)
        )

        if masked_axes_are_consecutive:
            # When advanced indices are consecutive, they appear in the same
            # position as in the original array
            masked_axis = _axis[0]
            # result = jnp.moveaxis(result, masked_axis, 0)
        else:
            # When advanced indices are separated by slices, they come first
            # in the result array
            masked_axis = 0

        if output_axis is not None:
            return jnp.moveaxis(result, masked_axis, output_axis)
        return result

    return apply_mask
