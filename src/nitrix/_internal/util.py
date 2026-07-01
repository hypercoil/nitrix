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
    cast,
)

import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from jaxtyping import Array, Bool, Complex, Float, Shaped

# A pytree of arbitrary structure; jax operations preserve it but mypy cannot
# express the recursive shape, so it is the one place we accept ``Any``.
PyTree = Any


def _conform_bform_weight(
    weight: Shaped[Array, '...'],
) -> Shaped[Array, '...']:
    if weight.ndim == 1:
        return weight
    elif weight.shape[-2] != 1:
        return weight[..., None, :]
    return weight


def _dim_or_none(x: int, align: bool, i: int, ndmax: int) -> Optional[int]:
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


def atleast_4d(
    *pparams: Shaped[Array, '...'],
) -> Union[Shaped[Array, '...'], Sequence[Shaped[Array, '...']]]:
    """
    Promote each input tensor to at least four dimensions.

    Any tensor with fewer than four axes has singleton axes prepended until
    it is four-dimensional; tensors that are already at least rank four are
    returned unchanged.  (Note this prepends leading axes, unlike
    :func:`numpy.atleast_3d`, which appends/centres a trailing axis.)

    Parameters
    ----------
    *pparams : tensor
        One or more tensors of arbitrary shape.

    Returns
    -------
    tensor or tuple of tensor
        The promoted tensor if a single argument was passed, otherwise a tuple
        containing the promoted form of each input in argument order.
    """
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
    Return the axes not named in a selection.

    Given the rank of a tensor and a selection of one or more axes, this
    returns the standard (nonnegative) axis numbers of every remaining axis,
    in ascending order.

    Parameters
    ----------
    ndim : int
        Total number of axes in the tensor.
    axis : int or sequence of int
        Axis or axes to exclude from the complement. These are indexed
        positively from the start of the tensor.

    Returns
    -------
    tuple of int
        The axis numbers not present in ``axis``, in ascending order.
    """
    if isinstance(axis, int):
        axis = (axis,)
    mask = [True for _ in range(ndim)]
    for a in axis:
        mask[a] = False
    return tuple(i for i, a in enumerate(mask) if a)


# TODO:
# We're repeating a lot of code below so we can guarantee the return type.
# We should figure out if there's a better way to do this with mypy.
def standard_axis_number_strict(axis: int, ndim: int) -> int:
    """
    Convert an axis number to a standard (nonnegative) axis number.

    A negative axis is rewritten as its equivalent counted from the start of
    the tensor. An axis that falls outside the valid range for a tensor of
    rank ``ndim`` raises an error.

    Parameters
    ----------
    axis : int
        Axis number to convert. May be negative to index from the end.
    ndim : int
        Total number of axes in the tensor.

    Returns
    -------
    int
        The equivalent nonnegative axis number in the range ``[0, ndim)``.

    Raises
    ------
    ValueError
        If ``axis`` does not name a valid axis of a rank-``ndim`` tensor.
    """
    if axis < 0 and axis >= -ndim:
        axis += ndim
    elif axis < -ndim or axis >= ndim:
        raise ValueError(f'Invalid axis: {axis}')
    return axis


def standard_axis_number(axis: int, ndim: int) -> Optional[int]:
    """
    Convert an axis number to a standard (nonnegative) axis number.

    A negative axis is rewritten as its equivalent counted from the start of
    the tensor. This is the non-raising counterpart of
    :func:`standard_axis_number_strict`.

    Parameters
    ----------
    axis : int
        Axis number to convert. May be negative to index from the end.
    ndim : int
        Total number of axes in the tensor.

    Returns
    -------
    int or None
        The equivalent nonnegative axis number in the range ``[0, ndim)``, or
        ``None`` if ``axis`` does not name a valid axis of a rank-``ndim``
        tensor.
    """
    if axis < 0 and axis >= -ndim:
        axis += ndim
    elif axis < -ndim or axis >= ndim:
        return None
    return axis


def negative_axis_number_strict(axis: int, ndim: int) -> int:
    """
    Convert an axis number to a negative axis number.

    A nonnegative axis is rewritten as its equivalent counted from the end of
    the tensor. An axis that falls outside the valid range for a tensor of
    rank ``ndim`` raises an error.

    Parameters
    ----------
    axis : int
        Axis number to convert. May already be negative.
    ndim : int
        Total number of axes in the tensor.

    Returns
    -------
    int
        The equivalent negative axis number in the range ``[-ndim, 0)``.

    Raises
    ------
    ValueError
        If ``axis`` does not name a valid axis of a rank-``ndim`` tensor.
    """
    if axis >= 0:
        axis -= ndim
    if axis >= 0 or axis < -ndim:
        raise ValueError(f'Invalid axis: {axis}')
    return axis


def negative_axis_number(axis: int, ndim: int) -> Optional[int]:
    """
    Convert an axis number to a negative axis number.

    A nonnegative axis is rewritten as its equivalent counted from the end of
    the tensor. This is the non-raising counterpart of
    :func:`negative_axis_number_strict`.

    Parameters
    ----------
    axis : int
        Axis number to convert. May already be negative.
    ndim : int
        Total number of axes in the tensor.

    Returns
    -------
    int or None
        The equivalent negative axis number in the range ``[-ndim, 0)``, or
        ``None`` if ``axis`` does not name a valid axis of a rank-``ndim``
        tensor.
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
    Build a permutation that moves selected axes to the front.

    The returned tuple is a permutation of ``range(ndim)`` suitable for use
    as the argument to a transpose: the selected axes are placed first, in the
    order given, followed by the remaining axes in ascending order. This
    operation might not work as expected if the selected axes are not sorted.

    Parameters
    ----------
    ndim : int
        Total number of axes in the tensor.
    axis : int or tuple of int
        Axis or axes to promote to the outermost positions. These may be
        negative and are standardised internally.

    Returns
    -------
    tuple of int
        A length-``ndim`` permutation placing the promoted axes first.
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = [standard_axis_number_strict(ax, ndim) for ax in axis]
    return (*axis, *axis_complement(ndim, axis))


def _demote_axis(
    ndim: int,
    axis: Sequence[int],
) -> Iterator[int]:
    """
    Yield the inverse permutation used for axis demotion.

    This is the helper behind :func:`demote_axis`. It walks the axes of a
    rank-``ndim`` tensor in order and, for each, yields the destination
    position: axes named in ``axis`` are assigned the leading positions (in
    the order they appear), while the remaining axes fill the trailing
    positions.

    Parameters
    ----------
    ndim : int
        Total number of axes in the tensor.
    axis : sequence of int
        Standardised axis numbers to demote from the leading positions.

    Yields
    ------
    int
        The destination position for each axis, in axis order.
    """
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
    """
    Build a permutation that moves leading axes back to selected positions.

    The returned tuple is a permutation of ``range(ndim)`` suitable for use
    as the argument to a transpose. It is the inverse of the permutation
    produced by :func:`promote_axis`: axes currently at the front are routed
    to the positions named in ``axis``, and the remaining axes fill the gaps
    in ascending order.

    Parameters
    ----------
    ndim : int
        Total number of axes in the tensor.
    axis : int or tuple of int
        Target position or positions to which the leading axes are demoted.
        These may be negative and are standardised internally.

    Returns
    -------
    tuple of int
        A length-``ndim`` permutation demoting the leading axes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = [standard_axis_number_strict(ax, ndim) for ax in axis]
    return tuple(_demote_axis(ndim=ndim, axis=axis))


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_axis(
    tensor: Shaped[Array, '...'], axis: int, n_folds: int
) -> Shaped[Array, '...']:
    """
    Split an axis into two by folding it into a given number of folds.

    The chosen axis of length ``L`` is reshaped into two consecutive axes of
    sizes ``L // n_folds`` and ``n_folds``. The axis length must be divisible
    by ``n_folds``.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    axis : int
        Axis to fold. May be negative and is standardised internally.
    n_folds : int
        Number of folds; becomes the size of the newly inserted trailing
        axis.

    Returns
    -------
    tensor
        The reshaped tensor, whose rank is one greater than that of the
        input.
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
def unfold_axes(
    tensor: Shaped[Array, '...'], axes: Union[int, Tuple[int, ...]]
) -> Shaped[Array, '...']:
    """
    Merge consecutive axes into a single new axis.

    The named axes are collapsed into one axis whose length is the product of
    their sizes; the surrounding axes are preserved. This function will fail
    if the specified axes are not consecutive. If a single axis is given, the
    tensor is returned unchanged.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    axes : int or tuple of int
        Consecutive axes to merge. These may be negative and are standardised
        internally. A single integer is a no-op.

    Returns
    -------
    tensor
        The reshaped tensor, whose rank is reduced by ``len(axes) - 1``.
    """

    def _prod(x: int, y: int) -> int:
        return x * y

    if isinstance(axes, int):
        return tensor
    shape = tensor.shape
    axes_std = [standard_axis_number_strict(ax, tensor.ndim) for ax in axes]
    current = [shape[ax] for ax in axes_std]
    prod = reduce(_prod, current)
    # fmt: off
    new_shape = (
        tensor.shape[:axes_std[0]] +
        (prod,) +
        tensor.shape[axes_std[-1] + 1:]
    )
    # fmt: on
    return tensor.reshape(new_shape)


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_and_promote(
    tensor: Shaped[Array, '...'], axis: int, n_folds: int
) -> Shaped[Array, '...']:
    """
    Fold an axis and move the new fold axis to the front.

    The chosen axis is first split via :func:`fold_axis` into a pair of axes,
    and the newly created fold axis (of size ``n_folds``) is then promoted to
    the outermost position via a transpose.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    axis : int
        Axis to fold. May be negative and is standardised internally.
    n_folds : int
        Number of folds; becomes the size of the promoted leading axis.

    Returns
    -------
    tensor
        The folded and transposed tensor, whose rank is one greater than that
        of the input.
    """
    axis = standard_axis_number_strict(axis, tensor.ndim)
    folded = fold_axis(tensor, axis, n_folds)
    return jnp.transpose(folded, promote_axis(folded.ndim, axis + 1))


@partial(jax.jit, static_argnames=('target_address', 'axes'))
def demote_and_unfold(
    tensor: Shaped[Array, '...'],
    target_address: int,
    axes: Union[int, Tuple[int, ...]] | None = None,
) -> Shaped[Array, '...']:
    """
    Demote the leading axis to a target position and merge it there.

    This is the approximate inverse of :func:`fold_and_promote`. The leading
    axis is first demoted to ``target_address`` via a transpose, then merged
    with its neighbours via :func:`unfold_axes`.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    target_address : int
        Position to which the leading axis is demoted before merging.
    axes : int or tuple of int or None
        Consecutive axes to merge after demotion. When ``None``, the pair
        ``(target_address - 1, target_address)`` is used.

    Returns
    -------
    tensor
        The transposed and merged tensor.
    """
    if axes is None:
        axes = (target_address - 1, target_address)
    demoted = jnp.transpose(tensor, demote_axis(tensor.ndim, target_address))
    # ``unfold_axes`` is jax.jit-wrapped, so its stub erases the return type
    # to ``Any``; restore it (no runtime cost).
    return cast(Shaped[Array, '...'], unfold_axes(demoted, axes))


def broadcast_ignoring(
    x: Shaped[Array, '...'],
    y: Shaped[Array, '...'],
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[Shaped[Array, '...'], Shaped[Array, '...']]:
    """
    Broadcast two tensors together while leaving specified axes untouched.

    The two tensors are broadcast against one another as usual, except that
    the sizes along the ignored axes are excluded from the broadcasting rules
    and each tensor retains its own size there. This can be useful, for
    instance, when concatenating tensors along the ignored axis.

    Parameters
    ----------
    x : tensor
        First tensor.
    y : tensor
        Second tensor.
    axis : int or tuple of int
        Axis or axes to exclude from broadcasting. These may be negative and
        are standardised internally.

    Returns
    -------
    x : tensor
        First tensor broadcast to the common shape, retaining its own sizes
        along the ignored axes.
    y : tensor
        Second tensor broadcast to the common shape, retaining its own sizes
        along the ignored axes.
    """

    def _form_reduced_shape(
        axes: Sequence[int],
        shape: Tuple[int, ...],
        ndim: int,
    ) -> Tuple[Tuple[int, ...], Tuple[Optional[int], ...]]:
        axes = tuple(standard_axis_number(a, ndim) for a in axes)
        shape_reduced = tuple(
            1 if i in axes else shape[i] for i in range(ndim)
        )
        return shape_reduced, axes

    def _form_final_shape(
        axes_out: Tuple[Optional[int], ...],
        axes_in: Tuple[Optional[int], ...],
        shape_in: Tuple[int, ...],
        common_shape: Tuple[int, ...],
    ) -> Iterator[int]:
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
    axis_seq = sorted(axis)
    shape_x, shape_y = x.shape, y.shape
    shape_x_reduced, axes_x = _form_reduced_shape(axis_seq, shape_x, x.ndim)
    shape_y_reduced, axes_y = _form_reduced_shape(axis_seq, shape_y, y.ndim)
    common_shape = jnp.broadcast_shapes(shape_x_reduced, shape_y_reduced)
    axes_out = tuple(
        standard_axis_number(a, len(common_shape)) for a in axis_seq
    )
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
    f: Callable[..., Any],
    f_dim: int,
    align_outer: bool = False,
    # structuring_arg: Optional[Union[Callable, int]] = None,
) -> PyTree:
    """
    Apply a function across the outer axes of one or more tensors.

    This is intended as a quality-of-life feature for the common case of
    applying a function across the outer axes of a tensor, eliminating the
    need to write nested calls to :func:`jax.vmap`. The number of outer axes
    is inferred per input from its rank and the corresponding entry of
    ``f_dim``, and the maximum across inputs determines how many vectorising
    maps are stacked.

    Parameters
    ----------
    x : PyTree
        The positional arguments to ``f``, arranged as a PyTree of tensors;
        the leaves are unpacked and passed to ``f`` in order.
    f : callable
        Function to apply over the inner (core) axes of each input.
    f_dim : int
        Number of inner axes that ``f`` itself consumes. A single integer is
        broadcast across every leaf of ``x``; all remaining axes are treated
        as outer axes and mapped over.
    align_outer : bool, optional (default: ``False``)
        Whether the outer axes of inputs of differing rank are aligned from
        the outermost axis (``True``) or from the innermost outer axis
        (``False``). Broadcast across every leaf of ``x``.

    Returns
    -------
    PyTree
        The result of applying ``f`` over the inner axes and mapping over the
        outer axes, with the outer axes reconstituted in the output.
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
    f: Callable[..., Any],
    f_dim: int,
    align_outer: bool = False,
    # structuring_arg: Optional[Union[Callable, int]] = None,
) -> Callable[..., Any]:
    """
    Transform a function so that it applies across outer axes.

    This is the transform-returning counterpart of
    :func:`apply_vmap_over_outer`: it returns a new callable that, when
    invoked with the tensor arguments, maps ``f`` over their outer axes. It
    partially applies ``f``, ``f_dim`` and ``align_outer`` so the resulting
    function need only be passed the data.

    Parameters
    ----------
    f : callable
        Function to apply over the inner (core) axes of each input.
    f_dim : int
        Number of inner axes that ``f`` itself consumes; all remaining axes
        are mapped over.
    align_outer : bool, optional (default: ``False``)
        Whether the outer axes of inputs of differing rank are aligned from
        the outermost axis (``True``) or from the innermost outer axis
        (``False``).

    Returns
    -------
    callable
        A function that applies ``f`` across the outer axes of its tensor
        arguments.
    """
    return partial(
        apply_vmap_over_outer,
        f=f,
        f_dim=f_dim,
        align_outer=align_outer,
        # structuring_arg=structuring_arg,
    )


def argsort(seq: Sequence[Any], reverse: bool = False) -> List[int]:
    # Sources:
    # (1) https://stackoverflow.com/questions/3382352/ ...
    #     equivalent-of-numpy-argsort-in-basic-python
    # (2) http://stackoverflow.com/questions/3071415/ ...
    #     efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def orient_and_conform(
    input: Shaped[Array, '...'],
    axis: Union[int, Sequence[int]],
    reference: Optional[Shaped[Array, '...']] = None,
    dim: Optional[int] = None,
) -> Shaped[Array, '...']:
    """
    Reorient a tensor along chosen axes and pad its rank to match a reference.

    The input's existing axes are routed to the positions named in ``axis``
    (transposing first if they are given out of order), and singleton axes are
    then inserted so that the result has the same total number of axes as a
    reference tensor or as an explicit target rank.

    **Note:** If both ``reference`` and ``dim`` are provided, then ``dim``
    takes precedence.

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
        assert reference is not None
        dim = reference.ndim
    # can't rely on this when we compile with jit
    # TODO: Would there be any benefit to checkify this?
    assert len(axis) == input.ndim, (
        'Output orientation axis required for each input dimension'
    )
    standard_axes = [standard_axis_number_strict(ax, dim) for ax in axis]
    axis_order = argsort(standard_axes)
    # I think XLA will be smart enough to know when this is a no-op
    input = input.transpose(axis_order)
    standard_axes_set = set(standard_axes)
    shape = [1] * dim
    for size, ax in zip(input.shape, standard_axes_set):
        shape[ax] = size
    return input.reshape(*shape)


def promote_to_rank(
    tensor: Shaped[Array, '...'], rank: int
) -> Shaped[Array, '...']:
    """
    Promote a tensor to a given rank by prepending singleton axes.

    Leading singleton axes are inserted until the tensor has ``rank`` axes.
    If the tensor already has at least ``rank`` axes, it is returned
    unchanged.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    rank : int
        Target number of axes.

    Returns
    -------
    tensor
        The input with leading singleton axes prepended as needed.
    """
    ndim = tensor.ndim
    if ndim >= rank:
        return tensor
    return tensor.reshape((1,) * (rank - ndim) + tensor.shape)


def extend_to_size(
    tensor: Shaped[Array, '...'],
    shape: Sequence[int],
    fill: float = float('nan'),
) -> Shaped[Array, '...']:
    """
    Pad a tensor up to a target shape, filling new entries.

    The tensor is first promoted to the rank of ``shape`` and then embedded in
    the upper-left corner of an array of the requested shape, so that its
    existing entries retain their positions. Any new entries created by this
    extension are populated with ``fill``.

    Parameters
    ----------
    tensor : tensor
        Input tensor. Must be no larger than ``shape`` along any axis.
    shape : sequence of int
        Target shape.
    fill : float, optional (default: ``float('nan')``)
        Value with which to populate newly created entries.

    Returns
    -------
    tensor
        A tensor of shape ``shape`` with the input embedded at the origin.
    """
    tensor = promote_to_rank(tensor, len(shape))
    out = jnp.full(shape, fill, dtype=tensor.dtype)
    return out.at[tuple(slice(s) for s in tensor.shape)].set(tensor)


def extend_to_max_size(
    tensors: Sequence[Shaped[Array, '...']],
    fill: float = float('nan'),
) -> Tuple[Shaped[Array, '...'], ...]:
    """
    Pad every tensor in a sequence up to a common maximal shape.

    All tensors are promoted to the maximum rank present in the sequence, and
    each is then extended so that its size along every axis equals the
    greatest size found among the inputs along that axis. Any new entries
    created by this extension are populated with ``fill``.

    Parameters
    ----------
    tensors : sequence of tensor
        Tensors to extend to a common shape.
    fill : float, optional (default: ``float('nan')``)
        Value with which to populate newly created entries.

    Returns
    -------
    tuple of tensor
        The input tensors, each extended to the common maximal shape, in the
        original order.
    """
    ndim_max = max(t.ndim for t in tensors)
    tensors = tuple(promote_to_rank(t, ndim_max) for t in tensors)
    shape_max = tuple(
        max(t.shape[i] for t in tensors) for i in range(ndim_max)
    )
    return tuple(extend_to_size(t, shape_max, fill=fill) for t in tensors)


def complex_decompose(
    complex: Complex[Array, '...'],
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
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


def complex_recompose(
    ampl: Float[Array, '...'], phase: Float[Array, '...']
) -> Complex[Array, '...']:
    """
    Reconstitute a complex-valued tensor from its amplitude and phase.

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


def amplitude_apply(
    func: Callable[[Float[Array, '...']], Float[Array, '...']],
) -> Callable[[Complex[Array, '...']], Complex[Array, '...']]:
    """
    Lift a real-valued function to act on complex amplitudes only.

    The returned wrapper decomposes a complex tensor into amplitude and phase
    via :func:`complex_decompose`, applies ``func`` to the amplitude, and
    recombines the transformed amplitude with the unchanged phase via
    :func:`complex_recompose`.

    Parameters
    ----------
    func : callable
        A function mapping a real-valued amplitude tensor to a real-valued
        amplitude tensor of the same shape.

    Returns
    -------
    callable
        A function that applies ``func`` to the amplitude of a complex tensor
        while leaving its phase unchanged.
    """

    def wrapper(complex: Complex[Array, '...']) -> Complex[Array, '...']:
        ampl, phase = complex_decompose(complex)
        return complex_recompose(func(ampl), phase)

    return wrapper


def conform_mask(
    tensor: Shaped[Array, '...'],
    mask: Bool[Array, '...'],
    axis: Sequence[int],
) -> Shaped[Array, '...']:
    """
    Broadcast a mask so that it can be applied elementwise to a tensor.

    The mask is oriented so that its axes align with the named axes of the
    tensor and is then tiled along all remaining axes, yielding an array of
    the same shape as the tensor that can be applied elementwise as a mask or
    weight.

    Parameters
    ----------
    tensor : tensor
        Tensor the mask is to be conformed to.
    mask : boolean tensor
        Mask whose axes correspond, in order, to the axes named in ``axis``.
    axis : sequence of int
        Axes of ``tensor`` along which the mask varies. These may be negative
        and are standardised internally.

    Returns
    -------
    tensor
        The mask tiled to the full shape of ``tensor``.

    See also
    --------
    :func:`apply_mask`
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = sorted(standard_axis_number_strict(ax, tensor.ndim) for ax in axis)
    mask = orient_and_conform(mask, axis, reference=tensor)
    axis_set = set(axis)
    tile = [1 if i in axis_set else e for i, e in enumerate(tensor.shape)]
    # if mask.ndim != tensor.ndim:
    #     breakpoint()
    return jnp.tile(mask, tile)


def apply_mask(
    tensor: Shaped[Array, '...'],
    msk: Bool[Array, '...'],
    axis: int,
) -> Shaped[Array, '...']:
    """
    Select entries of a tensor along an axis using a one-dimensional mask.

    Unlike :func:`mask_tensor`, which zeroes out masked entries in place, this
    function removes the masked-out positions, so the size of the tensor along
    the chosen axis shrinks to the number of ``True`` entries in the mask.

    **Note:** This function will only work if the mask is one-dimensional. For
    multi-dimensional masks, use :func:`conform_mask`.

    **Note:** Use of this function is strongly discouraged. It is incompatible
    with :func:`jax.jit` because the output shape depends on the mask
    contents.

    Parameters
    ----------
    tensor : tensor
        Tensor to be masked.
    msk : boolean tensor
        One-dimensional mask selecting entries along ``axis``.
    axis : int
        Axis of ``tensor`` along which the mask is applied.

    Returns
    -------
    tensor
        The tensor with masked-out positions removed along ``axis``; that axis
        is resized to the number of ``True`` entries in ``msk``.

    See also
    --------
    :func:`conform_mask`
    :func:`mask_tensor`
    """
    tensor = jnp.asarray(tensor)
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx: Tuple[int, ...] = ()
    else:
        shape_sfx = tensor.shape[(axis + 1) :]
    msk = jnp.tile(msk, (*shape_pfx, 1))
    return tensor[msk].reshape(*shape_pfx, -1, *shape_sfx)


def mask_tensor(
    tensor: Shaped[Array, '...'],
    mask: Bool[Array, '...'],
    axis: Sequence[int],
    fill_value: Union[float, Shaped[Array, '...']] = 0,
) -> Shaped[Array, '...']:
    """
    Replace masked-out entries of a tensor with a fill value.

    The mask is first conformed to the tensor via :func:`conform_mask`, and
    entries where the mask is ``False`` are then replaced by ``fill_value``.
    Unlike :func:`apply_mask`, the shape of the tensor is preserved, making
    this variant compatible with :func:`jax.jit`.

    Parameters
    ----------
    tensor : tensor
        Tensor to be masked.
    mask : boolean tensor
        Mask whose axes correspond, in order, to the axes named in ``axis``.
    axis : sequence of int
        Axes of ``tensor`` along which the mask varies.
    fill_value : float or tensor, optional (default: ``0``)
        Value substituted wherever the conformed mask is ``False``. May be a
        scalar or an array broadcastable against ``tensor``.

    Returns
    -------
    tensor
        The input tensor with masked-out entries replaced by ``fill_value``,
        of the same shape as ``tensor``.

    See also
    --------
    :func:`conform_mask`
    :func:`apply_mask`
    """
    mask = conform_mask(tensor=tensor, mask=mask, axis=axis)
    return jnp.where(mask, tensor, fill_value)


def masker(
    mask: Bool[Array, '...'],
    axis: int | Sequence[int],
    output_axis: int | None = None,
) -> Callable[[Shaped[Array, '...']], Shaped[Array, '...']]:
    """
    Build a JIT-compatible function that extracts masked entries of a tensor.

    The mask is materialised once, at construction time, into the coordinates
    of its ``True`` entries; the returned function then uses these coordinates
    to index the masked entries out of any input tensor of compatible shape.
    Because the number of selected entries is fixed when the closure is built,
    the resulting function has a static output shape and is compatible with
    :func:`jax.jit`.

    **Note:** This function comes with some memory overhead. Specifically, it
    closes over an integer array of the same size as the number of ``True``
    elements in the mask. When applying a very large mask to a tensor, it is
    important to consider the trade-off between memory and the potential
    performance gains of JIT compilation.

    **Note:** Just like any JIT-compiled function, the resulting function must
    be recompiled when the shape of the input tensor changes.

    Parameters
    ----------
    mask : boolean tensor
        Mask selecting the entries to extract. Its axes correspond, in order,
        to the axes named in ``axis``. Must contain at least one ``True``
        entry.
    axis : int or sequence of int
        Axis or axes of the input tensor over which the mask is applied. All
        entries must share the same sign: either all negative (indexing from
        the end) or all nonnegative (indexing from the start); mixing signs
        raises an error.
    output_axis : int or None, optional (default: ``None``)
        If given, the axis to which the selected entries are moved in the
        output. If ``None``, the selected entries remain at the position
        determined by the masked axes.

    Returns
    -------
    callable
        A JIT-compiled function mapping a tensor to the tensor of its masked
        entries.

    Raises
    ------
    ValueError
        If the axis entries mix signs, or if the mask contains no ``True``
        entries.
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
    def apply_mask(
        tensor: Shaped[Array, '...'],
    ) -> Shaped[Array, '...']:
        _axis = tuple(
            standard_axis_number_strict(ax, tensor.ndim) for ax in axis
        )
        indexer: List[Any] = [slice(None)] * tensor.ndim
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
