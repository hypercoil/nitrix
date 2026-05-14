# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterised-sphere (regular-grid) topology helpers.

The 2D equirectangular parameterisation of a sphere lives on a
regular grid -- the "height" axis maps to latitude, the "width"
axis maps to longitude.  Unlike a Euclidean grid, the boundary of
this grid has non-trivial topology:

- The **longitudinal** axis (``width``) is **periodic**: longitude
  ``2π`` is the same line as ``0``.
- The **latitudinal** axis (``height``) is **pole-bounded**: the
  rows at ``h = 0`` and ``h = H - 1`` are the (compressed) pole
  rows.  "Going over the pole" lands on the *opposite* longitude
  half of the grid -- pixel ``(0, j)`` has neighbour
  ``(1, j + W / 2)`` across the north pole.

This module is the parameterised-sphere counterpart to
``nitrix.geometry.sphere``, which hosts *mesh-based* spherical
primitives (vertex coordinates, k-ring adjacency, geodesic
distance).  The two are deliberately split: mesh code reaches for
``sphere``, grid code reaches for ``sphere_grid``.  Composing
either with the surface-mesh ↔ parameterised-grid resampling
operation (``surfa.SphericalMapBarycentric`` upstream, not in
``nitrix``) is out of scope here.

The primitives in this module are pure pad / slice / flip / roll
ops -- fully ``jax.jit`` / ``jax.grad`` friendly.

Suggested composition for sphere-grid morphology / smoothing /
convolution with VALID-padding kernels::

    padded = sphere_grid_pad_2d(mask, pad=k_radius)
    dilated = dilate(padded, structuring_element=se, padding='VALID')
    result = sphere_grid_unpad_2d(dilated, pad=k_radius)

This is the canonical pattern; we deliberately do NOT thread
``padding='spherical'`` through the morphology / smoothing kernels
because the pole-flip topology is not expressible as a single
boundary mode in ``lax.conv_general_dilated``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Num


__all__ = [
    'sphere_grid_pad_2d',
    'sphere_grid_unpad_2d',
]


PadSpec = Union[int, Tuple[int, int]]


def _resolve_pad(pad: PadSpec) -> Tuple[int, int]:
    if isinstance(pad, int):
        return (int(pad), int(pad))
    if not (isinstance(pad, tuple) and len(pad) == 2):
        raise ValueError(
            f'pad must be int or (h_pad, w_pad) tuple; got {pad!r}.'
        )
    return (int(pad[0]), int(pad[1]))


def _move_axes_to_front(
    x: Array, axes: Tuple[int, ...],
) -> Tuple[Array, Tuple[int, ...]]:
    '''Move ``axes`` to the front of ``x`` in the given order.

    Returns ``(reshaped, original_axes_normalised)`` where the
    returned array has those axes as its leading dims.
    '''
    n = x.ndim
    axes_norm = tuple(ax % n for ax in axes)
    if len(set(axes_norm)) != len(axes_norm):
        raise ValueError(
            f'axes must be distinct (after normalisation); got '
            f'{axes_norm} from {axes}.'
        )
    perm_front = list(axes_norm)
    perm_rest = [d for d in range(n) if d not in axes_norm]
    perm = perm_front + perm_rest
    return jnp.transpose(x, perm), axes_norm


def _move_axes_back(
    x: Array, n_front: int, axes_norm: Tuple[int, ...],
) -> Array:
    '''Inverse of ``_move_axes_to_front``.

    ``x`` has its first ``n_front`` dims corresponding to
    ``axes_norm``; rebuild the original ordering.
    '''
    n = x.ndim
    perm_inverse = [0] * n
    perm_front = list(axes_norm)
    perm_rest = [d for d in range(n) if d not in axes_norm]
    perm = perm_front + perm_rest
    for new_idx, old_idx in enumerate(perm):
        perm_inverse[old_idx] = new_idx
    return jnp.transpose(x, perm_inverse)


def sphere_grid_pad_2d(
    image: Num[Array, '... H W ...'],
    pad: PadSpec,
    *,
    height_axis: int = -2,
    width_axis: int = -1,
    pole_negate_channel: Optional[int] = None,
    pole_negate_axis: int = -1,
) -> Num[Array, '... H_padded W_padded ...']:
    '''Pad a 2D parameterised-sphere image with the sphere topology.

    The padding scheme:

    - **Longitudinal** axis (``width_axis``) is treated as a
      circular wrap: the first ``w_pad`` columns are taken from
      the rightmost ``w_pad`` of the input, and vice versa.
    - **Latitudinal** axis (``height_axis``) is pole-bounded:
      the top ``h_pad`` rows are built from the rows just inside
      the north pole (rows ``1 : h_pad + 1``), *flipped vertically*
      and *rolled horizontally by* ``W // 2``.  The bottom
      ``h_pad`` rows are symmetric: rows ``H - h_pad - 1 : H - 1``,
      flipped and rolled.  The pole rows themselves (``row 0`` and
      ``row H - 1``) are NOT used for reflection -- a pole row is
      a (compressed) point on the sphere, and reflecting it would
      duplicate it.

    For 2D flow / vector fields, the longitudinal component of the
    flow reverses sign across each pole (because longitude reverses
    on pole crossing).  Pass ``pole_negate_channel`` to identify
    which channel-axis entry to negate in the pole-padded rows.
    The latitudinal flow component does NOT reverse and should be
    left alone.

    Parameters
    ----------
    image
        Channel-last image, ``(..., H, W, ...)``.  ``H`` and ``W``
        are at ``height_axis`` and ``width_axis``; everything else
        is preserved.
    pad
        Pad amount.  ``int`` -> symmetric ``h_pad = w_pad = pad``;
        ``(h_pad, w_pad)`` -> per-axis.  After padding the height
        becomes ``H + 2 * h_pad`` and the width
        ``W + 2 * w_pad``.
    height_axis
        Axis index for the latitudinal axis.  Default ``-2``
        assumes a trailing channel axis (or scalar 2D image with
        no trailing channel; either works -- the axis selection is
        explicit).
    width_axis
        Axis index for the longitudinal axis.  Default ``-1``
        (last axis).  Note: if you pass channel-last images
        ``(*, H, W, C)``, set ``height_axis=-3``, ``width_axis=-2``.
    pole_negate_channel
        If not ``None``, index along ``pole_negate_axis`` whose
        entries should be sign-flipped in the pole-padded rows.
        Used for the longitudinal-flow component of 2D flow
        fields.
    pole_negate_axis
        Axis along which ``pole_negate_channel`` indexes.  Default
        ``-1`` (last axis).  Only consulted when
        ``pole_negate_channel`` is not ``None``.

    Returns
    -------
    Padded image with ``H + 2 * h_pad`` along ``height_axis`` and
    ``W + 2 * w_pad`` along ``width_axis``; all other axes
    unchanged.

    Notes
    -----
    This is the right padding scheme for 2D conv / morphology /
    smoothing on equirectangular-parameterised spherical images
    (FreeSurfer surface registration, surface-based saliency, etc.)
    The pole flip + W/2 roll is the natural "go over the pole"
    topology of an equirectangular grid.

    Raises
    ------
    ValueError
        If ``height_axis == width_axis``, if ``W`` is not even
        (the pole roll uses ``W // 2``; an odd ``W`` would
        introduce a half-pixel shift that this primitive doesn't
        try to correct).
    '''
    h_pad, w_pad = _resolve_pad(pad)
    if height_axis % image.ndim == width_axis % image.ndim:
        raise ValueError(
            f'height_axis and width_axis must be distinct after '
            f'normalisation; got {height_axis} and {width_axis}.'
        )
    if h_pad == 0 and w_pad == 0:
        return image

    H = image.shape[height_axis]
    W = image.shape[width_axis]
    if W % 2 != 0:
        raise ValueError(
            f'sphere_grid_pad_2d: width must be even (the pole roll '
            f'is by W // 2); got W={W}.  Pad to an even width '
            'manually if your parameterisation has an odd longitudinal '
            'grid.'
        )
    if h_pad > H - 2:
        raise ValueError(
            f'pad height {h_pad} too large for image height {H}; '
            f'the pole-skip scheme requires h_pad <= H - 2.'
        )

    # Move height and width axes to the front so we can index them
    # uniformly regardless of caller layout.
    moved, axes_norm = _move_axes_to_front(
        image, (height_axis, width_axis),
    )
    # Now ``moved`` has shape ``(H, W, *rest)``.

    # 1) Longitudinal wrap on the W axis.
    if w_pad > 0:
        left_strip = moved[:, -w_pad:, ...]
        right_strip = moved[:, :w_pad, ...]
        moved_w = jnp.concatenate(
            [left_strip, moved, right_strip], axis=1,
        )
    else:
        moved_w = moved
    # moved_w shape: ``(H, W + 2 * w_pad, *rest)``

    # 2) Pole pad on the H axis.
    # Skip the pole row itself: top pad uses rows 1..h_pad+1, flipped
    # vertically and rolled longitudinally by (W_padded // 2).
    if h_pad > 0:
        W_padded = moved_w.shape[1]
        roll_amount = W_padded // 2
        top_block = moved_w[1: h_pad + 1, ...]
        top_block = top_block[::-1, ...]  # flip vertically
        top_block = jnp.roll(top_block, roll_amount, axis=1)
        bottom_block = moved_w[-(h_pad + 1):-1, ...]
        bottom_block = bottom_block[::-1, ...]
        bottom_block = jnp.roll(bottom_block, roll_amount, axis=1)

        # Apply sign-flip on the negation channel (if requested) within
        # the pole-pad blocks.
        if pole_negate_channel is not None:
            # ``pole_negate_axis`` is given relative to the *input*
            # ndim; we moved (height, width) to front, so the original
            # axis index needs to be remapped.  Simpler: only allow
            # axes other than height/width.  We then need the new
            # position in ``moved_w``.
            neg_axis_norm = pole_negate_axis % image.ndim
            if neg_axis_norm in axes_norm:
                raise ValueError(
                    'pole_negate_axis cannot be the same as '
                    'height_axis or width_axis.'
                )
            # After moving (height, width) to front, the remaining
            # axes are in their original relative order.  Find the
            # new index of ``neg_axis_norm``:
            remaining = [d for d in range(image.ndim) if d not in axes_norm]
            new_neg_axis = 2 + remaining.index(neg_axis_norm)
            # Negate at the specified channel index.
            top_block = _negate_at_index(
                top_block, new_neg_axis, pole_negate_channel,
            )
            bottom_block = _negate_at_index(
                bottom_block, new_neg_axis, pole_negate_channel,
            )

        moved_hw = jnp.concatenate(
            [top_block, moved_w, bottom_block], axis=0,
        )
    else:
        moved_hw = moved_w

    return _move_axes_back(moved_hw, 2, axes_norm)


def _negate_at_index(x: Array, axis: int, index: int) -> Array:
    '''Negate ``x`` at ``index`` along ``axis``, leave the rest unchanged.

    Equivalent to ``x.at[..., index, ...].set(-x[..., index, ...])``
    along the named axis; written generically.
    '''
    n = x.ndim
    axis_norm = axis % n
    sl = [slice(None)] * n
    sl[axis_norm] = slice(index, index + 1)
    sl = tuple(sl)
    return x.at[sl].set(-x[sl])


def sphere_grid_unpad_2d(
    image: Num[Array, '... H_padded W_padded ...'],
    pad: PadSpec,
    *,
    height_axis: int = -2,
    width_axis: int = -1,
) -> Num[Array, '... H W ...']:
    '''Strip the padding added by ``sphere_grid_pad_2d``.

    The inverse operation is a plain slice; provided as a companion
    so the composition pattern reads cleanly::

        padded = sphere_grid_pad_2d(image, pad=k)
        out_padded = some_valid_padded_kernel(padded)
        out = sphere_grid_unpad_2d(out_padded, pad=k)

    Parameters
    ----------
    image
        Padded image.
    pad, height_axis, width_axis
        Match the arguments passed to ``sphere_grid_pad_2d``.

    Returns
    -------
    The un-padded image with the original ``H`` and ``W``.
    '''
    h_pad, w_pad = _resolve_pad(pad)
    if h_pad == 0 and w_pad == 0:
        return image
    h_norm = height_axis % image.ndim
    w_norm = width_axis % image.ndim
    sl = [slice(None)] * image.ndim
    sl[h_norm] = slice(h_pad, image.shape[h_norm] - h_pad)
    sl[w_norm] = slice(w_pad, image.shape[w_norm] - w_pad)
    return image[tuple(sl)]
