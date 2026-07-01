# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Documentation utilities.
"""

from __future__ import annotations

from collections import UserDict
from typing import Any, Callable


def tensor_dimensions(dims: str) -> str:
    return (
        f"""
    Tensor Dimensions
    -----------------\
    {dims}

    """
        + ': {.striped .hover}'
    )


def form_docstring(
    formatter: Callable[[], 'DocTemplateFormat'],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        fmt = formatter()
        func.__doc__ = (func.__doc__ or '').format_map(fmt)
        return func

    return decorator


class DocTemplateFormat(UserDict[str, str]):
    """
    Format mapping that leaves unspecified placeholders untouched.

    This mapping is used with :meth:`str.format_map` so that multiple
    documentation decorators may be applied to a single function in
    succession. Any placeholder that is not supplied by the current
    decorator is left intact for a later pass, because a missing key is
    rendered back into its original ``{key}`` form rather than raising a
    :class:`KeyError`.
    """

    def __missing__(self, key: str) -> str:
        return f'{{{key}}}'
