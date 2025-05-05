# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Documentation utilities.
"""

from collections import UserDict
from typing import Callable


def tensor_dimensions(dims: str):
    return (
        f"""
    Tensor Dimensions
    -----------------\
    {dims}

    """
        + ': {.striped .hover}'
    )


def form_docstring(formatter: callable):
    def decorator(func: Callable[[], DocTemplateFormat]):
        fmt = formatter()
        func.__doc__ = (func.__doc__ or '').format_map(fmt)
        return func

    return decorator


class DocTemplateFormat(UserDict[str, str]):
    """
    Enable multiple documentation decorators to be applied to a single
    function, with each pass leaving intact any cells that are not specified
    in the current decorator.
    """

    def __missing__(self, key):
        return f'{{{key}}}'
