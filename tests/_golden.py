# -*- coding: utf-8 -*-
"""Golden-corpus + tolerance helpers for the nitrix kernel suite.

Each fixture under ``tests/golden/<name>.npz`` bundles the inputs *and* the
reference output, so a test recomputes the reference from the stored inputs
and asserts it still matches the checked-in output (cross-release
reproducibility, SPEC_UPDATE §2.8).  Tolerances live in
``tests/tolerance.toml``.  Regenerate with ``python tools/regen_golden.py``.
"""

from __future__ import annotations

import pathlib
import tomllib
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike

_HERE = pathlib.Path(__file__).resolve().parent
GOLDEN_DIR = _HERE / 'golden'
_TOLERANCE_PATH = _HERE / 'tolerance.toml'


def load_golden(name: str) -> dict[str, np.ndarray]:
    """Load a golden fixture as a ``name -> array`` dict."""
    with np.load(GOLDEN_DIR / f'{name}.npz') as data:
        return {key: data[key] for key in data.files}


def tol(
    op: str,
    dtype: DTypeLike,
    backend: Optional[str] = None,
) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for ``op`` at ``dtype`` (optional backend)."""
    with open(_TOLERANCE_PATH, 'rb') as handle:
        table = tomllib.load(handle)
    entry = table[op][np.dtype(dtype).name]
    if backend is not None and backend in entry:
        entry = entry[backend]
    return float(entry['atol']), float(entry['rtol'])
