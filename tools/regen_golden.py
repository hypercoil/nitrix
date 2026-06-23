# -*- coding: utf-8 -*-
"""Regenerate the golden corpus under ``tests/golden/``.

Deterministic (fixed seeds, float32).  Run from the repo root with the
project venv::

    python tools/regen_golden.py

Each fixture bundles inputs + reference output as an ``.npz`` (see
``tests/_golden.py``).  Regeneration is from the *reference* path only --
never the fused kernel.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from nitrix.nn.attention import reference_scaled_dot_product_attention

_GOLDEN_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'golden'
)


def _save(name: str, **arrays: np.ndarray) -> None:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_GOLDEN_DIR / f'{name}.npz', **arrays)
    print(f'wrote {name}.npz')


def _randn(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _ref(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    bias: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    causal: bool = False,
) -> np.ndarray:
    out = reference_scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        causal=causal,
    )
    return np.asarray(out, dtype=np.float32)


def regen_attention() -> None:
    h, d, dv = 2, 8, 8
    b, s, t = 2, 6, 6

    # dense
    rng = np.random.RandomState(0)
    q, k, v = (
        _randn(rng, b, h, s, d),
        _randn(rng, b, h, t, d),
        _randn(rng, b, h, t, dv),
    )
    _save(
        'attention_dense_float32',
        q=q,
        k=k,
        v=v,
        causal=np.array(False),
        out=_ref(q, k, v),
    )

    # windowed additive bias (shared across batch: (h, s, t))
    rng = np.random.RandomState(1)
    q, k, v = (
        _randn(rng, b, h, s, d),
        _randn(rng, b, h, t, d),
        _randn(rng, b, h, t, dv),
    )
    bias = _randn(rng, h, s, t)
    _save(
        'attention_windowed_bias_float32',
        q=q,
        k=k,
        v=v,
        bias=bias,
        causal=np.array(False),
        out=_ref(q, k, v, bias=bias),
    )

    # causal (s == t)
    rng = np.random.RandomState(2)
    q, k, v = (
        _randn(rng, b, h, s, d),
        _randn(rng, b, h, s, d),
        _randn(rng, b, h, s, dv),
    )
    _save(
        'attention_causal_float32',
        q=q,
        k=k,
        v=v,
        causal=np.array(True),
        out=_ref(q, k, v, causal=True),
    )

    # cross-attention (s != t)
    rng = np.random.RandomState(3)
    s_c, t_c = 6, 9
    q, k, v = (
        _randn(rng, b, h, s_c, d),
        _randn(rng, b, h, t_c, d),
        _randn(rng, b, h, t_c, dv),
    )
    _save(
        'attention_cross_float32',
        q=q,
        k=k,
        v=v,
        causal=np.array(False),
        out=_ref(q, k, v),
    )


def regen_ssm() -> None:
    from nitrix.nn.ssm import reference_selective_scan

    rng = np.random.RandomState(10)
    b, length, d, n = 2, 6, 4, 3
    x = _randn(rng, b, length, d)
    # delta is post-softplus (positive); A is negative (contractive).
    delta = np.log1p(np.exp(_randn(rng, b, length, d))).astype(np.float32)
    a = -np.exp(_randn(rng, d, n)).astype(np.float32)
    bmat = _randn(rng, b, length, n)
    cmat = _randn(rng, b, length, n)
    dvec = _randn(rng, d)
    out = np.asarray(
        reference_selective_scan(
            x, delta, a, bmat, cmat, dvec, method='sequential'
        ),
        dtype=np.float32,
    )
    _save(
        'selective_scan_float32',
        x=x,
        delta=delta,
        A=a,
        B=bmat,
        C=cmat,
        D=dvec,
        out=out,
    )


def main() -> None:
    regen_attention()
    regen_ssm()


if __name__ == '__main__':
    main()
