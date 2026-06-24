# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Central manifest of numerically-divergent dispatch sites (the ``driver`` axis).

Registered **eagerly** (imported by ``nitrix.__init__``) so that:

- ``nitrix.divergent_ops()`` is complete the moment ``import nitrix`` runs --
  a parity-sensitive consumer can audit every divergent op without importing
  each subpackage that owns one; and
- the P4 completeness guard has a single source of truth to check the
  ``resolve_driver(op=...)`` call sites against.

Each entry is the cross-variant **contract**: the ``canonical`` variant
reproducibility mode forces, the hardware-``fast`` pick (descriptive), the
accepted ``driver`` values, and the per-dtype tolerance the golden corpus pins
``variant ~= canonical`` against (suite P3).  The site code (in its own module)
only references the op by name via ``resolve_driver``; the contract lives here.

See ``docs/feature-requests/reproducible-dispatch.md``.
"""

from __future__ import annotations

from .config import register_divergent_op

# --- nn.ssm.selective_scan ------------------------------------------------
# sequential lax.scan vs parallel associative_scan reassociate the first-order
# recurrence differently; 'chunked' is a third (XLA-stable, memory-sparing)
# variant.  Canonical = the bit-exact sequential oracle.
register_divergent_op(
    'nn.ssm.selective_scan',
    canonical='sequential',
    fast={'gpu': 'associative', 'cpu': 'sequential'},
    driver_values=('sequential', 'associative', 'chunked'),
    tolerance={'float32': 2e-3, 'float64': 1e-10},
    summary='selective scan: sequential vs associative vs chunked',
)

# --- signal.iir -----------------------------------------------------------
# FFT-convolution (truncates the impulse response at impulse_atol) vs sequential
# 'scan' vs parallel 'associative' recurrence.  Canonical = sequential 'scan'.
register_divergent_op(
    'signal.iir',
    canonical='scan',
    fast={'gpu': 'fft', 'cpu': 'scan'},
    driver_values=('fft', 'scan', 'associative'),
    tolerance={'float32': 1e-4, 'float64': 1e-9},
    summary='IIR SOS cascade: FFT-convolution vs sequential vs associative scan',
)
