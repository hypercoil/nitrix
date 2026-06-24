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

# --- metrics.joint_histogram ----------------------------------------------
# Soft joint-histogram accumulation: one-hot matmul (deterministic) vs .at[].add
# scatter (a non-associative atomic add -- RUN-TO-RUN nondeterministic on GPU;
# deterministic on CPU).  Canonical = 'onehot' (deterministic on any platform;
# O(N*bins) memory -- reproducibility's cost at scale).  This is the determinism
# half of the principle, not just cross-platform reassociation.
register_divergent_op(
    'metrics.joint_histogram',
    canonical='onehot',
    fast={'gpu': 'onehot', 'cpu': 'scatter'},
    driver_values=('onehot', 'scatter'),
    tolerance={'float32': 2e-4, 'float64': 1e-10},
    summary='joint histogram: one-hot matmul (deterministic) vs atomic scatter',
)

# --- register.field_smooth ------------------------------------------------
# The SVF/Demons/SyN velocity-field Gaussian regulariser: FIR shifted-slice
# convolution (GPU) vs Young-van Vliet recursion (CPU, sigma>=0.5).  They differ
# ~1-2% within a few sigma of the edge.  Canonical = 'fir' (the more faithful
# path; a regulariser, not the objective, so the loose tolerance is by design).
register_divergent_op(
    'register.field_smooth',
    canonical='fir',
    fast={'gpu': 'fir', 'cpu': 'recursive'},
    driver_values=('fir', 'recursive'),
    tolerance={'float32': 3e-2, 'float64': 3e-2},
    summary='velocity-field Gaussian regulariser: FIR vs Young-van Vliet recursion',
)

# --- geometry.cubic_bspline_prefilter -------------------------------------
# The CubicBSpline interpolant's recursive prefilter: parallel associative_scan
# (GPU) vs sequential scan (CPU) reassociate the long pole filter differently.
# Canonical = sequential.  (order-0/1 gathers are bit-identical cross-backend
# and are NOT divergent ops -- only this order-3 prefilter is.)
register_divergent_op(
    'geometry.cubic_bspline_prefilter',
    canonical='sequential',
    fast={'gpu': 'associative', 'cpu': 'sequential'},
    driver_values=('sequential', 'associative'),
    tolerance={'float32': 1e-4, 'float64': 1e-10},
    summary='cubic B-spline prefilter: sequential vs associative recursion',
)
