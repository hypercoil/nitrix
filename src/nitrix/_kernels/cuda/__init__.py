# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
NVIDIA / Pallas Triton kernels.

Each module here implements one kernel variant; the public API in
``nitrix.semiring`` etc. selects between the Pallas path and the JAX
fallback at call time.
"""
