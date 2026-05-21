# -*- coding: utf-8 -*-
"""Shared pytest / hypothesis configuration.

JAX JIT-compiles each new ``(shape, dtype)`` signature on first use, so
the *first* hypothesis example for a given draw is dominated by compile
time (hundreds of ms) while replays hit the cache (sub-ms).  Hypothesis's
default 200 ms deadline therefore fires spuriously -- reported as a
``FlakyFailure`` ("Unreliable test timings") -- and the property
strategies that build large arrays trip the ``too_slow`` data-generation
health check on loaded / CI machines.

Neither reflects a correctness problem, and -- importantly -- neither is
about the *inputs* a test exercises.  Disabling deadlines and suppressing
the timing-only health checks suite-wide removes the flakiness **without
shrinking the input space**: every example hypothesis would have drawn is
still drawn and asserted, we only drop the wall-clock assertions that JIT
compilation makes unreliable.  Correctness, shrinking, and example
counts are unchanged.

(``data_too_large`` is suppressed only to silence the cosmetic
"overly large repr" warning hypothesis emits when it tries to print a
big generated array on failure; it does not gate inputs either.)
"""
from hypothesis import HealthCheck, settings

settings.register_profile(
    'nitrix',
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile('nitrix')
