"""Deprecated: the pure kernel functions moved to `kernellib.functional`.

This module re-exports them unchanged so existing imports keep working, and
warns on import. The math, and every kernel value, is identical: kernellib
ported this module and its tests verbatim.

Replace

    from pyrox_gp._src.kernels import rbf_kernel

with

    from kernellib.functional import rbf_kernel
"""

from __future__ import annotations

import warnings

from kernellib.functional import (
    constant_kernel as constant_kernel,
    cosine_kernel as cosine_kernel,
    kernel_add as kernel_add,
    kernel_mul as kernel_mul,
    linear_kernel as linear_kernel,
    matern_kernel as matern_kernel,
    periodic_kernel as periodic_kernel,
    polynomial_kernel as polynomial_kernel,
    rational_quadratic_kernel as rational_quadratic_kernel,
    rbf_kernel as rbf_kernel,
    white_kernel as white_kernel,
)


warnings.warn(
    "pyrox_gp._src.kernels is deprecated; import the kernel functions from "
    "kernellib.functional instead.",
    DeprecationWarning,
    stacklevel=2,
)
