"""Session-wide pytest configuration.

Enables float64 before any test module — and therefore before any
package under test — imports JAX-dependent code.

The flag is process-global and sticky, so the suite already ran under
x64: eight test modules set it at module scope, and pytest imports every
test module during collection, before running the first test. What
varied was *when* it was set relative to the first ``import gaussx``.
That matters because `gaussx` evaluates its Gaussian log-density
constant (``_LOG_2PI = jnp.log(2.0 * jnp.pi)``) at import time. Import
gaussx first and the constant is a float32, worth ~3e-7 on an 8-step
Kalman marginal — enough to break the 1e-13 agreements that
``tests/gp/test_markov_flow.py`` pins against dense references, and to
do it only when some other module happens to be collected first.

Setting it here removes the ordering dependence: conftest is imported
before collection begins.
"""

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)
