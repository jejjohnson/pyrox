"""Layer 0 — deprecated home of the pure JAX kernel functions.

The closed-form kernel functions (RBF, Matern, Periodic, Linear,
RationalQuadratic, Polynomial, Cosine, White, Constant) moved to
`kernellib.functional`; ``pyrox_gp._src.kernels`` re-exports them with a
``DeprecationWarning``. The scalable construction surface (kernel operators,
mixed-precision Gram matrices, Nyström / RFF) lives in `kernellib` as well,
and the linear algebra underneath it in `gaussx`.
"""
