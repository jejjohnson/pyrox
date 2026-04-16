"""Layer 0 — pure JAX kernel functions.

pyrox owns the *math definitions* of kernel forms (RBF, Matern,
Periodic, Linear, RationalQuadratic, Polynomial, Cosine, White,
Constant) as small, self-contained closed-form expressions. These are
tutorial-grade and read in ten lines.

The companion *scalable construction* surface — numerically stable
matrix assembly, mixed-precision accumulation, structured operators,
batched matvec, prediction caches, Cholesky-with-jitter, solvers —
lives in `gaussx`. See `gaussx.stable_rbf_kernel`, `gaussx.cholesky`,
`gaussx.log_marginal_likelihood`, `gaussx.predict_mean`, etc.

Higher-level :class:`pyrox.gp.Kernel` subclasses (Wave 2 Layer 1, see
issue #20) wrap these formulas in a NumPyro-aware ``Parameterized``
shell and can opt in to gaussx's scalable variants when needed.
"""
