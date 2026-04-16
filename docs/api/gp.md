# GP API

Wave 2 ships the dense-GP foundation: kernel *math functions*, concrete
`Parameterized` kernel classes, and abstract component protocols. Scalable
matrix construction (numerically stable assembly, implicit operators,
batched matvec, Cholesky-with-jitter, solver strategies) lives in
[`gaussx`](https://github.com/jejjohnson/gaussx).

!!! note "Split with gaussx"
    pyrox owns the kernel *function* side — closed-form math primitives
    readable in a dozen lines. `gaussx` owns the *scalable construction*
    side. The `Parameterized` kernel classes below wrap the math and can
    opt in to gaussx's scalable variants when needed.

## Concrete kernels

Each `Parameterized` kernel registers its hyperparameters with positivity
constraints where appropriate. Attach priors with `set_prior`, autoguides
with `autoguide`, and flip `set_mode("model" | "guide")`.

::: pyrox.gp.RBF
::: pyrox.gp.Matern
::: pyrox.gp.Periodic
::: pyrox.gp.Linear
::: pyrox.gp.RationalQuadratic
::: pyrox.gp.Polynomial
::: pyrox.gp.Cosine
::: pyrox.gp.White
::: pyrox.gp.Constant

## Component protocols

Abstract bases for the orthogonal component stack. Wave 2 ships only the
contracts here; concrete `Solver`, `Guide`, `Integrator`, and `Likelihood`
implementations land in later waves.

::: pyrox.gp.Kernel
::: pyrox.gp.Solver
::: pyrox.gp.Guide
::: pyrox.gp.Integrator
::: pyrox.gp.Likelihood

## Math primitives

Pure JAX kernel functions. Stateless, differentiable, composable —
``(Array, ..., hyperparams) -> Gram``. No NumPyro, no protocols.

::: pyrox.gp._src.kernels
    options:
      show_root_heading: false
