---
name: add-gp-component
description: Add a GP building block to pyrox-gp other than a kernel — a variational guide, a likelihood (observation model), inducing features, a non-Gaussian inference strategy (Laplace / EP / Gauss-Newton style), a multi-output kernel or a GP model entry point — following the package's protocols, the kernel-context rule and gaussx-backed linear algebra. Use when asked to add, port or implement such a component in packages/pyrox-gp.
---

# Add a GP component (pyrox-gp)

Read `packages/pyrox-gp/AGENTS.md` first; it lists each extension point with
an exemplar. A kernel has its own skill (`add-kernel`).

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` in `pyrox_gp` and in `gaussx` (GP recipes,
quadrature likelihoods, Kalman filters, natural-parameter algebra): most
numerical cores of a GP component already exist in gaussx. Port pyrox-side
only the probabilistic wiring.

## 2. Write it against its protocol

| Component | Base / protocol | Must provide | Exemplar |
|---|---|---|---|
| Guide | `Guide` (`_protocols.py`) | `sample(key)`, `log_prob(f)`; for `svgp_elbo` also `predict(K_xz, K_zz_op, K_xx_diag) -> (mean, var)` and `kl_divergence(prior_cov_op)` | `WhitenedGuide`, `NaturalGuide` (`_guides.py`) |
| Likelihood | `Likelihood` (`_protocols.py`) | `log_prob(f, y, X=None)` summed over points; `latent_dim` static field if > 1 | `PoissonLikelihood`, `StudentTLikelihood` (`_likelihoods.py`) |
| Inducing features | `InducingFeatures` protocol (`_inducing.py`) | `num_features`, `K_uu(kernel, *, jitter)` → an `lx.DiagonalLinearOperator` for orthogonal features (jitter via `_diagonal_with_jitter`), else a PSD-tagged operator; `k_ux(x, kernel)` | `FourierInducingFeatures` |
| Non-Gaussian inference | `_NonGaussStrategy` (`_models.py`) | `fit(prior, likelihood, y) -> NonGaussConditionedGP` | `LaplaceInference` (`_inference_nongauss.py`) |
| Multi-output kernel | the `MultiOutputKernel` union (`_multi_output_models.py`) | `num_outputs`, `num_latents`, `full_covariance_operator`, `cross_covariance_operator`, `diag(X) -> (N, P)`; add it to the union and `_latent_kernels` | `ICMKernel` (`_multi_output.py`) |

Rules for all of them:

- `eqx.Module` with static configuration fields; trainable parts are child
  modules; a `@classmethod init(...)` with keyword-only options.
- Linear algebra through gaussx on PSD-tagged operators, honouring a
  `solver: AbstractSolverStrategy | None` (guides resolve it with
  `_resolve_solver`); no new `cho_solve` / `solve_triangular` /
  `jnp.linalg.inv` on a Gram.
- Every multi-call kernel evaluation inside `_kernel_context(kernel)` (or
  `_kernel_contexts` for several latents).
- Sites registered by the component use a caller-supplied name (`gp_sample`
  / `gp_factor` style) or the bridge (`pyrox_sample` in a `PyroxModule`).
- Inference strategies reuse `_prior_K`, `_per_point_grad_hess`,
  `_posterior_from_diag_sites`, `_psd_safe_cholesky`,
  `_laplace_log_marginal`; their `fit` loops are eager by design, so say so
  in the docstring.
- Docstring: the equations, the reference, the sites (if any), an
  `Examples:` block; no Sphinx markup.

## 3. Export and document

`src/pyrox_gp/__init__.py` (import + `__all__`), `::: pyrox_gp.Name` in
`docs/api/gp.md`, `make capabilities`.

## 4. Tests (`packages/pyrox-gp/tests/gp/test_<area>.py`)

- Against a dense reference: the same quantity from the materialised Gram
  with `jnp.linalg` / numpyro `MultivariateNormal`, or a closed form
  (Gaussian likelihood → exact GP). State each tolerance's source.
- Guides / likelihoods: `log_prob` and shapes; an SVI smoke run.
- Inference strategies: one unmarked tiny smoke test (`fit` converges on 5–10
  points), deeper convergence tests marked `slow`.
- Inducing features: `K_uu` matches the dense covariance of the features,
  and is diagonal when they are orthogonal (`test_inducing_features.py`
  pattern).
- A kernel with priors inside the component draws once per model call
  (duplicate-site check under `handlers.trace()`).

## 5. Verify

`uv run pytest --no-cov packages/pyrox-gp/tests -m "not slow"` plus your
slow tests, then the `pre-pr-check` skill.
