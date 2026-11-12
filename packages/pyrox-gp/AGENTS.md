# pyrox-gp (`pyrox_gp`) — agent rules

Gaussian-process building blocks on the pyrox bridge: kernels with priors,
guides, likelihoods, exact / sparse / multi-output / warped GPs,
non-Gaussian inference and pathwise sampling. The root
[`AGENTS.md`](../../AGENTS.md) applies too.

## Public layout

One flat facade, `pyrox_gp` (`__init__.py` re-exports; implementation in
private `_*.py` modules, grouped by concept in `docs/capabilities.md`):

| Concept | Module | Exemplar |
|---|---|---|
| Protocols | `_protocols.py` (`Kernel` = `kernellib.AbstractKernel`; `Guide`, `Likelihood` re-exported from `pyrox.likelihoods`) | — |
| Kernels | `_kernels.py` (`_ParameterizedKernel(Parameterized, Kernel)`) | `RBF`; `Matern` for a static structural field |
| Exact GP | `_models.py` (`GPPrior`, `ConditionedGP`, `gp_factor`, `gp_sample`) | — |
| Guides | `_guides.py` | `FullRankGuide`, `WhitenedGuide`, `NaturalGuide` |
| Likelihoods | `_likelihoods.py`, `_warped.py` | `PoissonLikelihood`, `StudentTLikelihood`, `WarpedGaussianLikelihood` |
| SVGP inference | `_inference.py` (`svgp_elbo`, `svgp_factor`, `ConjugateVI`) | — |
| Non-Gaussian inference | `_inference_nongauss.py` | `LaplaceInference` |
| Sparse / inducing | `_sparse.py`, `_inducing.py`, `_inducing_init.py`, `_preconditioned.py` | `FourierInducingFeatures` |
| Multi-output | `_multi_output.py`, `_multi_output_models.py` | `ICMKernel`, `MultiOutputGPPrior` |
| Latent factor | `_latent_factor.py`, `_latent_factor_models.py`, `_latent_init.py` | — |
| Pathwise | `_pathwise.py` | `PathwiseSampler` |
| Shared helpers (private) | `_context.py` (deprecated aliases of `pyrox._core.call_context` and `call_contexts`), `_basis/` (RFF draws; the private `_isotropic_spectral_density`; deprecated re-exports of the `functax.basis` family modules), `_kernel_operator.py` (`KernelOperator`, `freeze_kernel`) | — |

The Markov / state-space track (`MarkovGPPrior`, the SDE kernels, the Markov non-Gaussian and sparse strategies, `NormalizingKalmanPrior`) lives in pyrox-ssm. `__init__.py`'s `__getattr__` raises an `ImportError` naming the `pyrox_ssm` home of each old name (`_MOVED_TO_SSM`; `tests/gp/test_moved_names.py`); pyrox-gp cannot re-export it, since pyrox-ssm is a sibling.

## Contracts

- **Kernels** subclass `_ParameterizedKernel`: a `pyrox_name` field with a
  fixed default (two instances with the same name collide in one trace, so siblings need distinct names),
  `init_*` floats, `setup()` registering positive params, a `@pyrox_method
  __call__` that calls `kernellib.functional`, a `diag` override for
  non-stationary kernels, and `_frozen_cls` / `_frozen_params` /
  `_frozen_static` so `frozen()` returns the kernellib kernel (the
  matrix-free path, `init_inducing` and `_basis` need it). Kernel math
  belongs in kernellib.
- **The kernel context.** Evaluate a kernel more than once per model call
  (Gram + `diag`, several blocks, several latents) only inside
  `with call_context(kernel):` / `call_contexts(kernels)` (from `pyrox._core`), so its
  hyperparameters are drawn once (`tests/gp/test_models.py`).
- **Guides** subclass `Guide` (`sample(key)`, `log_prob(f)`); `svgp_elbo`
  also needs `predict(K_xz, K_zz_op, K_xx_diag)` and
  `kl_divergence(prior_cov_op)`. Build with `@classmethod init(num_inducing,
  *, …)`; take `solver: AbstractSolverStrategy | None` and resolve it with
  `_resolve_solver`; do the linear algebra with gaussx
  (`gaussian_kl`, `whitened_svgp_predict`, `safe_cholesky`).
- **Likelihoods** of a family gaussx has subclass `_likelihoods._GPObservation`
  (a `pyrox.likelihoods.AbstractObservation` and a `Likelihood`): `build(y,
  theta, data)` returns the gaussx likelihood, inputs arrive in
  `data["X"]`, and the base gives `log_prob(f, y, X=None)` summed over points.
  Multi-latent ones set `latent_dim` (a property or a defaulted static field)
  and map their `(N, latent_dim)` latents to gaussx's layout in
  `_gaussx_latents`. Any other likelihood subclasses `Likelihood`. The
  inference code calls `build` on one site (`_site_log_prob`) for an
  `AbstractObservation` and `log_prob` otherwise; trainable parts are child
  modules. Only `GaussianLikelihood` (dispatched on the core
  `pyrox.likelihoods.GaussianLikelihood`, which it subclasses) gets the
  closed-form ELBO paths.
- **Inducing features** satisfy the `InducingFeatures` protocol
  (`num_features`, `K_uu(kernel, *, jitter)`, `k_ux(x, kernel)`); `K_uu`
  is an `lx.DiagonalLinearOperator` when the features are orthogonal
  (Fourier, spherical-harmonic, Laplacian), with the jitter folded in by
  `_diagonal_with_jitter`, and otherwise a PSD-tagged operator (Slepian);
  never `+ jnp.eye` on a diagonal one.
- **Inference strategies** are `eqx.Module`s with static configuration that
  satisfy `_NonGaussStrategy.fit(prior, likelihood, y)` and return
  `NonGaussConditionedGP`; reuse the helpers in `_inference_nongauss.py`
  (`_prior_K`, `_per_point_grad_hess`, `_posterior_from_diag_sites`, …).
- **Multi-output kernels** join the closed union `MultiOutputKernel` and
  `_latent_kernels` in `_multi_output_models.py`; distinct latent kernels
  need distinct `pyrox_name`s (`RBF(pyrox_name=f"RBF_q{q}")`).
- **Linear algebra** goes through gaussx with the model's `solver`
  (default `DenseSolver()`); Grams are `lx.MatrixLinearOperator(K,
  lx.positive_semidefinite_tag)` with jitter on the diagonal.
- **Optional dependencies**: `optax` (`QuasiNewtonInference`, imported
  lazily), `flows` (`gauss_flows`; tests use
  `pytest.importorskip("gauss_flows")`).

## Docs

Every public name gets a `::: pyrox_gp.Name` entry in `docs/api/gp.md`
(the multi-output model entry points are still missing there).

## Tests

- `uv run pytest --no-cov packages/pyrox-gp/tests -m "not slow"`.
- `tests/gp/` and `tests/basis/`; module-level data builders rather than
  fixtures; `handlers.seed(rng_seed=0)` around kernels with priors.
- Compare against dense references (`jnp.linalg` / numpyro
  `MultivariateNormal` on the materialised Gram, `kernellib.functional` for
  kernels); `atol` 1e-5 / 1e-6 typical, 1e-10–1e-12 for exact-equivalence
  checks under x64.
- Mark convergence / NUTS / SVI tests `slow`, and keep one unmarked smoke
  test per feature.
