# pyrox-ssm (`pyrox_ssm`) — agent rules

State-space models as NumPyro models: temporal GPs and Gaussian state-space models whose inference recurses in time. The root [`AGENTS.md`](../../AGENTS.md) applies too.

## Public layout

One facade, `pyrox_ssm` (`__init__.py` re-exports; implementation in private `_*.py` modules), plus `pyrox_ssm.flows` behind the `[flows]` extra and `pyrox_ssm.ensemble` behind the `[ensemble]` extra. The Markov-GP track moved here from pyrox-gp with every name and signature unchanged; the old `pyrox_gp` names raise an `ImportError` naming their `pyrox_ssm` home.

| Concept | Module | Names |
|---|---|---|
| Linear-Gaussian SSM (the one `gaussx.kalman_filter` call site of the core package) | `_lgssm.py` | `LinearGaussianSSMPrior`, `ConditionedLinearGaussianSSM`, `lgssm_factor` |
| Nonlinear SSM (moment-matched `gaussx.nonlinear_kalman_filter`: EKF, UKF, CKF, Gauss-Hermite, Monte Carlo) | `_nonlinear.py` | `nonlinear_ssm_factor` |
| Markov GP (Kalman filter, RTS smoother) | `_markov.py` | `MarkovGPPrior`, `ConditionedMarkovGP`, `markov_gp_factor`, `markov_gp_sample`, `markov_gp_from_innovations`, `markov_gp_innovations_sample` |
| Sparse Markov GP | `_sparse_markov.py` | `SparseMarkovGPPrior`, `SparseConditionedMarkovGP`, `sparse_markov_elbo`, `sparse_markov_factor` |
| Non-Gaussian likelihoods | `_inference_nongauss_markov.py` | `LaplaceMarkovInference`, `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov`, `ExpectationPropagationMarkov`, `NonGaussConditionedMarkovGP`, `markov_laplace_factor` |
| Ragged sites (padding + observation mask) | `_sites.py` | `pad_sites` |
| Shared GP plus one GP per site on a merged grid (block-diagonal LGSSM) | `_hierarchical.py` | `SharedPlusSiteMarkovGP` |
| Per-site numerics (private) | `_markov_sites.py` | `as_gaussx_likelihood`, `_cavity_site_derivatives`, `_check_scalar_latent`, `_require_gauss_hermite`, `_pushforward_moments` |
| Normalizing Kalman filter (`[flows]`) | `flows/_markov_flow.py` | `pyrox_ssm.flows.NormalizingKalmanPrior`, `normalizing_kalman_factor` |
| Ensemble Kalman log likelihood of a filterax run (`[ensemble]`) | `ensemble.py` | `pyrox_ssm.ensemble.ensemble_factor` |
| SDE kernels (re-exported from gaussx) | `__init__.py` | `SDEKernel`, `SDEParams`, `MaternSDE`, `SumSDE`, `ProductSDE`, … |

## Contracts

- **Never import pyrox-gp, pyrox-lgm or pyrox-nn** (`tests/test_ssm_import_guard.py`: an AST scan plus a subprocess import check). Shared code lives in `pyrox` or gaussx. Likelihoods are any `pyrox.likelihoods.Likelihood` or `pyrox.likelihoods.AbstractObservation` and the sparse guides any `pyrox.likelihoods.Guide` with `predict(K_xz, K_zz_op, K_xx_diag)` and `kl_divergence(prior_cov_op)`, so callers pass pyrox-gp's likelihoods and guides unchanged.
- **Linear algebra goes through gaussx** (`kalman_filter`, `rts_smoother`, Gaussian densities); never `jnp.linalg` on a covariance.
- **Per-site likelihood numerics** go through `_markov_sites.as_gaussx_likelihood` and gaussx (`site_derivatives`, `expected_log_likelihood`): it wraps the `(log_density, data)` pair that `pyrox.likelihoods.DistLikelihood.build` returns into a `gaussx.CallableLikelihood`, calls `build` on one site for any other `AbstractObservation` and `log_prob` on one point for any other `Likelihood`; a check pyrox-gp also needs is copied into `_markov_sites.py`, never imported.
- **`pyrox_ssm.flows` needs the `[flows]` extra**: importing it needs `flowjax`, and a non-`None` warp needs `gauss_flows` (imported lazily by `_require_flows`). `pyrox_ssm` itself never imports `flows`.
- **`pyrox_ssm.ensemble` needs the `[ensemble]` extra** (filterax). It is the only module that imports filterax, and `pyrox_ssm` itself never imports it, so `import pyrox_ssm` works without filterax (`tests/test_ssm_import_guard.py`). The root `dev` group installs `pyrox-ssm[ensemble]`, so its tests run in CI.

## Docs and tests

- Every public name gets a `::: pyrox_ssm.Name` entry in `docs/api/ssm.md`; the notebooks live in `docs/notebooks/ssm/`.
- `uv run pytest --no-cov packages/pyrox-ssm/tests`.
- `tests/test_ssm_public_api.py` checks every `__all__` name exists and none is duplicated; `tests/test_ssm_docstrings.py` rejects Sphinx markup in the source.
- The tests import nothing from pyrox-gp or kernellib: `tests/ssm_reference.py` holds the closed-form Matérn kernel, test-local Bernoulli / Poisson / Student-t likelihoods, a full-rank guide and a per-site Gaussian expectation. Dense references use the closed-form Gram or `gaussx.sde_autocovariance` with `jnp.linalg` (fine in tests), at 1e-10 for exact equivalences under x64.
- Test file basenames must be unique across the workspace (`tests/` has no `__init__.py`); the flows tests skip without `gauss_flows`.
