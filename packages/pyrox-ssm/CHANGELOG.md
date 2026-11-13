# Changelog

## [0.2.0](https://github.com/jejjohnson/pyrox/compare/pyrox-ssm-v0.1.0...pyrox-ssm-v0.2.0) (2026-11-12)


### ⚠ BREAKING CHANGES

* these names are removed from pyrox_gp and importing them raises ImportError; import them from pyrox_ssm instead (`from pyrox_ssm import MarkovGPPrior`): MarkovGPPrior, ConditionedMarkovGP, markov_gp_factor, markov_gp_sample, SparseMarkovGPPrior, SparseConditionedMarkovGP, sparse_markov_elbo, sparse_markov_factor, LaplaceMarkovInference, GaussNewtonMarkovInference, PosteriorLinearizationMarkov, ExpectationPropagationMarkov, NonGaussConditionedMarkovGP, SDEKernel, SDEParams, MaternSDE, ConstantSDE, CosineSDE, PeriodicSDE, ProductSDE, QuasiPeriodicSDE, SumSDE, IntegratedWienerSDE. NormalizingKalmanPrior and normalizing_kalman_factor move to pyrox_ssm.flows (`from pyrox_ssm.flows import NormalizingKalmanPrior`, with the pyrox-ssm[flows] extra).

### Features

* **ssm:** add ensemble_factor over filterax.differentiable_assimilate, behind [ensemble] ([#407](https://github.com/jejjohnson/pyrox/issues/407)) ([06f64d3](https://github.com/jejjohnson/pyrox/commit/06f64d3871270718684601a969537721984ffed7))
* **ssm:** add LinearGaussianSSMPrior and route MarkovGPPrior's Kalman filtering through it ([#398](https://github.com/jejjohnson/pyrox/issues/398)) ([3fbdb1b](https://github.com/jejjohnson/pyrox/commit/3fbdb1b5d6f6b754fca35ba97373656ae2a047f9)), closes [#365](https://github.com/jejjohnson/pyrox/issues/365)
* **ssm:** add markov_gp_from_innovations, a whitened linear-time latent path for NUTS and SVI ([#404](https://github.com/jejjohnson/pyrox/issues/404)) ([9a4fc25](https://github.com/jejjohnson/pyrox/commit/9a4fc250af31e82690bfbbcf136a61cc8800a4a6))
* **ssm:** add nonlinear_ssm_factor over gaussx's moment-matched nonlinear Kalman filter ([#405](https://github.com/jejjohnson/pyrox/issues/405)) ([2875e04](https://github.com/jejjohnson/pyrox/commit/2875e04af65303f48924e2b2925594516fd98382)), closes [#368](https://github.com/jejjohnson/pyrox/issues/368)
* **ssm:** add SharedPlusSiteMarkovGP, a shared GP plus one GP per site on a merged time grid ([#403](https://github.com/jejjohnson/pyrox/issues/403)) ([8fb9c6b](https://github.com/jejjohnson/pyrox/commit/8fb9c6b57cf2f87c9808ba8d419229c2a8daf7d1)), closes [#370](https://github.com/jejjohnson/pyrox/issues/370)
* **ssm:** first-class observation masks on the Markov GP, and pad_sites for ragged sites ([#385](https://github.com/jejjohnson/pyrox/issues/385)) ([34fc398](https://github.com/jejjohnson/pyrox/commit/34fc398d5dcf0d1d2e19a57a12e7d18f886f4bb3))
* **ssm:** run the non-Gaussian Markov strategies on optimistix fixed points so they jit and differentiate ([#400](https://github.com/jejjohnson/pyrox/issues/400)) ([017f0e3](https://github.com/jejjohnson/pyrox/commit/017f0e3f85e0e50feb799687634abb28aacb2317)), closes [#367](https://github.com/jejjohnson/pyrox/issues/367)


### Bug Fixes

* **ssm:** report the corrected evidence from PosteriorLinearizationMarkov and ExpectationPropagationMarkov ([#401](https://github.com/jejjohnson/pyrox/issues/401)) ([fa8de57](https://github.com/jejjohnson/pyrox/commit/fa8de573c426fa989b63c8e3e7cad9de4f00bb54)), closes [#392](https://github.com/jejjohnson/pyrox/issues/392)


### Code Refactoring

* move the Markov-GP track from pyrox-gp to pyrox-ssm ([#383](https://github.com/jejjohnson/pyrox/issues/383)) ([4b57f2f](https://github.com/jejjohnson/pyrox/commit/4b57f2f512c5bb70d7320d27db242a63b9c7d043))

## 0.1.0 (2026-11-10)


### Features

* **ssm:** add the packages/pyrox-ssm package scaffold ([#378](https://github.com/jejjohnson/pyrox/issues/378)) ([ad0e87c](https://github.com/jejjohnson/pyrox/commit/ad0e87cb83721f117afd7db17110c340bf08859b))

## Changelog
