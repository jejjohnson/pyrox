# Changelog

## [0.3.1](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.3.0...pyrox-v0.3.1) (2026-11-14)


### Features

* **inference:** return each member's starting parameters from EnsembleVI.run and EnsembleMAP.run ([#430](https://github.com/jejjohnson/pyrox/issues/430)) ([4c1ba84](https://github.com/jejjohnson/pyrox/commit/4c1ba8416aef0d6ca65f8ed29db3cfb8ae8cf119))


### Bug Fixes

* **inference:** let an autoguide be reused after ensemble_vi, and accept any data tuple ([#429](https://github.com/jejjohnson/pyrox/issues/429)) ([9ed0693](https://github.com/jejjohnson/pyrox/commit/9ed06936900815e602ab4ad1b7ef78d5ad18d094))

## [0.3.0](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.2.1...pyrox-v0.3.0) (2026-11-12)


### ⚠ BREAKING CHANGES

* these names are removed from pyrox_gp and importing them raises ImportError; import them from pyrox_ssm instead (`from pyrox_ssm import MarkovGPPrior`): MarkovGPPrior, ConditionedMarkovGP, markov_gp_factor, markov_gp_sample, SparseMarkovGPPrior, SparseConditionedMarkovGP, sparse_markov_elbo, sparse_markov_factor, LaplaceMarkovInference, GaussNewtonMarkovInference, PosteriorLinearizationMarkov, ExpectationPropagationMarkov, NonGaussConditionedMarkovGP, SDEKernel, SDEParams, MaternSDE, ConstantSDE, CosineSDE, PeriodicSDE, ProductSDE, QuasiPeriodicSDE, SumSDE, IntegratedWienerSDE. NormalizingKalmanPrior and normalizing_kalman_factor move to pyrox_ssm.flows (`from pyrox_ssm.flows import NormalizingKalmanPrior`, with the pyrox-ssm[flows] extra).

### Features

* **ssm:** add ensemble_factor over filterax.differentiable_assimilate, behind [ensemble] ([#407](https://github.com/jejjohnson/pyrox/issues/407)) ([06f64d3](https://github.com/jejjohnson/pyrox/commit/06f64d3871270718684601a969537721984ffed7))


### Code Refactoring

* move the Markov-GP track from pyrox-gp to pyrox-ssm ([#383](https://github.com/jejjohnson/pyrox/issues/383)) ([4b57f2f](https://github.com/jejjohnson/pyrox/commit/4b57f2f512c5bb70d7320d27db242a63b9c7d043))

## [0.2.1](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.2.0...pyrox-v0.2.1) (2026-11-10)


### Features

* **ssm:** add the packages/pyrox-ssm package scaffold ([#378](https://github.com/jejjohnson/pyrox/issues/378)) ([ad0e87c](https://github.com/jejjohnson/pyrox/commit/ad0e87cb83721f117afd7db17110c340bf08859b))

## [0.2.0](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.1.2...pyrox-v0.2.0) (2026-11-10)


### ⚠ BREAKING CHANGES

* **nn:** pyrox.inference.last_layer_laplace and LastLayerLaplace, released in pyrox-v0.1.2, are now pyrox_nn.last_layer_laplace and pyrox_nn.LastLayerLaplace, and the pyrox[laplace] extra is gone.

### Code Refactoring

* **nn:** move last_layer_laplace from the pyrox core to pyrox-nn ([#338](https://github.com/jejjohnson/pyrox/issues/338)) ([eb160ad](https://github.com/jejjohnson/pyrox/commit/eb160addfad27177ce45409ead31fbbc7a373a4e))

## [0.1.2](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.1.1...pyrox-v0.1.2) (2026-10-23)


### Features

* **pyrox:** add last-layer ggn laplace posterior ([#324](https://github.com/jejjohnson/pyrox/issues/324)) ([7d57fc5](https://github.com/jejjohnson/pyrox/commit/7d57fc5ab497d9e0dd49d3fed5cc5d2bda22a3ad))


### Bug Fixes

* **build:** ship a license file inside each package ([#300](https://github.com/jejjohnson/pyrox/issues/300)) ([a5c0501](https://github.com/jejjohnson/pyrox/commit/a5c05015573ab80122da11c86d9b574d5176ec4e))

## [0.1.1](https://github.com/jejjohnson/pyrox/compare/pyrox-v0.1.0...pyrox-v0.1.1) (2026-08-21)


### Features

* **gp:** per-dimension (ARD) lengthscales for RBF, Matern, RationalQuadratic ([#209](https://github.com/jejjohnson/pyrox/issues/209)) ([a0d3d9d](https://github.com/jejjohnson/pyrox/commit/a0d3d9d997c445d0295b7a2afa98d69b246ba6de))
* **inference:** per-parameter-group learning rates + ensembled tempered MAP ([#211](https://github.com/jejjohnson/pyrox/issues/211)) ([e343f83](https://github.com/jejjohnson/pyrox/commit/e343f8380ed2e9d221d8fc3f9a7466ace4e2dd5b))

## 0.1.0 (2026-07-25)


### ⚠ BREAKING CHANGES

* **core:** modules without pyrox_name now register sites under their class name instead of a per-instance id-derived scope; unnamed same-class siblings in one trace collide loudly.
* pyrox.gp, pyrox.nn, pyrox._basis, pyrox.api, and pyrox.preprocessing moved to the new pyrox-gp and pyrox-nn packages.

### Bug Fixes

* **core:** delta guide under SVI, simplex normal guide, and _core hardening ([#186](https://github.com/jejjohnson/pyrox/issues/186)) ([b0ef508](https://github.com/jejjohnson/pyrox/commit/b0ef5083180cc24417839869ef650aaf4fc700ba))
* **core:** deterministic site-name scoping — class-name fallback replaces id-based names ([#187](https://github.com/jejjohnson/pyrox/issues/187)) ([1602310](https://github.com/jejjohnson/pyrox/commit/1602310e1674174c574094db464236e770d1c6a3))


### Code Refactoring

* split pyrox into a three-package uv workspace (pyrox, pyrox-gp, pyrox-nn) ([#177](https://github.com/jejjohnson/pyrox/issues/177)) ([bd7a99d](https://github.com/jejjohnson/pyrox/commit/bd7a99dd4aae36509ba7cf6ef6360a3668349660))

## Changelog
