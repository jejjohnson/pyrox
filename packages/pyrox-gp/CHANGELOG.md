# Changelog

## [0.3.3](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.3.2...pyrox-gp-v0.3.3) (2026-11-28)


### Features

* **gp:** laplace and gauss-newton fits take a solver choice and run on a matrix-free prior ([#583](https://github.com/jejjohnson/pyrox/issues/583)) ([d75552a](https://github.com/jejjohnson/pyrox/commit/d75552a515e10985e0e7327382d0b72fab09bd51)), closes [#551](https://github.com/jejjohnson/pyrox/issues/551)
* **gp:** matfree_solver and bbmm_solver presets ([#580](https://github.com/jejjohnson/pyrox/issues/580)) ([e2342a6](https://github.com/jejjohnson/pyrox/commit/e2342a6c948e5b6b3b63528954277b1c5a9ace92)), closes [#557](https://github.com/jejjohnson/pyrox/issues/557)
* **gp:** posterior linearisation and EP run under jit on the site posterior, with a solver choice ([#582](https://github.com/jejjohnson/pyrox/issues/582)) ([17a6008](https://github.com/jejjohnson/pyrox/commit/17a60086c05be82f9ec8db2f7408ea03e0248c17)), closes [#552](https://github.com/jejjohnson/pyrox/issues/552)
* **gp:** solver, preconditioner and logdet fields on the GP priors ([#579](https://github.com/jejjohnson/pyrox/issues/579)) ([5bea8d0](https://github.com/jejjohnson/pyrox/commit/5bea8d0646692065f17da07d29ecbbc13fd4a828)), closes [#550](https://github.com/jejjohnson/pyrox/issues/550)
* **gp:** stochastic site variances from gaussx 0.8.4 for posterior linearisation, EP and Laplace ([#584](https://github.com/jejjohnson/pyrox/issues/584)) ([5c51534](https://github.com/jejjohnson/pyrox/commit/5c51534bb5979d6e4fe329c5587f8369818673e0))


### Bug Fixes

* **gp:** preconditioned_cg_solver builds logdet="nystrom" from gaussx NystromLogdet ([#570](https://github.com/jejjohnson/pyrox/issues/570)) ([ea8ca68](https://github.com/jejjohnson/pyrox/commit/ea8ca681d905de7ad195c4da31964592f4cd5f25)), closes [#556](https://github.com/jejjohnson/pyrox/issues/556)
* **gp:** preconditioned_cg_solver takes a diagonal that varies by point ([#578](https://github.com/jejjohnson/pyrox/issues/578)) ([5de2f03](https://github.com/jejjohnson/pyrox/commit/5de2f037f3669148ed3c67b52731f24030917c26)), closes [#555](https://github.com/jejjohnson/pyrox/issues/555)
* **gp:** prior draws on a matrix-free prior raise, or draw with method="krylov" without forming the kernel matrix ([#574](https://github.com/jejjohnson/pyrox/issues/574)) ([724d455](https://github.com/jejjohnson/pyrox/commit/724d455afd769d881f44db0ace0e93d7b0424809)), closes [#559](https://github.com/jejjohnson/pyrox/issues/559)
* **gp:** register lineax.conj for KernelOperator so LSMRSolver runs on a matrix-free GP ([#571](https://github.com/jejjohnson/pyrox/issues/571)) ([439b963](https://github.com/jejjohnson/pyrox/commit/439b96307c4bb8d6cfe3ccb9493f2e15fbc0855f)), closes [#554](https://github.com/jejjohnson/pyrox/issues/554)
* **gp:** the non-Gaussian strategies refuse a matrix-free prior instead of forming the kernel matrix ([#572](https://github.com/jejjohnson/pyrox/issues/572)) ([3d41a21](https://github.com/jejjohnson/pyrox/commit/3d41a21431452373566c1f2f89a943cce1fb424d)), closes [#549](https://github.com/jejjohnson/pyrox/issues/549)
* **gp:** the pathwise sampler solves with the prior's solver and no longer forms the kernel matrix ([#573](https://github.com/jejjohnson/pyrox/issues/573)) ([ecdeb75](https://github.com/jejjohnson/pyrox/commit/ecdeb75be58c5409a2a163b321d7db042b4f033d)), closes [#558](https://github.com/jejjohnson/pyrox/issues/558)
* **gp:** the sparse guides' solver reaches predict and kl_divergence ([#585](https://github.com/jejjohnson/pyrox/issues/585)) ([9b34578](https://github.com/jejjohnson/pyrox/commit/9b34578e5fd568564c3cbc53052ae11750be01e5)), closes [#560](https://github.com/jejjohnson/pyrox/issues/560)

## [0.3.2](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.3.1...pyrox-gp-v0.3.2) (2026-11-26)


### Features

* **gp:** add gp_laplace_factor, the Laplace approximate evidence as a NumPyro factor ([#531](https://github.com/jejjohnson/pyrox/issues/531)) ([cf63c96](https://github.com/jejjohnson/pyrox/commit/cf63c9617710a6242b17824f1eec9ee64423906b))
* **gp:** add include_noise to the exact GP predict methods ([#529](https://github.com/jejjohnson/pyrox/issues/529)) ([97d6c75](https://github.com/jejjohnson/pyrox/commit/97d6c75ea422cc2b76b9e82e9c25ce72bdce8790))
* **gp:** add log_marginal on GPPrior and MultiOutputGPPrior, with the factors built on them ([#525](https://github.com/jejjohnson/pyrox/issues/525)) ([e2c7d2f](https://github.com/jejjohnson/pyrox/commit/e2c7d2faf600eb63688be12b531675931b1089ed))
* **gp:** add oilmm_factor and OILMMGPPrior.log_marginal ([#527](https://github.com/jejjohnson/pyrox/issues/527)) ([30bea2a](https://github.com/jejjohnson/pyrox/commit/30bea2ab6547bf8fc2d213ee91977b7e20d7059e))
* **gp:** add SparseGPPrior.condition(guide) and the multi-output form, returning the conditioned sparse GPs ([#526](https://github.com/jejjohnson/pyrox/issues/526)) ([dd58a33](https://github.com/jejjohnson/pyrox/commit/dd58a33b421c61a21e1ff28dc777516ac3bb6441))
* **gp:** give lfr_factor the (name, prior, ...) order of the other factors ([#528](https://github.com/jejjohnson/pyrox/issues/528)) ([c47620b](https://github.com/jejjohnson/pyrox/commit/c47620b7d44d23d1037e662a03c27e9c8da307a5))

## [0.3.1](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.3.0...pyrox-gp-v0.3.1) (2026-11-19)


### Features

* **lgm:** let SPDE.project_points take a grid with a one-node axis ([#441](https://github.com/jejjohnson/pyrox/issues/441)) ([f4e14db](https://github.com/jejjohnson/pyrox/commit/f4e14dbf32ca2b3650ef37d930e5c6733df9c46f)), closes [#397](https://github.com/jejjohnson/pyrox/issues/397)

## [0.3.0](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.2.0...pyrox-gp-v0.3.0) (2026-11-12)


### ⚠ BREAKING CHANGES

* these names are removed from pyrox_gp and importing them raises ImportError; import them from pyrox_ssm instead (`from pyrox_ssm import MarkovGPPrior`): MarkovGPPrior, ConditionedMarkovGP, markov_gp_factor, markov_gp_sample, SparseMarkovGPPrior, SparseConditionedMarkovGP, sparse_markov_elbo, sparse_markov_factor, LaplaceMarkovInference, GaussNewtonMarkovInference, PosteriorLinearizationMarkov, ExpectationPropagationMarkov, NonGaussConditionedMarkovGP, SDEKernel, SDEParams, MaternSDE, ConstantSDE, CosineSDE, PeriodicSDE, ProductSDE, QuasiPeriodicSDE, SumSDE, IntegratedWienerSDE. NormalizingKalmanPrior and normalizing_kalman_factor move to pyrox_ssm.flows (`from pyrox_ssm.flows import NormalizingKalmanPrior`, with the pyrox-ssm[flows] extra).

### Code Refactoring

* move the Markov-GP track from pyrox-gp to pyrox-ssm ([#383](https://github.com/jejjohnson/pyrox/issues/383)) ([4b57f2f](https://github.com/jejjohnson/pyrox/commit/4b57f2f512c5bb70d7320d27db242a63b9c7d043))

## [0.2.0](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.9...pyrox-gp-v0.2.0) (2026-10-23)


### ⚠ BREAKING CHANGES

* **gp:** `pyrox_gp._src.kernels` is removed; replace `from pyrox_gp._src.kernels import rbf_kernel` with `from kernellib.functional import rbf_kernel` (same for the other kernel functions).

### Features

* **gp:** gradient posterior for ConditionedGP (KD9) ([#309](https://github.com/jejjohnson/pyrox/issues/309)) ([64249a8](https://github.com/jejjohnson/pyrox/commit/64249a8a72b0737bb014a5b390d5fde080cea94d))
* **gp:** remove the pyrox_gp._src.kernels deprecation shim ([78115da](https://github.com/jejjohnson/pyrox/commit/78115da2d21d2da7519ea7668d3877fd26ae5d35))
* **nn:** re-export GeoFNO, LearnedDeformation, ManifoldNO and BasisTransform from geonnax ([#303](https://github.com/jejjohnson/pyrox/issues/303)) ([0bff178](https://github.com/jejjohnson/pyrox/commit/0bff17865652a7cd0980672aac937c68697849a0))
* **pyrox-nn:** add GaussianRegressionHead, a Bayesian heteroscedastic regression head ([#325](https://github.com/jejjohnson/pyrox/issues/325)) ([c3c1827](https://github.com/jejjohnson/pyrox/commit/c3c18273b0fe3140c420057146bf56b6e3120195))


### Bug Fixes

* **build:** ship a license file inside each package ([#300](https://github.com/jejjohnson/pyrox/issues/300)) ([a5c0501](https://github.com/jejjohnson/pyrox/commit/a5c05015573ab80122da11c86d9b574d5176ec4e))

## [0.1.9](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.8...pyrox-gp-v0.1.9) (2026-10-09)


### Features

* **gp:** matrix-free GPPrior path for large exact GPs ([#286](https://github.com/jejjohnson/pyrox/issues/286)) ([471ddce](https://github.com/jejjohnson/pyrox/commit/471ddcebb4514b79afe2eff691ae27f2d752d6f1))

## [0.1.8](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.7...pyrox-gp-v0.1.8) (2026-10-05)


### Features

* **gp:** add init_inducing for inducing-point selection (P3) ([#280](https://github.com/jejjohnson/pyrox/issues/280)) ([333e1a5](https://github.com/jejjohnson/pyrox/commit/333e1a5a2f161a55ffe861259920da7a5edf8da9)), closes [#249](https://github.com/jejjohnson/pyrox/issues/249)


### Bug Fixes

* **gp:** graph heat/Matérn spectra for LaplacianInducingFeatures (P2) ([#282](https://github.com/jejjohnson/pyrox/issues/282)) ([7cc2593](https://github.com/jejjohnson/pyrox/commit/7cc2593083cff61b2b5a0be627c316e8226d031d)), closes [#248](https://github.com/jejjohnson/pyrox/issues/248)

## [0.1.7](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.6...pyrox-gp-v0.1.7) (2026-09-26)


### Features

* **gp:** frozen() on kernels; _basis delegates to kernellib ([#238](https://github.com/jejjohnson/pyrox/issues/238)) ([53303ee](https://github.com/jejjohnson/pyrox/commit/53303eedb1e7a90d5c8841beeefcb235aa41fac5))

## [0.1.6](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.5...pyrox-gp-v0.1.6) (2026-09-25)


### Features

* **gp:** take kernel math and the Kernel base class from kernellib ([#235](https://github.com/jejjohnson/pyrox/issues/235)) ([127fe16](https://github.com/jejjohnson/pyrox/commit/127fe163371c971cf2247cb84e62c52a3f5a39cf))

## [0.1.5](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.4...pyrox-gp-v0.1.5) (2026-09-23)


### Features

* **gp:** structured Kronecker-exact multi-output GP + delegate cavity/Newton sites to gaussx ([#233](https://github.com/jejjohnson/pyrox/issues/233)) ([c831aa8](https://github.com/jejjohnson/pyrox/commit/c831aa871eed293f95ba070bd6f408a277eef0bc))

## [0.1.4](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.3...pyrox-gp-v0.1.4) (2026-08-28)


### Features

* **gp:** input-dependent warps (non-stationary transformed GPs) ([#225](https://github.com/jejjohnson/pyrox/issues/225)) ([87c0a2d](https://github.com/jejjohnson/pyrox/commit/87c0a2d2ada58d6cf7e2d0151dc65de3aa13e87c))

## [0.1.3](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.2...pyrox-gp-v0.1.3) (2026-08-26)


### Features

* **gp:** let MarkovGPPrior take a non-stationary SDE kernel (gh-222) ([#223](https://github.com/jejjohnson/pyrox/issues/223)) ([e8c8b59](https://github.com/jejjohnson/pyrox/commit/e8c8b59a24d0ddd95c7681dfa3ad78fd2994f1bd))

## [0.1.2](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.1...pyrox-gp-v0.1.2) (2026-08-24)


### Features

* **gp:** normalizing Kalman filter model surface with NumPyro hyperpriors ([#216](https://github.com/jejjohnson/pyrox/issues/216)) ([aa8d258](https://github.com/jejjohnson/pyrox/commit/aa8d2585d6706c109c04a1c2b89b6c3e8dd95516))

## [0.1.1](https://github.com/jejjohnson/pyrox/compare/pyrox-gp-v0.1.0...pyrox-gp-v0.1.1) (2026-08-21)


### Features

* **gp:** collapsed latent-factor likelihood with marginalized decoder ([#208](https://github.com/jejjohnson/pyrox/issues/208)) ([505d21a](https://github.com/jejjohnson/pyrox/commit/505d21a9f9cc02e2681ad34c0f1d8d181dd7848d))
* **gp:** latent-factor GP prior model surface + NumPyro hook ([#210](https://github.com/jejjohnson/pyrox/issues/210)) ([72979ec](https://github.com/jejjohnson/pyrox/commit/72979ecb121676d2dd90d9e98ac6b4afb9163fa1))
* **gp:** per-dimension (ARD) lengthscales for RBF, Matern, RationalQuadratic ([#209](https://github.com/jejjohnson/pyrox/issues/209)) ([a0d3d9d](https://github.com/jejjohnson/pyrox/commit/a0d3d9d997c445d0295b7a2afa98d69b246ba6de))
* **gp:** transformed Gaussian processes via a warped likelihood ([#213](https://github.com/jejjohnson/pyrox/issues/213)) ([2f28cbb](https://github.com/jejjohnson/pyrox/commit/2f28cbbeb47bde21de1d5b9a5ae903cf6ddf199a))
* **gp:** warped latent-factor regression (conjugacy-preserving output warp) ([#212](https://github.com/jejjohnson/pyrox/issues/212)) ([7b84c8c](https://github.com/jejjohnson/pyrox/commit/7b84c8c6c9d81ea2ee4f75d137e8627563de0c7a))
* **inference:** per-parameter-group learning rates + ensembled tempered MAP ([#211](https://github.com/jejjohnson/pyrox/issues/211)) ([e343f83](https://github.com/jejjohnson/pyrox/commit/e343f8380ed2e9d221d8fc3f9a7466ace4e2dd5b))

## 0.1.0 (2026-07-25)


### ⚠ BREAKING CHANGES

* pyrox.gp, pyrox.nn, pyrox._basis, pyrox.api, and pyrox.preprocessing moved to the new pyrox-gp and pyrox-nn packages.

### Features

* **gp:** add OILMM projected multi-output GP model ([#194](https://github.com/jejjohnson/pyrox/issues/194)) ([d916b1f](https://github.com/jejjohnson/pyrox/commit/d916b1f2aef9ffe9630e061e0a0f804f66eeef84))
* **gp:** multi-output GP model layer — exact and sparse SVGP ([#193](https://github.com/jejjohnson/pyrox/issues/193)) ([5b82f0a](https://github.com/jejjohnson/pyrox/commit/5b82f0afd623c4825972fdda1f75b5a07bac6df8))


### Bug Fixes

* **core:** delta guide under SVI, simplex normal guide, and _core hardening ([#186](https://github.com/jejjohnson/pyrox/issues/186)) ([b0ef508](https://github.com/jejjohnson/pyrox/commit/b0ef5083180cc24417839869ef650aaf4fc700ba))


### Code Refactoring

* split pyrox into a three-package uv workspace (pyrox, pyrox-gp, pyrox-nn) ([#177](https://github.com/jejjohnson/pyrox/issues/177)) ([bd7a99d](https://github.com/jejjohnson/pyrox/commit/bd7a99dd4aae36509ba7cf6ef6360a3668349660))

## Changelog
