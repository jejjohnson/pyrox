# Changelog

## [0.0.9](https://github.com/jejjohnson/pyrox/compare/v0.0.8...v0.0.9) (2026-05-02)


### Features

* **gp:** non-Gaussian inference strategies and likelihoods ([#47](https://github.com/jejjohnson/pyrox/issues/47), [#48](https://github.com/jejjohnson/pyrox/issues/48)) ([#120](https://github.com/jejjohnson/pyrox/issues/120)) ([a365252](https://github.com/jejjohnson/pyrox/commit/a365252de737fd780900a640e905d3d5b6f72280))
* **nn:** dense variational dropout — sparse variational dropout layer ([#51](https://github.com/jejjohnson/pyrox/issues/51)) ([#126](https://github.com/jejjohnson/pyrox/issues/126)) ([336dbf8](https://github.com/jejjohnson/pyrox/commit/336dbf86f3742755c95244f01086a2068c7a3e61))


### Bug Fixes

* **gp:** re-land [#114](https://github.com/jejjohnson/pyrox/issues/114) — SDE composition kernels onto main ([#116](https://github.com/jejjohnson/pyrox/issues/116)) ([5b36b4d](https://github.com/jejjohnson/pyrox/commit/5b36b4dffbd39cf19a5609aaba4d12c80d6d6fb8))
* **gp:** re-land [#117](https://github.com/jejjohnson/pyrox/issues/117) — MarkovGPPrior onto main (orphaned by stacked-PR squash-merge) ([#119](https://github.com/jejjohnson/pyrox/issues/119)) ([d788dce](https://github.com/jejjohnson/pyrox/commit/d788dce477eb96180d04b22aac04dc153f8b951d))

## [0.0.8](https://github.com/jejjohnson/pyrox/compare/v0.0.7...v0.0.8) (2026-04-29)


### Features

* **gp:** add SDEKernel protocol and MaternSDE (Markov GP foundation, [#37](https://github.com/jejjohnson/pyrox/issues/37) partial) ([#113](https://github.com/jejjohnson/pyrox/issues/113)) ([67ce79b](https://github.com/jejjohnson/pyrox/commit/67ce79bf60134b3807dbd2a6ff968df244051835))
* **nn:** canonical cosine RFF layers + spectral-kernel notebook trio ([#112](https://github.com/jejjohnson/pyrox/issues/112)) ([5840f8c](https://github.com/jejjohnson/pyrox/commit/5840f8cf07c2c2020b622a932c7816a3825684c3))

## [0.0.7](https://github.com/jejjohnson/pyrox/compare/v0.0.6...v0.0.7) (2026-04-28)


### Features

* **gp:** multi-output kernels (LMC/ICM/OILMM) + gaussx-structured inducing operators ([#100](https://github.com/jejjohnson/pyrox/issues/100)) ([a2451a8](https://github.com/jejjohnson/pyrox/commit/a2451a848f1101f482ab116fc0e5bcbfe5c156a7))
* **gp:** pathwise posterior samplers via Matheron's rule ([#39](https://github.com/jejjohnson/pyrox/issues/39)) ([#101](https://github.com/jejjohnson/pyrox/issues/101)) ([714d997](https://github.com/jejjohnson/pyrox/commit/714d99797758268438654e5e47c74f1e651c8b26))
* **nn,gp:** spectral layers + inducing features for scalable GPs ([#41](https://github.com/jejjohnson/pyrox/issues/41) + [#49](https://github.com/jejjohnson/pyrox/issues/49)) ([#84](https://github.com/jejjohnson/pyrox/issues/84)) ([c122a87](https://github.com/jejjohnson/pyrox/commit/c122a87544d244192db622731d24b4a648eba853))
* **nn:** geographic and spherical coordinate encoders ([#99](https://github.com/jejjohnson/pyrox/issues/99)) ([1709f5f](https://github.com/jejjohnson/pyrox/commit/1709f5fc2958795fc587c2791ce1cb1ab5fdb7ea))
* **nn:** siren — sinusoidal representation network layer + composite ([#98](https://github.com/jejjohnson/pyrox/issues/98)) ([6f15abd](https://github.com/jejjohnson/pyrox/commit/6f15abd0f0048dbcd601bb4debcc879f889095a6))
* **nn:** unified conditioning API — FiLM, HyperLinear, HyperSIREN, HyperFourierFeatures ([#104](https://github.com/jejjohnson/pyrox/issues/104)) ([7232c7c](https://github.com/jejjohnson/pyrox/commit/7232c7c78a844ac08bf88600502996b6620a020e))

## [0.0.6](https://github.com/jejjohnson/pyrox/compare/v0.0.5...v0.0.6) (2026-04-21)


### Features

* **inference:** add layered ensemble-of-MAP / ensemble-of-VI runner ([#81](https://github.com/jejjohnson/pyrox/issues/81)) ([8c44f8f](https://github.com/jejjohnson/pyrox/commit/8c44f8fb84d95fbbf7a67d95b212a4a21e45504d))
* **nn:** add Bayesian Neural Field layer family, preprocessing, and BNFEstimator ([#83](https://github.com/jejjohnson/pyrox/issues/83)) ([acbda2c](https://github.com/jejjohnson/pyrox/commit/acbda2c890f935d90f5c80a5e7a2738a508d47ad))

## [0.0.5](https://github.com/jejjohnson/pyrox/compare/v0.0.4...v0.0.5) (2026-04-16)


### Features

* **nn:** bayesian dense layers, SSGP random features, MC dropout, NCP ([#31](https://github.com/jejjohnson/pyrox/issues/31), [#32](https://github.com/jejjohnson/pyrox/issues/32)) ([#68](https://github.com/jejjohnson/pyrox/issues/68)) ([5e75815](https://github.com/jejjohnson/pyrox/commit/5e75815a03f71bf24393b104e33b1c49bed247ce))

## [0.0.4](https://github.com/jejjohnson/pyrox/compare/v0.0.3...v0.0.4) (2026-04-16)


### Features

* **gp:** natural guide, delta guide, and gaussx-backed guide utilities ([#29](https://github.com/jejjohnson/pyrox/issues/29)) ([#66](https://github.com/jejjohnson/pyrox/issues/66)) ([b709a48](https://github.com/jejjohnson/pyrox/commit/b709a48644626e072a00389fac904848232f9c2b))
* **gp:** svgp inference entry points, CVI, and likelihood families ([#30](https://github.com/jejjohnson/pyrox/issues/30)) ([#67](https://github.com/jejjohnson/pyrox/issues/67)) ([67d80a8](https://github.com/jejjohnson/pyrox/commit/67d80a80a0d9902e8bfc8ecb0d4602c359fdd26b))
* **gp:** wave 3 prior-side whitening + sparse SVGP building blocks ([#28](https://github.com/jejjohnson/pyrox/issues/28)) ([#64](https://github.com/jejjohnson/pyrox/issues/64)) ([4bd48c6](https://github.com/jejjohnson/pyrox/commit/4bd48c6cff0b4c8c2b509d027a876a1f5a2c72dc))

## [0.0.3](https://github.com/jejjohnson/pyrox/compare/v0.0.2...v0.0.3) (2026-04-16)


### Features

* **gp:** wave 2 Epic 2.A — kernel math primitives + Parameterized kernel classes ([#60](https://github.com/jejjohnson/pyrox/issues/60)) ([f6ee3e1](https://github.com/jejjohnson/pyrox/commit/f6ee3e1592ea4ec322e2d02a5a0c6489fe93f1ed))
* **gp:** wave 2 Epic 2.B — GPPrior, ConditionedGP, gp_factor, gp_sample ([#62](https://github.com/jejjohnson/pyrox/issues/62)) ([bd318eb](https://github.com/jejjohnson/pyrox/commit/bd318eb5d48cf7f006a18a7097d0f722a2edcf31))

## [0.0.2](https://github.com/jejjohnson/pyrox/compare/v0.0.1...v0.0.2) (2026-04-16)


### Features

* **_core:** implement PyroxModule, Parameterized, and context lifecycle ([#57](https://github.com/jejjohnson/pyrox/issues/57)) ([bfdb3d2](https://github.com/jejjohnson/pyrox/commit/bfdb3d2a7cea306b248514f0b7e517633a799cab))

## 0.0.1 (2026-04-15)


### Features

* **wave-0:** rename scaffold to pyrox and bootstrap package identity ([#53](https://github.com/jejjohnson/pyrox/issues/53)) ([ea2c95a](https://github.com/jejjohnson/pyrox/commit/ea2c95a09b4796d21aa9a0bd3b5c067c08584313))

## Changelog
