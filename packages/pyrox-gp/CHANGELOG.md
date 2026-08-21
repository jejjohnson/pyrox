# Changelog

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
