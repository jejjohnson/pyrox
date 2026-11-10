# Changelog

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
