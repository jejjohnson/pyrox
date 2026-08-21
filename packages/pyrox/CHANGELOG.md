# Changelog

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
