# Changelog

## [0.1.1](https://github.com/jejjohnson/pyrox/compare/pyrox-nn-v0.1.0...pyrox-nn-v0.1.1) (2026-10-17)


### Features

* **nn:** re-export GeoFNO, LearnedDeformation, ManifoldNO and BasisTransform from geonnax ([#303](https://github.com/jejjohnson/pyrox/issues/303)) ([0bff178](https://github.com/jejjohnson/pyrox/commit/0bff17865652a7cd0980672aac937c68697849a0))


### Bug Fixes

* **build:** ship a license file inside each package ([#300](https://github.com/jejjohnson/pyrox/issues/300)) ([a5c0501](https://github.com/jejjohnson/pyrox/commit/a5c05015573ab80122da11c86d9b574d5176ec4e))

## 0.1.0 (2026-07-25)


### ⚠ BREAKING CHANGES

* **core:** modules without pyrox_name now register sites under their class name instead of a per-instance id-derived scope; unnamed same-class siblings in one trace collide loudly.
* pyrox.gp, pyrox.nn, pyrox._basis, pyrox.api, and pyrox.preprocessing moved to the new pyrox-gp and pyrox-nn packages.

### Bug Fixes

* **core:** deterministic site-name scoping — class-name fallback replaces id-based names ([#187](https://github.com/jejjohnson/pyrox/issues/187)) ([1602310](https://github.com/jejjohnson/pyrox/commit/1602310e1674174c574094db464236e770d1c6a3))


### Code Refactoring

* split pyrox into a three-package uv workspace (pyrox, pyrox-gp, pyrox-nn) ([#177](https://github.com/jejjohnson/pyrox/issues/177)) ([bd7a99d](https://github.com/jejjohnson/pyrox/commit/bd7a99dd4aae36509ba7cf6ef6360a3668349660))

## Changelog
