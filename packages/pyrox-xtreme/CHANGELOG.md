# Changelog

## [0.0.3](https://github.com/jejjohnson/xtremax/compare/v0.0.2...v0.0.3) (2026-08-20)


### Bug Fixes

* **distributions:** stabilise GEV variance/skew/kurtosis for small ξ ([#95](https://github.com/jejjohnson/xtremax/issues/95)) ([b550c74](https://github.com/jejjohnson/xtremax/commit/b550c743f6c7898f32a2ff7758a1700d9e07e580))

## [0.0.2](https://github.com/jejjohnson/xtremax/compare/v0.0.1...v0.0.2) (2026-08-19)


### Features

* **distributions:** covariance() for the pipekit ObservationNoise seam (epic [#42](https://github.com/jejjohnson/xtremax/issues/42)) ([#89](https://github.com/jejjohnson/xtremax/issues/89)) ([43c8f76](https://github.com/jejjohnson/xtremax/commit/43c8f763873aeefb532ff0ecbaf560fd1436b2e1))
* **extraction:** record where each block maximum occurred ([#30](https://github.com/jejjohnson/xtremax/issues/30)) ([2a4b329](https://github.com/jejjohnson/xtremax/commit/2a4b329cf880c6852e010114ad3a41deaf38bc52))


### Bug Fixes

* **distributions:** correctness & NumPyro contract for the EVT classes ([#79](https://github.com/jejjohnson/xtremax/issues/79)) ([d9e77fd](https://github.com/jejjohnson/xtremax/commit/d9e77fd40bc62fcebffadf77c79d546d85fcbf17))
* extraction & simulations correctness (epic [#40](https://github.com/jejjohnson/xtremax/issues/40)) ([#86](https://github.com/jejjohnson/xtremax/issues/86)) ([6ee001a](https://github.com/jejjohnson/xtremax/commit/6ee001a041d7ca5d40eb77e886e1421d4b1ca6aa))
* **point_processes:** correctness wave — thinning sign, Hawkes GOF, gradients, invariants ([#80](https://github.com/jejjohnson/xtremax/issues/80)) ([b5dc127](https://github.com/jejjohnson/xtremax/commit/b5dc127ce7d134ad39d18a7f45386190bbdeaea8))
* **primitives:** stabilize the EVT kernels near the ξ→0 limit ([#77](https://github.com/jejjohnson/xtremax/issues/77)) ([b285925](https://github.com/jejjohnson/xtremax/commit/b285925256d911aa15ea7e9364b13cc18a1ac59a))

## 0.0.1 (2026-06-02)


### Features

* port EV distributions, extraction, and simulation primitives ([8c1b8f1](https://github.com/jejjohnson/xtremax/commit/8c1b8f14d116f08c6703eff75f19050d7a54e3c8))
* port EV distributions, extraction, simulations with pure primitive layer ([#8](https://github.com/jejjohnson/xtremax/issues/8)) ([8c1b8f1](https://github.com/jejjohnson/xtremax/commit/8c1b8f14d116f08c6703eff75f19050d7a54e3c8))
* port temporal point processes (HPP + IPP) with three-layer API ([#10](https://github.com/jejjohnson/xtremax/issues/10)) ([d74f48c](https://github.com/jejjohnson/xtremax/commit/d74f48c1bd10f1d8b8cf429540b5497c07998278))
* **primitives:** non-stationary GEV return levels + spatial building blocks ([#29](https://github.com/jejjohnson/xtremax/issues/29)) ([10d620d](https://github.com/jejjohnson/xtremax/commit/10d620df27f87983f9d8c64eec5cb41d97e5e5c9))
* renewal, Hawkes, marked, and thinning TPP families ([#11](https://github.com/jejjohnson/xtremax/issues/11)) ([2aed22d](https://github.com/jejjohnson/xtremax/commit/2aed22d604e3751ad2930d37b7f370117912ba7b))
* spatial point processes (HPP, IPP, marked) ([#12](https://github.com/jejjohnson/xtremax/issues/12)) ([16c0c63](https://github.com/jejjohnson/xtremax/commit/16c0c6348ebc0ad3e236034db002c5914d97e670))
* spatiotemporal point processes (HPP, IPP, marked, Hawkes) ([#13](https://github.com/jejjohnson/xtremax/issues/13)) ([f525e3f](https://github.com/jejjohnson/xtremax/commit/f525e3f12777b91c2e998335c9af19e8a2f794aa))


### Bug Fixes

* address PR review comments ([10cb70a](https://github.com/jejjohnson/xtremax/commit/10cb70a334994762318cb618e6ac0248f4e9fb3c))


### Miscellaneous

* release as 0.0.1 ([b1d4ba3](https://github.com/jejjohnson/xtremax/commit/b1d4ba30d3e0f4b4f0f94ba9ec9285973b3a7129))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
