# Changelog

## [0.0.1](https://github.com/jejjohnson/pyrox/compare/pyrox-xtreme-v0.0.4...pyrox-xtreme-v0.0.1) (2026-11-20)


### Features

* add spatial extremes curriculum and dual-tool documentation build (jejjohnson/xtremax[#104](https://github.com/jejjohnson/pyrox/issues/104)) ([6d431ff](https://github.com/jejjohnson/pyrox/commit/6d431ff04bf8f56851ce7ece69f079ed9d495161))
* **distributions:** covariance() for the pipekit ObservationNoise seam (epic jejjohnson/xtremax[#42](https://github.com/jejjohnson/pyrox/issues/42)) (jejjohnson/xtremax[#89](https://github.com/jejjohnson/pyrox/issues/89)) ([11f5554](https://github.com/jejjohnson/pyrox/commit/11f555429dc6ea45d9e99bd70dfca8125116965f))
* **extraction:** record where each block maximum occurred (jejjohnson/xtremax[#30](https://github.com/jejjohnson/pyrox/issues/30)) ([0fc1663](https://github.com/jejjohnson/pyrox/commit/0fc16636afe55e4fdb17739a8e1c6ce273f528af))
* port EV distributions, extraction, and simulation primitives ([c9563b3](https://github.com/jejjohnson/pyrox/commit/c9563b3afe2586a6f7e64d9a0b46e6d864b01ae6))
* port EV distributions, extraction, simulations with pure primitive layer (jejjohnson/xtremax[#8](https://github.com/jejjohnson/pyrox/issues/8)) ([c9563b3](https://github.com/jejjohnson/pyrox/commit/c9563b3afe2586a6f7e64d9a0b46e6d864b01ae6))
* port temporal point processes (HPP + IPP) with three-layer API (jejjohnson/xtremax[#10](https://github.com/jejjohnson/pyrox/issues/10)) ([8d4ff73](https://github.com/jejjohnson/pyrox/commit/8d4ff73057c23c6ee115ee13740f9eaad9b01aef))
* **primitives:** non-stationary GEV return levels + spatial building blocks (jejjohnson/xtremax[#29](https://github.com/jejjohnson/pyrox/issues/29)) ([5011f4a](https://github.com/jejjohnson/pyrox/commit/5011f4adf73066033139af720303a09e3f7ed84c))
* renewal, Hawkes, marked, and thinning TPP families (jejjohnson/xtremax[#11](https://github.com/jejjohnson/pyrox/issues/11)) ([2f617bd](https://github.com/jejjohnson/pyrox/commit/2f617bd431a1520f793811754874926d626050d1))
* spatial point processes (HPP, IPP, marked) (jejjohnson/xtremax[#12](https://github.com/jejjohnson/pyrox/issues/12)) ([9784d69](https://github.com/jejjohnson/pyrox/commit/9784d697abc58ee4fa1493ae28c47cfaeb070440))
* spatiotemporal point processes (HPP, IPP, marked, Hawkes) (jejjohnson/xtremax[#13](https://github.com/jejjohnson/pyrox/issues/13)) ([27be606](https://github.com/jejjohnson/pyrox/commit/27be606f494316834bf7347cc29c3d977a5a6e72))


### Bug Fixes

* address PR review comments ([e25ac61](https://github.com/jejjohnson/pyrox/commit/e25ac618f9e087e8ebdb19954a58a35069554994))
* **distributions:** correctness & NumPyro contract for the EVT classes (jejjohnson/xtremax[#79](https://github.com/jejjohnson/pyrox/issues/79)) ([60b0d6f](https://github.com/jejjohnson/pyrox/commit/60b0d6fa2d716d233f90323a7b0ef00a084ff984))
* **distributions:** stabilise GEV variance/skew/kurtosis for small ξ (jejjohnson/xtremax[#95](https://github.com/jejjohnson/pyrox/issues/95)) ([210d13b](https://github.com/jejjohnson/pyrox/commit/210d13bda93fd0c807c4c33c4f2c49eca7d953ed))
* extraction & simulations correctness (epic jejjohnson/xtremax[#40](https://github.com/jejjohnson/pyrox/issues/40)) (jejjohnson/xtremax[#86](https://github.com/jejjohnson/pyrox/issues/86)) ([d33cac6](https://github.com/jejjohnson/pyrox/commit/d33cac6ae8957a4b6e4150d489d9f81ae178ea37))
* **point_processes:** correctness wave — thinning sign, Hawkes GOF, gradients, invariants (jejjohnson/xtremax[#80](https://github.com/jejjohnson/pyrox/issues/80)) ([3eb972c](https://github.com/jejjohnson/pyrox/commit/3eb972cae54e5f193aac828172ba5fb5286bcdee))
* **primitives:** stabilize the EVT kernels near the ξ→0 limit (jejjohnson/xtremax[#77](https://github.com/jejjohnson/pyrox/issues/77)) ([5633c0b](https://github.com/jejjohnson/pyrox/commit/5633c0bcc96b855cb2b0787c2e2c751df44dd7cb))
* **xtreme:** a batched renewal law gives one event sequence per process ([13b4626](https://github.com/jejjohnson/pyrox/commit/13b462697d0c0a20508c9ff9640390cb39a2b5b6))
* **xtreme:** batched exponential-Hawkes parameters pair with their own event row ([0b2b731](https://github.com/jejjohnson/pyrox/commit/0b2b7318101b6fce40de2e1ccaae4fd4448a5a9c))
* **xtreme:** declustering and the extremal index accept chunked inputs ([6ae41e5](https://github.com/jejjohnson/pyrox/commit/6ae41e569114401b02ac73cf97999cbfa985af6b))
* **xtreme:** Fréchet and Weibull survival goes through the stable GEV primitives ([64cc9bb](https://github.com/jejjohnson/pyrox/commit/64cc9bbc81b84450be31e37a69f0d4be32971eae))
* **xtreme:** GEV and Gumbel log survival stay in log space in the deep tail, and the Gumbel density is -inf at -inf ([b636c74](https://github.com/jejjohnson/pyrox/commit/b636c7412651b91a29dda9ac7791c8cdeab8f7cf))
* **xtreme:** the GEV and GPD endpoint density applies at the endpoint only ([9340fe8](https://github.com/jejjohnson/pyrox/commit/9340fe8b160048650bff0d6f44327c52ec49b04e))
* **xtreme:** the GEV, Fréchet and Weibull mean excess takes S(u) from the stable survival function ([699b9ca](https://github.com/jejjohnson/pyrox/commit/699b9ca512892db68385de57f7e9a55fdcaceec2))
* **xtreme:** the non-stationary return-level solve grows its lower bracket too ([1f56072](https://github.com/jejjohnson/pyrox/commit/1f56072ab1ff804d2cb94cc6df3f29c55f533659))
* **xtreme:** the renewal count on [s, e] is m(e) - m(s) ([34a6a2e](https://github.com/jejjohnson/pyrox/commit/34a6a2ea1103877ef2e4d9e619f5857226095e10))


### Miscellaneous

* release as 0.0.1 ([b6b593f](https://github.com/jejjohnson/pyrox/commit/b6b593fc52fc503261bed1153128ef1c588fc974))

## [0.0.4](https://github.com/jejjohnson/xtremax/compare/v0.0.3...v0.0.4) (2026-09-25)


### Features

* add spatial extremes curriculum and dual-tool documentation build ([#104](https://github.com/jejjohnson/xtremax/issues/104)) ([c0fe6e7](https://github.com/jejjohnson/xtremax/commit/c0fe6e7e895c9141c399e3c946989dbb5e29e89e))

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
