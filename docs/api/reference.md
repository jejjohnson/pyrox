# API Reference

pyrox is probabilistic modeling with [Equinox](https://github.com/patrick-kidger/equinox)
and [NumPyro](https://num.pyro.ai): Bayesian neural networks, Gaussian processes, and
composable GP building blocks. The reference is organised by subpackage:

| Section | What's inside |
|---------|---------------|
| [Core](core.md) | The Equinox-to-NumPyro bridge — `PyroxModule`, `PyroxParam`, `PyroxSample`, `Parameterized`, `pyrox_method` |
| [GP](gp.md) | Kernels, sparse/variational GPs, guides, likelihoods, non-Gaussian inference strategies, pathwise samplers, Markov (Kalman) GPs, multi-output GPs |
| [NN](nn.md) | Bayesian and uncertainty-aware layers — dense variants, spectral / random-feature layers, SNGP, ensembles, heteroscedastic heads, the BNF stack |
| [NN — Geo encoders](nn/geo_encoders.md) | Longitude/latitude, cyclic, spherical-harmonic, and Slepian input encoders |
| [NN — Conditioning](nn/conditioning.md) | FiLM, affine conditioners, and hypernetworks for conditional neural fields |
| [NN — MFN](nn/mfn.md) | Multiplicative filter networks (Fourier / Gabor) and their Bayesian variants |
| [Inference](inference.md) | Ensemble MAP / VI runners and primitives on top of NumPyro |
| [Preprocessing](preprocessing.md) | Pandas-aware spatiotemporal feature extraction for the BNF workflow |
| [Estimator](api.md) | The high-level scikit-learn-style `BayesianNeuralFieldEstimator` |

## Conventions

A few patterns hold across the whole package:

- **Modules are pytrees.** Everything is an `equinox.Module` (often a
  `Parameterized` subclass from [Core](core.md)) — immutable, `jit` / `grad` /
  `vmap`-safe, and rendered into NumPyro models via `pyrox_sample` sites.

- **Three integration patterns.** Deterministic modules work with
  `eqx.tree_at` surgery (Pattern A), `PyroxModule` registers params/sites
  automatically (Pattern B), and `Parameterized` adds constraint-aware
  `set_prior` / `autoguide` / `set_mode("model" | "guide")` (Pattern C). See the
  regression masterclass notebooks for the same model written all three ways.

- **gaussx owns the linear algebra.** Solver strategies
  (`gaussx.DenseSolver`, `CGSolver`, `BBMMSolver`, …), structured operators,
  and Gaussian distributions come from
  [gaussx](https://github.com/jejjohnson/gaussx); every pyrox entry point with
  a `solver=` keyword accepts any `gaussx.AbstractSolverStrategy`.

- **geonnax owns the deterministic cores.** Spherical encoders, SIREN / MFN
  backbones, random-feature maps, and basis functions are re-exported from
  [geonnax](https://github.com/jejjohnson/geonnax); pyrox wraps them with
  priors and posteriors.

::: pyrox
    options:
      show_root_heading: false
      members: ["__version__"]
