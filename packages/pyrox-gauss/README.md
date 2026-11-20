<div align="center">

<img alt="pyrox-gauss" src="../../docs/assets/icon-gauss.svg" width="120">

# pyrox-gauss

**Gaussian NumPyro distributions for JAX on gaussx's structured operators: multivariate normals in covariance or precision form, linear-Gaussian state-space models, Gauss-Markov chains and GMRF fields, on gaussx and lineax.**

[![Tests](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pyrox/)
[![codecov](https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pyrox)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

📖 [Documentation](https://jejjohnson.github.io/pyrox/) ·
📚 [Gauss API](https://jejjohnson.github.io/pyrox/reference/gauss/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

NumPyro's `MultivariateNormal` takes a dense covariance, so a Gaussian over a 100 × 100 grid holds a $`10^4 \times 10^4`$ matrix and factorises it in $`\mathcal{O}(N^3)`$, though the covariance may be a Kronecker product of two small ones, the precision of a random walk may be tridiagonal, and a time series may be a Markov chain with a two-dimensional state. pyrox-gauss gives NumPyro Gaussian distributions whose covariance or precision is a [lineax](https://github.com/patrick-kidger/lineax) operator, so one log density

```math
\log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{1}{2} \left( D_x \log 2\pi + \log \det \boldsymbol{\Sigma} + (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
```

is computed with the structure the operator carries, where $`\mathbf{x} \in \mathbb{R}^{D_x}`$ is the value, $`\boldsymbol{\mu} \in \mathbb{R}^{D_x}`$ the mean and $`\boldsymbol{\Sigma} \in \mathbb{R}^{D_x \times D_x}`$ the covariance as an operator (or its inverse, the precision $`\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}`$). [gaussx](https://github.com/jejjohnson/gaussx) dispatches the log-determinant, the solve and the sample on that structure: a Kronecker product factorises its two factors, a precision needs a product and no solve, and a state-space model runs a Kalman filter. Each distribution is a NumPyro `Distribution`, so it is a site under `numpyro.sample`, a prior, a likelihood with `obs=`, or a guide.

```python
# Shapes: N_x, N_y grid sizes; D_x = N_x N_y; T time steps
# (K_x, K_y, Q_rw, the state-space matrices and the field as in the quickstart)

# today (pyrox-gauss)
# f ~ 𝒩(0, K_x ⊗ K_y), never forming the (D_x, D_x) matrix
field = MultivariateNormal(jnp.zeros(D_x), gx.Kronecker(K_x, K_y))
# today (pyrox-gauss)
# x ~ 𝒩(0, Λ⁻¹) with a tridiagonal precision Λ: no solve in the log density
walk = MultivariateNormalPrecision(jnp.zeros(T), Q_rw)
# today (pyrox-gauss)
# y_1:T from x_t = F x_{t-1} + η_t, y_t = H x_t + ε_t; log p(y) by a Kalman filter
series = LGSSM(F, H, Q, R, m0, P0, T)
# today (pyrox-gauss)
# any of them as a NumPyro site
f = numpyro.sample("f", field)
```

None of it is new machinery. Every log-determinant, solve, sample, Kalman filter and GMRF field is [gaussx](https://github.com/jejjohnson/gaussx)'s, called through its public API on [lineax](https://github.com/patrick-kidger/lineax) operators; the distributions are [NumPyro](https://github.com/pyro-ppl/numpyro) `Distribution` subclasses and [Equinox](https://github.com/patrick-kidger/equinox) pytrees. The classes moved here from gaussx (#453) with their names, constructors and numerics unchanged, so moving to them is an import change; the GMRF fields themselves (`gaussx.GaussianMRF`, `IntrinsicGMRF`, `ConstrainedGMRF`, `BYM2GMRF`) stay in gaussx, because gaussx's Laplace and VB code and argminax use them as linear-algebra objects.

pyrox-gauss is one package of the pyrox workspace; the [workspace README](../../README.md) introduces the others.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gauss_overview/pyrox_gauss_overview.png" alt="pyrox-gauss in one picture: you supply a mean and a covariance or precision as a lineax operator, a state-space model, or a gaussx GMRF field; one Gaussian log density whose cost is set by the structure, where a Kronecker covariance factorises its two small factors, a precision needs a product and no solve, and a series sums one-step prediction errors by a Kalman filter; pyrox-gauss offers the multivariate normals, the state-space densities, the Gauss-Markov chain and the GMRF distribution, with KL in closed form; out comes a NumPyro site with its exact log density, samples and moments, and a closed-form KL; gaussx and lineax, NumPyro, JAX and Equinox, and einx and jaxtyping underneath" width="100%"></p>

## 🌟 Highlights

- 🧮 **Any operator as a covariance or a precision**: `MultivariateNormal` takes $`\boldsymbol{\Sigma}`$ and `MultivariateNormalPrecision` takes $`\boldsymbol{\Lambda}`$ as any lineax operator (Kronecker, block-diagonal, low-rank, diagonal, tridiagonal, dense), and both accept a gaussx solver strategy.
- ⏱️ **State-space models as densities**: `LGSSM` is a density over a whole `(T, M)` series whose `log_prob` is the Kalman marginal likelihood and whose `sample` simulates forward; `MaskedLGSSM` scores only the observed entries, exactly; `LGSSMFactory` builds one per mask.
- ⛓️ **Gauss-Markov chains**: `MarkovGaussian` is a density over a state path $`\mathbf{x}_{0:T-1}`$ for a latent prior or a structured guide, with its marginals, pairwise marginals, block-tridiagonal precision and conversions to and from the precision form derived on demand.
- 🗺️ **GMRF fields as NumPyro sites**: `GMRFDistribution` wraps any gaussx GMRF field (proper, intrinsic, constrained, BYM2) for `numpyro.sample` and adds no numerics.
- 📏 **Closed-form KL in NumPyro**: `kl_divergence` is registered between any two pyrox-gauss normals and between them and `numpyro.distributions.MultivariateNormal`, so NumPyro's mean-field ELBO uses the closed form.
- 🧱 **The workspace's floor**: pyrox-gauss imports gaussx and NumPyro and nothing from pyrox, which [`test_gauss_import_guard.py`](tests/test_gauss_import_guard.py) checks; pyrox-gp, pyrox-ssm and pyrox-lgm build on it.

## 📐 One log density, five structures

Every class is the same Gaussian log density with a different structure carried by its operator, and gaussx picks the computation from it.

| Class | Parameterised by | Event shape | `log_prob` by | Reach for it when |
|---|---|---|---|---|
| `MultivariateNormal` | $`\boldsymbol{\mu}`$, $`\boldsymbol{\Sigma}`$ as an operator | `(D_x,)` | a structured log-determinant and solve | the covariance has structure (Kronecker over a grid, low rank plus diagonal) |
| `MultivariateNormalPrecision` | $`\boldsymbol{\mu}`$, $`\boldsymbol{\Lambda}`$ as an operator | `(D_x,)` | a product with $`\boldsymbol{\Lambda}`$ and its log-determinant, no solve | the precision is sparse (a random walk, a GMRF, natural parameters) |
| `LGSSM`, `MaskedLGSSM` | $`\mathbf{F}, \mathbf{H}, \boldsymbol{\Sigma}_\eta, \boldsymbol{\Sigma}_\varepsilon`$, the prior on $`\mathbf{x}_0`$ | `(T, M)` | a Kalman filter | a series of observations from a linear state |
| `MarkovGaussian` | $`\mathbf{F}_t, \boldsymbol{\Sigma}_{\eta, t}`$ and offsets, the prior on $`\mathbf{x}_0`$ | `(T, D_x)` | the chain's transition densities | a prior or a guide over a latent path |
| `GMRFDistribution` | a gaussx GMRF field | the field's | the field's own `log_prob` | a spatial or areal field built with gaussx |

The precision form needs no solve because

```math
\log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}) = -\frac{1}{2} \left( D_x \log 2\pi - \log \det \boldsymbol{\Lambda} + (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) \right)
```

and the state-space form never builds the $`TM \times TM`$ covariance of the series: the Kalman filter returns $`\log p(\mathbf{y}_{1:T})`$ as a sum of one-step prediction errors, each a Gaussian density in $`M`$ dimensions.

## 📥 Install

We have not put pyrox-gauss on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in gaussx (from its GitHub tag), NumPyro, JAX, Equinox, lineax, einx and jaxtyping, and needs Python 3.12 or newer; it needs no override and has no extras.

```bash
uv add "pyrox-gauss @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gauss"
```

## 🧠 Quickstart: one Gaussian, three structures

Three densities, each built with structure and checked against NumPyro's dense `MultivariateNormal` on the same value: a field on a $`30 \times 40`$ grid whose covariance is the Kronecker product of two exponential kernels, $`\boldsymbol{\Sigma} = \mathbf{K}_x \otimes \mathbf{K}_y`$; a random walk of 500 steps whose precision is tridiagonal; and 200 noisy observations of a local-level model, $`x_t = x_{t-1} + \eta_t`$ and $`y_t = x_t + \varepsilon_t`$.

```python
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pyrox_gauss as pg


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N_x = 30, N_y = 40; D_x = 1200


# k(s, s') = exp(−|s − s'| / ℓ), an exponential kernel on a 1-D grid
def exponential(s, lengthscale):
    return jnp.exp(-jnp.abs(s[:, None] - s[None, :]) / lengthscale)


# K_x and K_y, PSD-tagged so gaussx can use a Cholesky
# → (N_x, N_x), (N_y, N_y)
K_x = lx.MatrixLinearOperator(
    exponential(jnp.linspace(0, 1, 30), 0.2), lx.positive_semidefinite_tag
)
K_y = lx.MatrixLinearOperator(
    exponential(jnp.linspace(0, 1, 40), 0.3), lx.positive_semidefinite_tag
)
# today (pyrox-gauss)
# f ~ 𝒩(0, K_x ⊗ K_y), the covariance kept as two factors
field = pg.MultivariateNormal(jnp.zeros(1200), gx.Kronecker(K_x, K_y))
# → (D_x,)
f = field.sample(jr.key(0))
```

**A random walk in precision form.** A first-order random walk with precision $`\tau`$ has a tridiagonal precision, $`\tau`$ times the second differences, plus 1 on the first node to make it proper; the log density is a product with it and no solve.

```python
# Shapes: T = 500 steps

# Λ = τ D, tridiagonal, with τ = 4 and the first node made proper
# → (T, T) as an operator
tau = 4.0
diagonal = jnp.full(500, 2 * tau).at[0].set(tau + 1.0).at[-1].set(tau)
off_diagonal = jnp.full(499, -tau)
Q_rw = lx.TaggedLinearOperator(
    lx.TridiagonalLinearOperator(diagonal, off_diagonal, off_diagonal),
    (lx.symmetric_tag, lx.positive_semidefinite_tag),
)
# today (pyrox-gauss)
# x ~ 𝒩(0, Λ⁻¹)
walk = pg.MultivariateNormalPrecision(jnp.zeros(500), Q_rw)
# → (T,)
x = walk.sample(jr.key(1))
```

**A local-level series.** The state-space form is a density over all 200 observations at once: $`\mathbf{F} = \mathbf{H} = 1`$, $`\sigma_\eta^2 = 0.1`$, $`\sigma_\varepsilon^2 = 0.5`$ and $`x_0 \sim \mathcal{N}(0, 1)`$.

```python
# Shapes: T = 200 steps; M = 1 observation per step

# today (pyrox-gauss)
# y_1:T from x_t = x_{t−1} + η_t, y_t = x_t + ε_t, as one density over (T, M)
one = jnp.eye(1)
series = pg.LGSSM(one, one, 0.1 * one, 0.5 * one, jnp.zeros(1), one, 200)
# → (T, M)
y = series.sample(jr.key(2))
```

**Compare.** For each, the dense reference builds the full covariance (or precision) and NumPyro's `MultivariateNormal` scores the same value.

```python
import numpyro.distributions as dist


# Shapes: as above

# Cov(y_t, y_t') = 1 + 0.1 min(t, t') + 0.5 δ_tt', the local level's
steps = jnp.arange(1, 201)
level = 1.0 + 0.1 * jnp.minimum(steps[:, None], steps[None, :]) + 0.5 * jnp.eye(200)
kron = gx.Kronecker(K_x, K_y).as_matrix()
checks = {
    "field": (field, f, dist.MultivariateNormal(jnp.zeros(1200), kron)),
    "walk": (
        walk,
        x,
        dist.MultivariateNormal(jnp.zeros(500), precision_matrix=Q_rw.as_matrix()),
    ),
    "series": (series, y, dist.MultivariateNormal(jnp.zeros(200), level)),
}
for name, (structured, value, reference) in checks.items():
    score = structured.log_prob(value)
    gap = score - reference.log_prob(value.reshape(-1))
    print(f"{name:>6}: log_prob {float(score):9.3f}, gap {float(gap):.1e}")
```

This prints

```
 field: log_prob    72.135, gap 1.4e-12
  walk: log_prob  -377.870, gap 5.1e-13
series: log_prob  -262.839, gap 1.7e-13
```

that is, the log density of each draw under its structured distribution, and how far it is from the dense reference's: about $`10^{-12}`$ or less, round-off in double precision. The structured routes never formed what the references did. The field's log density used the two factors of the Kronecker product, a $`30 \times 30`$ and a $`40 \times 40`$, where the reference factorised a $`1200 \times 1200`$ matrix; the walk's took one product with the tridiagonal $`\boldsymbol{\Lambda}`$ and no solve; the series' came from a Kalman filter over 200 one-dimensional steps. Each is a NumPyro `Distribution`, so `numpyro.sample("f", field)` puts it in a model as a prior, or with `obs=` as a likelihood.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gauss_roles/pyrox_gauss_roles-dark.svg">
    <img alt="pyrox-gauss on the GeoStack role ladder: the log density, a sample and the KL between two Gaussians as algorithms; the multivariate normals in covariance and precision form, the linear-Gaussian state-space models and their factory, the Gauss-Markov chain, the GMRF distribution and the abstract normal as models; gaussx's solver strategies, borrowed, as strategies; the moments, matrices and chain forms read off a density as primitives; lineax's and gaussx's operators and gaussx's GMRF fields, borrowed, as structure; the NumPyro calls every distribution is used through beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gauss_roles/pyrox_gauss_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the table below lists them by area.

`pyrox_gauss` is one namespace; its `__all__` holds every name below.

| Area | Symbols | Notes |
|---|---|---|
| Normals | `MultivariateNormal`, `MultivariateNormalPrecision`, `AbstractMultivariateNormal` | Covariance and precision forms over lineax operators, and their shared base (`covariance_operator`, `precision_operator`, `scale_tril`, `entropy`, `kl`) |
| State-space models | `LGSSM`, `MaskedLGSSM`, `LGSSMFactory` | Densities over a series by the Kalman filter, with masked entries marginalised exactly |
| Markov chains | `MarkovGaussian` | A density over a state path, with marginals, pairwise marginals and the precision form |
| GMRF fields | `GMRFDistribution` | Any gaussx GMRF field as a NumPyro distribution |

Importing `pyrox_gauss` also registers the closed-form `kl_divergence` between its normals and NumPyro's `MultivariateNormal`.

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[Gauss API](https://jejjohnson.github.io/pyrox/reference/gauss/)**: every public name, generated from the docstrings.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts (no import of pyrox, numerics through gaussx's public API, the GMRF fields kept in gaussx) and the tests that hold them.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gauss_ecosystem/pyrox_gauss_ecosystem-dark.svg">
    <img alt="Packages that import pyrox-gauss: pyrox-gp (MultivariateNormal and MultivariateNormalPrecision), pyrox-ssm (MultivariateNormal, and LGSSM, MaskedLGSSM and LGSSMFactory through its flows extra) and pyrox-lgm (GMRFDistribution), all in the pyrox workspace; pyrox-gauss builds on gaussx, lineax, NumPyro, JAX, Equinox, einx and jaxtyping" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gauss_ecosystem/pyrox_gauss_ecosystem-light.svg" width="100%">
  </picture>
</p>

pyrox-gauss is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack), and sits at its floor beside the pyrox core. It owns the NumPyro face of gaussx's Gaussians, and composes the rest in:

| Concern | Owner |
|---|---|
| Log-determinants, solves, samples, the Kalman filter, the GMRF fields and their structure dispatch | [`gaussx`](https://github.com/jejjohnson/gaussx) |
| Operators and their tags | [`lineax`](https://github.com/patrick-kidger/lineax) |
| The `Distribution` base, `numpyro.sample` and `kl_divergence` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Distributions as pytrees | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Axis-naming array operations and shape annotations | [`einx`](https://github.com/fferflo/einx) and [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) |

Dependencies point one way. pyrox-gauss imports neither the pyrox core nor a sibling ([`test_gauss_import_guard.py`](tests/test_gauss_import_guard.py) checks it). Three siblings import it: pyrox-gp takes `MultivariateNormal` for its GP priors and `MultivariateNormalPrecision` for `NaturalGuide`; pyrox-ssm takes `MultivariateNormal` for its sparse prior and, in `pyrox_ssm.flows` (its `flows` extra), `LGSSM`, `MaskedLGSSM` and `LGSSMFactory`; pyrox-lgm takes `GMRFDistribution` for each component's NumPyro face. Nothing outside the workspace imports it.

## 🌳 Repository layout

```
packages/pyrox-gauss/
├── src/pyrox_gauss/
│   ├── __init__.py          ← the public names
│   ├── _mvn_base.py         ← AbstractMultivariateNormal
│   ├── _mvn.py, _mvn_prec.py ← the covariance and precision normals
│   ├── _lgssm.py            ← LGSSM, MaskedLGSSM, LGSSMFactory
│   ├── _markov_gaussian.py  ← MarkovGaussian
│   ├── _gmrf.py             ← GMRFDistribution
│   └── _numpyro_kl.py       ← the closed-form kl_divergence registrations
├── tests/                   ← gaussx's tests, moved with the code, and the import guard
└── AGENTS.md                ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                     # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox-gauss/tests                # this package's tests
make lint                                                        # ruff check on the whole repo
make typecheck                                                   # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the distributions.

## 📄 License

MIT. See [LICENSE](LICENSE).
