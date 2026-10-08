<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="pyrox" src="docs/assets/logo-light.svg" width="300">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml"><img alt="Type Check" src="https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/jejjohnson/pyrox"><img alt="codecov" src="https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pyrox/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pyrox/api/reference/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_treeat/"><b>Tutorials</b></a> ·
  <a href="#gallery"><b>Gallery</b></a>
</p>

**Probabilistic modeling with Equinox and NumPyro: Gaussian processes, Bayesian neural networks and latent Gaussian models.**

pyrox lets an Equinox module declare NumPyro sample and param sites under stable, module-scoped names.
NumPyro owns inference; pyrox makes modules visible to it.
Write one `__call__` and it runs unchanged under `handlers.trace`, NUTS, SVI with any `AutoGuide`, `Predictive`, `jit`, `vmap` and `grad`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
    <img alt="One PyroxModule with named sample and param sites runs unchanged under NumPyro handlers, NUTS, SVI, Predictive and JAX transforms" src="docs/assets/hero-light.svg" width="100%">
  </picture>
</p>

## The packages

pyrox is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) of four packages under [`packages/`](packages).
Each one installs on its own and pulls in what it needs.

| | Package | Import | What it holds |
|---|---|---|---|
| <img src="docs/assets/icon.svg" width="28" alt=""> | [`pyrox`](packages/pyrox) | `pyrox` | The Equinox ↔ NumPyro bridge (`PyroxModule`, `Parameterized`, `pyrox_method`) and ensemble MAP / VI inference |
| <img src="docs/assets/icon-gp.svg" width="28" alt=""> | [`pyrox-gp`](packages/pyrox-gp) | `pyrox_gp` | Kernels, guides and likelihoods; exact, sparse, Markov and multi-output GPs; pathwise sampling |
| <img src="docs/assets/icon-nn.svg" width="28" alt=""> | [`pyrox-nn`](packages/pyrox-nn) | `pyrox_nn` | Bayesian dense layers, SIREN and MFN, SNGP, deep VSSGP, and the Bayesian Neural Field estimator |
| <img src="docs/assets/icon-lgm.svg" width="28" alt=""> | [`pyrox-lgm`](packages/pyrox-lgm) | `pyrox_lgm` | Latent Gaussian models in precision form: GMRF components, PC priors and `inla()` |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/layers-dark.svg">
    <img alt="pyrox-nn builds on pyrox-gp, which builds on pyrox; pyrox-lgm builds on pyrox directly. All sit on JAX, Equinox and NumPyro and on the GeoML packages gaussx, kernellib and geonnax" src="docs/assets/layers-light.svg" width="100%">
  </picture>
</p>

Arrows point at what a package imports, and nothing points back up.
`pyrox-lgm` works in precision form and never imports `pyrox-gp`.

## Installation

pyrox is not on PyPI yet.
The name `pyrox` on PyPI belongs to an unrelated project, so `pip install pyrox` installs the wrong package.
Install from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
# Core: the bridge and ensemble inference
uv add "pyrox @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox"

# GP, NN and LGM packages; each one pulls in pyrox from the same repository
uv add "pyrox-gp @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gp"
uv add "pyrox-nn @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-nn"
uv add "pyrox-lgm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-lgm"
```

`pyrox-gp`, `pyrox-nn` and `pyrox-lgm` depend on [gaussx](https://github.com/jejjohnson/gaussx) and [kernellib](https://github.com/jejjohnson/kernellib), which are also installed from GitHub.
kernellib v0.0.15 pins gaussx v0.6.0 while pyrox needs v0.6.1, so until the next kernellib release add this override to your project's `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.6.1"]
```

Optional extras: `pyrox[optax]` for ensemble MAP, `pyrox-nn[bnf]` for the BNF estimator (pandas, optax), `pyrox-gp[flows]` for normalizing-flow warps, and `pyrox-lgm[xarray]` for xarray-backed `INLAResult` marginals.

To work on pyrox itself, clone it and run `make install`; see [Development](#development).

## Quick start

A Bayesian linear layer that owns its sites, fitted by NUTS and SVI from the same model.

```python
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from pyrox._core import PyroxModule, pyrox_method

# Shapes: N = 50 observations, D = 1 input, P = 1 output


class BayesianLinear(PyroxModule):
    pyrox_name = "BayesianLinear"  # scopes the site names
    in_features: int
    out_features: int

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N P"]:
        # W ~ 𝒩(0, I), a sample site;  b, a param site
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.out_features])
            .to_event(2),
        )  # (D, P)
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))  # (P,)
        return x @ W + b  # (N, D) → (N, P)


layer = BayesianLinear(in_features=1, out_features=1)


def model(x: Float[Array, "N D"], y: Float[Array, " N"] | None = None) -> None:
    # y = x W + b + ε,  ε ~ 𝒩(0, 0.1²)
    f = layer(x)[:, 0]  # (N, D) → (N,)
    numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)


x = jnp.linspace(-1.0, 1.0, 50)[:, None]  # (N, D)
y = 2.0 * x[:, 0] + 0.1 * jr.normal(jr.key(0), (50,))  # (N,)

# The sites NumPyro sees: ['BayesianLinear.weight', 'BayesianLinear.bias', 'obs']
sites = list(handlers.trace(handlers.seed(model, 0)).get_trace(x, y))

# Same model, two engines
mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300)
mcmc.run(jr.key(1), x, y)
svi = SVI(model, AutoNormal(model), Adam(1e-2), Trace_ELBO())
svi_result = svi.run(jr.key(2), 1000, x, y)

draws = Predictive(model, mcmc.get_samples())(jr.key(3), x)["obs"]  # (S, N), S = 300
```

## Three modeling patterns

pyrox is opinionated about how Equinox and NumPyro compose, not about when to reach for which piece.
Three patterns cover the common cases, from lightest to heaviest machinery.

**A. Plain Equinox, `eqx.tree_at`.**
When one field of an existing network becomes random, you need no pyrox machinery at all.
Sample the value in a NumPyro model and splice it into the module.

```python
def model(x, y=None):
    net = MLP(key=key)  # any eqx.Module
    W = numpyro.sample("W", prior)
    net = eqx.tree_at(lambda m: m.W, net, W)
    numpyro.sample("obs", dist.Normal(net(x), 0.1), obs=y)
```

**B. `PyroxModule` owns its sites.**
When the module is itself probabilistic (a Bayesian layer, a hierarchical component), subclass `PyroxModule`, as in the quick start.
Sites are named `<pyrox_name>.<site>`, cached per call, and stable across `jit`, `eqx.tree_at` and checkpoints.
Two instances of one class in the same model need distinct `pyrox_name`s; otherwise the trace rejects the duplicate sites.

**C. `Parameterized` for constraints, priors and guides.**
When a module has constrained hyperparameters with priors (GP kernels are the canonical case), declare them once in `setup()`.
`set_mode("model")` samples the priors for MCMC; `set_mode("guide")` draws from the per-parameter autoguides for SVI, without touching `__call__`.

```python
class RBFKernel(Parameterized):
    pyrox_name = "RBFKernel"

    def setup(self):
        self.register_param(
            "variance", jnp.array(1.0), constraint=dist.constraints.positive
        )
        self.register_param(
            "lengthscale", jnp.array(1.0), constraint=dist.constraints.positive
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.autoguide("variance", "normal")  # respects the positive constraint

    @pyrox_method
    def __call__(self, X1, X2):
        v, ell = self.get_param("variance"), self.get_param("lengthscale")
        # k(x, x′) = v exp(−‖x − x′‖² / 2ℓ²)
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ell**2, axis=-1)  # (N, M)
        return v * jnp.exp(-0.5 * sq)
```

All three patterns emit plain NumPyro sites, so they fit the same model to the same loss:

<p align="center"><img src="docs/images/readme/three_patterns_svi.png" alt="SVI loss over 400 steps for patterns A, B and C on the same latent GP classification model; the three curves overlap" width="70%"></p>

Every NumPyro handler composes with them: `trace`, `substitute`, `condition`, `scope`, `block` and `reparam`.
[`test_core_numpyro_integration.py`](packages/pyrox/tests/test_core_numpyro_integration.py) checks each one, plus MCMC, SVI, `Predictive` and the JAX transforms.

## Gallery

Every figure comes from an executed notebook in the [docs](https://jejjohnson.github.io/pyrox/).
pyrox-gp priors can be built to look like the field being modelled: here the kernel's smoothness tracks the ocean variable.

<p align="center"><img src="docs/images/readme/ocean_priors.png" alt="GP prior draws styled as sea-surface temperature (Matérn-5/2), sea-surface salinity (Matérn-3/2) and ocean colour (Matérn-1/2 with a log transform)" width="100%"></p>

<table>
  <tr>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/gp_pathwise/"><img src="docs/images/readme/pathwise_samples.png" alt="32 pathwise posterior samples of an exact GP against the analytic mean and two-sigma band"></a><br><b>Pathwise posterior samples</b><br><code>pyrox-gp</code></td>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/markov_gp_kalman/"><img src="docs/images/readme/markov_vs_dense.png" alt="Wall-clock of the log marginal likelihood: the Kalman path grows linearly in N, the dense Cholesky cubically"></a><br><b>Markov GP: linear in N, not cubic</b><br><code>pyrox-gp</code></td>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/multioutput_gp/"><img src="docs/images/readme/multioutput_gap.png" alt="A multi-output GP reconstructs a held-out gap in one output from a fully observed, correlated output"></a><br><b>Multi-output GP fills a gap</b><br><code>pyrox-gp</code></td>
  </tr>
  <tr>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/rff_as_neural_networks/"><img src="docs/images/readme/ensemble_gap.png" alt="An ensemble of 16 random-feature models: the predictive band widens across a held-out gap"></a><br><b>Ensemble band opens across a gap</b><br><code>pyrox.inference</code></td>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/lgm_mcmc_inla/"><img src="docs/images/readme/inla_vs_nuts.png" alt="Hyperparameter and coefficient marginals: the NUTS histogram against the INLA estimate, both near the truth"></a><br><b>INLA against NUTS</b><br><code>pyrox-lgm</code></td>
    <td width="33%"><a href="https://jejjohnson.github.io/pyrox/notebooks/spectral_kernel_models/"><img src="docs/images/readme/five_kernels.png" alt="2-D GP prior draws for RBF, Matérn-5/2, Matérn-3/2, Matérn-1/2 and ArcCosine kernels at one lengthscale"></a><br><b>Five kernels, one grid</b><br><code>pyrox-gp</code></td>
  </tr>
</table>

## Where it fits

pyrox is the probabilistic-modeling layer of the GeoML stack.
It builds on [gaussx](https://github.com/jejjohnson/gaussx) (structured linear algebra and Gaussians), [kernellib](https://github.com/jejjohnson/kernellib) (kernels and kernel operators) and [geonnax](https://github.com/jejjohnson/geonnax) (neural nets and basis functions), alongside [filterax](https://github.com/jejjohnson/filterax), [vardax](https://github.com/jejjohnson/vardax) and [optax_bayes](https://github.com/jejjohnson/optax_bayes).

## Documentation

- [Docs site](https://jejjohnson.github.io/pyrox/): tutorials, examples and the API reference
- [Vision](design_docs/pyrox/vision.md): motivation, user stories, design principles
- [Architecture](design_docs/pyrox/architecture.md): package layout and layer stacks
- [Boundaries](design_docs/pyrox/boundaries.md): scope and ecosystem
- [Decisions](design_docs/pyrox/decisions.md): design decisions with rationale

## Development

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install      # uv sync --all-groups + pre-commit hooks
make test         # all packages
make lint         # ruff check .  (entire repo)
make typecheck    # ty, per package
make docs-serve   # preview the docs locally
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor workflow and [`AGENTS.md`](AGENTS.md) for AI agent guidance.
The icons and diagrams are generated by [`docs/assets/render.py`](docs/assets/render.py); edit it and run `uv run --no-project python docs/assets/render.py`.

## License

MIT, see [`LICENSE`](LICENSE).
