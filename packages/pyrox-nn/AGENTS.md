# pyrox-nn (`pyrox_nn`) — agent rules

Bayesian and uncertainty-aware layers built by wrapping deterministic geonnax
cores, plus the Bayesian Neural Field estimator. The root
[`AGENTS.md`](../../AGENTS.md) applies too.

## Public layout

| Namespace | Holds |
|---|---|
| `pyrox_nn` | The layers: Bayesian dense (`_dense.py`), random features (`_features.py`), SIREN (`_siren.py`), MFN (`_mfn.py`), SNGP (`_sngp.py`), deep VSSGP (`_vssgp.py`), ensembles (`_ensemble.py`), heteroscedastic heads (`_heteroscedastic.py`), conditioning (`_conditioning.py`), Slepian (`_slepian.py`), BNF layers (`_bnf.py`); deterministic geonnax names re-exported for compatibility (`_geonnax.py`) |
| `pyrox_nn.api` | `BNFEstimator`, `BNFEstimatorMLE`, `BNFEstimatorVI`, `FittedBNF`, `EstimatorBase`, `FittedEstimator` (`[bnf]` extra) |
| `pyrox_nn.preprocessing` | pandas → array preprocessing (`[bnf]` extra) |

`import pyrox_nn` must never import pandas: the root `__init__` does not
import `api` or `preprocessing`, which import pandas at module scope.

## Contracts

- **A layer is a `PyroxModule`** with every configuration field
  `eqx.field(static=True)` and `pyrox_name: str | None =
  eqx.field(static=True, default=None)` (older layers declare a plain
  `str | None = None`; new ones use the static field). A layer that needs a
  key or validation is built with a `@classmethod init(...)` taking
  keyword-only options and raising `ValueError` (`BayesianSIREN`,
  `RandomFeatureGaussianProcess`); a plain-field layer such as
  `DenseReparameterization` uses the generated constructor.
- **Two shapes of layer:**
  - *pure prior* — sample the weights in a `@pyrox_method __call__` from a
    full-shape prior (`dist.Normal(0, s).expand([d_in, d_out]).to_event(2)`)
    and compute with einx (exemplar: `DenseReparameterization`,
    `BayesianSIREN`);
  - *wrapped core* — hold a geonnax core (`core: geonnax.XCore`), register
    replacements for its arrays with `pyrox_sample` / `pyrox_param`, splice
    them in with `eqx.tree_at`, then run it (exemplars: `_sngp.py`,
    `_vssgp.py`, `_heteroscedastic.py`).
- **Per-example cores** run over `(*batch, D)` inputs through
  `_batching.vmap_over_flat_batch`, with the weights sampled once per model
  call.
- **Deterministic architecture belongs in geonnax**; basis functions and
  spectral densities come from `pyrox_gp._basis`; a GP kernel's hyperprior
  sites register once inside `_kernel_context(kernel)` (see `HSGPFeatures`).
- **Raw primitives** (a KL `factor`) are named
  `self._pyrox_fullname("kl")`; `numpyro.prng_key()` needs a `seed` handler.
- **Docstrings** list the sites the layer registers and the priors, in
  MathJax.
- **The estimator** (`api/_bnf.py`) bridges a `PyroxModule` model to
  `pyrox.inference`: `init_fn` traces the model on a dummy input for the
  unobserved sites, `log_joint` replays it under `substitute`.

## Docs

Add `::: pyrox_nn.Name` to `docs/api/nn.md` or the topic page under
`docs/api/nn/`.

## Tests

- `uv run pytest --no-cov packages/pyrox-nn/tests -m "not slow"`.
- `tests/nn/` (layers), `tests/api/` and `tests/preprocessing/` (need the
  `dev` group's pandas and optax).
- Assert the exact site set of a new layer under
  `handlers.trace()` + `handlers.seed(rng_seed=0)` (see
  `tests/nn/test_siren.py::test_bayesian_siren_registers_sites`), and that it
  runs under SVI / `Predictive`.
- `tests/preprocessing/test_pandas.py::test_pandas_isolation` scans a path
  that no longer exists (`src/pyrox/nn`), so nothing checks pandas isolation
  today; don't rely on it.
