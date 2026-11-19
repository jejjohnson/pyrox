# pyrox-gauss (`pyrox_gauss`) — agent rules

Gaussian NumPyro distributions on gaussx's structured operators. The root [`AGENTS.md`](../../AGENTS.md) applies too.

## Public layout

One facade, `pyrox_gauss` (`__init__.py` re-exports; implementation in private `_*.py` modules). The classes moved from `gaussx._distributions` (jejjohnson/pyrox#453) with their names, signatures and numerics unchanged.

| Concept | Module | Names |
|---|---|---|
| The shared normal base | `_mvn_base.py` | `AbstractMultivariateNormal` |
| Covariance and precision normals | `_mvn.py`, `_mvn_prec.py` | `MultivariateNormal`, `MultivariateNormalPrecision` |
| Linear-Gaussian state-space models over observations | `_lgssm.py` | `LGSSM`, `MaskedLGSSM`, `LGSSMFactory` |
| A Markov chain over states | `_markov_gaussian.py` | `MarkovGaussian` |
| A gaussx GMRF field as a NumPyro distribution | `_gmrf.py` | `GMRFDistribution` |
| The closed-form `kl_divergence` registrations (run on import) | `_numpyro_kl.py` | none |
| Reshapes, operator views, `_LOG_2PI` (copied from gaussx, private) | `_utils.py`, `_einx.py` | none |

## Contracts

- **Never import `pyrox` or a sibling package** (`tests/test_gauss_import_guard.py`: an AST scan plus a subprocess import check). pyrox-gauss sits in tier 0 of this workspace, beside the pyrox core, and the model packages may import it.
- **Numerics go through gaussx's public API** (`gaussian_log_prob`, `sample_mvn`, `kalman_filter`, `diag_inv`, …). The one private import is `gaussx._linalg._diag_inv._diag_inv_structured`, in `_mvn_prec.py`, which keeps `MultivariateNormalPrecision.variance` on gaussx's exact structured paths with a lazy-inverse fallback; it is pinned with gaussx's version.
- **The GMRF fields stay in gaussx.** `GMRFDistribution` wraps a field and adds no numerics.

## Docs and tests

- Every public name gets a `::: pyrox_gauss.Name` entry in `docs/api/gauss.md`.
- `uv run pytest --no-cov packages/pyrox-gauss/tests`.
- The tests are gaussx's, moved with the code; they import gaussx's test helpers (`gaussx._testing`). Test file basenames start with `test_gauss_` (they must be unique across the workspace).
