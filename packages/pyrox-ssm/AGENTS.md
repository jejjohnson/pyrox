# pyrox-ssm (`pyrox_ssm`) — agent rules

State-space models as NumPyro models: temporal GPs and Gaussian state-space models whose inference recurses in time. The root [`AGENTS.md`](../../AGENTS.md) applies too.

## Public layout

One facade, `pyrox_ssm`. It is an empty scaffold for now: `__all__` is empty and names are added as the Markov and state-space code moves in.

## Contracts

- **Never import pyrox-gp, pyrox-lgm or pyrox-nn** (`tests/test_ssm_import_guard.py`: an AST scan plus a subprocess import check). Shared code lives in `pyrox` or gaussx.
- **Linear algebra goes through gaussx** (`kalman_filter`, `rts_smoother`, Gaussian densities); never `jnp.linalg` on a covariance.

## Docs and tests

- `uv run pytest --no-cov packages/pyrox-ssm/tests`.
- `tests/test_ssm_public_api.py` checks every `__all__` name exists and none is duplicated.
