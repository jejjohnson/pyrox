# pyrox-ssm

State-space models as NumPyro models for JAX: temporal GPs and Gaussian state-space models whose inference recurses in time.
It builds on [`pyrox`](../pyrox) and [gaussx](https://github.com/jejjohnson/gaussx) (Kalman filtering and smoothing, Gaussian algebra).

The boundary between the pyrox packages is the inference: [`pyrox-gp`](../pyrox-gp) solves over all N points at once, [`pyrox-lgm`](../pyrox-lgm) solves one sparse precision, and pyrox-ssm recurses in time.
This package imports nothing from pyrox-gp, pyrox-lgm or pyrox-nn.

## What's inside

Nothing yet: this is the installable, tested scaffold, and `pyrox_ssm.__all__` is empty.

## Install

pyrox-ssm is not on PyPI yet; install it from GitHub with uv:

```bash
uv add "pyrox-ssm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-ssm"
# normalizing-flow observation warps:
uv add "pyrox-ssm[flows] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-ssm"
```

Until kernellib pins gaussx v0.6.15, the project also needs the gaussx override described in the [root README](../../README.md#installation).
