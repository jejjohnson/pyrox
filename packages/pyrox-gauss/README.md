# pyrox-gauss

Gaussian NumPyro distributions for JAX whose covariance or precision is a structured [gaussx](https://github.com/jejjohnson/gaussx) operator: `MultivariateNormal`, `MultivariateNormalPrecision`, the linear-Gaussian state-space models `LGSSM`, `MaskedLGSSM` and `LGSSMFactory`, the Markov chain `MarkovGaussian`, and `GMRFDistribution`, which hands a gaussx GMRF field to `numpyro.sample`.

The classes moved here from gaussx (jejjohnson/pyrox#453) with their names, constructors and numerics unchanged, so moving to them is an import change. The GMRF fields themselves (`gaussx.GaussianMRF`, `IntrinsicGMRF`, `ConstrainedGMRF`, `BYM2GMRF`) stay in gaussx, because gaussx's Laplace and VB code and argminax use them as linear-algebra objects.

This package depends on gaussx and NumPyro only and imports nothing from `pyrox`.

## Install

pyrox-gauss is not on PyPI yet; install it from GitHub with uv:

```bash
uv add "pyrox-gauss @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gauss"
```

## Use it

```python
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
import pyrox_gauss as pg

cov = lx.MatrixLinearOperator(2.0 * jnp.eye(3), lx.positive_semidefinite_tag)
mvn = pg.MultivariateNormal(jnp.zeros(3), cov)


def model():
    return numpyro.sample("x", mvn)


x = numpyro.handlers.seed(model, 0)()
print(x.shape, mvn.log_prob(x).shape)  # (3,) ()
```
