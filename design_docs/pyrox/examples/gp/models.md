---
status: draft
version: 0.1.0
---

# GP Layer 2 -- Model Examples

Full GP workflows using `gp_sample`, `gp_factor`, and GPPrior. All model-level components live in `pyrox.gp`.

---

## Collapsed GP Regression (gp_factor)

### Gaussian likelihood -- f marginalized analytically

```python
import numpyro
import numpyro.distributions as dist
from pyrox.gp.kernels import RBF
from pyrox.gp.solvers import CholeskySolver
from pyrox.gp.models import GPPrior, gp_factor

def model(X, y):
    # Hyperparameter priors
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise = numpyro.sample("noise", dist.HalfNormal(1))

    # GP prior
    kernel = RBF(variance=variance, lengthscale=lengthscale)
    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X)

    # Collapsed: adds log p(y | X, theta) as a factor -- no guide needed
    gp_factor("gp", prior, y, noise_var=noise)

# Run MCMC over hyperparameters only
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
mcmc.run(key, X_train, y_train)
```

---

## Latent GP + Non-Gaussian Likelihood (gp_sample)

### Poisson counts -- f is a latent variable

```python
from pyrox.gp.models import gp_sample
from pyrox.gp.guides import WhitenedGuide

def model(X, y=None):
    kernel = RBF(variance=1.0, lengthscale=0.5)
    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X)
    guide = WhitenedGuide(prior=prior)

    # Latent GP -- requires a guide for SVI
    f = gp_sample("f", prior, guide=guide)

    # Non-Gaussian likelihood
    rate = jnp.exp(f)
    numpyro.sample("obs", dist.Poisson(rate), obs=y)

# Run SVI
svi = SVI(model, AutoNormal(model), optax.adam(1e-2), Trace_ELBO())
```

---

## Temporal GP via State-Space

### Matern kernel -> Kalman filter, O(NS^3)

```python
from pyrox.gp.kernels import Matern
from pyrox.gp.solvers import KalmanSolver
from pyrox.gp.models import MarkovGPPrior, gp_factor

def model(times, y):
    kernel = Matern(variance=1.0, lengthscale=1.0, nu=1.5)
    prior = MarkovGPPrior(kernel=kernel, solver=KalmanSolver(), times=times)
    noise = numpyro.sample("noise", dist.HalfNormal(1))
    gp_factor("gp", prior, y, noise_var=noise)
```

---

## Sparse Variational GP

### M << N inducing points, O(NM^2)

```python
from pyrox.gp.solvers import WoodburySolver
from pyrox.gp.guides import InducingPointGuide

def model(X, y=None):
    kernel = RBF(variance=1.0, lengthscale=0.5)
    prior = GPPrior(kernel=kernel, solver=WoodburySolver(), X=X)
    guide = InducingPointGuide(Z=inducing_locations, prior=prior)

    f = gp_sample("f", prior, guide=guide)
    numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)
```

---

*For primitive functions, see [gp_primitives.md](gp_primitives.md). For component protocols, see [gp_components.md](gp_components.md). For ecosystem integration, see [integration.md](integration.md).*
