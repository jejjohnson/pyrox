# One trend in the extremes, four priors

The README's quickstart puts a Gaussian process on the location of a Gumbel distribution. Here we write the same trend in four ways and compare them on one problem.

The data are $N = 60$ annual maxima $y_n$ at times $t_n$ in decades, drawn from a Gumbel distribution whose location $\mu_n$ drifts and oscillates. We want the 100-year return level $z_n$, the value that year $n$ exceeds with probability $1/100$. The model is $y_n \sim \mathrm{Gumbel}(\mu_0 + f_n, \sigma)$, where $\mu_0$ is a constant, $\sigma > 0$ the scale and $F_n$ the cumulative distribution function in year $n$, so that $z_n = F_n^{-1}(1 - 1/100)$. The trend $\mathbf{f} \in \mathbb{R}^{N}$ comes from four packages in turn:

- pyrox-gp gives it a Matérn-3/2 kernel $k_{\boldsymbol{\theta}}$ with variance $\sigma_f^2$ and lengthscale $\ell$, and so the kernel matrix $\mathbf{K}_{\boldsymbol{\theta}} \in \mathbb{R}^{N \times N}$;
- pyrox-ssm writes the same kernel as a recursion in time;
- pyrox-lgm makes it a random walk with precision $\tau$, scaled by the constant $c$ that brings its typical variance to one at $\tau = 1$, so $\mathbf{Q}_\tau \in \mathbb{R}^{N \times N}$ is tridiagonal apart from a constraint that fixes the mean of the walk;
- pyrox-nn makes it $\boldsymbol{\Phi} \mathbf{w}$, with $M = 20$ sines in the columns of $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$ and weights $\mathbf{w} \in \mathbb{R}^{M}$ whose variances, the diagonal of $\mathbf{S}_{\boldsymbol{\theta}} \in \mathbb{R}^{M \times M}$, are the spectral density of the same kernel at the frequencies of the sines.

A fit with no trend is there to compare against.

```python
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS, init_to_median
from pyrox_gp import GPPrior, Matern, gp_sample
from pyrox_lgm import RW1
from pyrox_nn import HSGPFeatures
from pyrox_ssm import MarkovGPPrior, MaternSDE, markov_gp_innovations_sample
from pyrox_xtreme import GumbelType1GEVD


jax.config.update("jax_enable_x64", True)

# Shapes: N = 60 years; M = 20 basis functions; S = 500 posterior draws

# t_n, the years in decades
# → (N,), (N,)
years = jnp.arange(60)
times = years / 10.0
# μ_n = 30 + 1.5 sin(2π t_n / 8) + t_n / 4
# → (N,)
true_loc = 30.0 + 1.5 * jnp.sin(2.0 * jnp.pi * times / 8.0) + times / 4.0
# today (pyrox-xtreme)
# y_n ~ Gumbel(μ_n, 1)
# → (N,)
truth = GumbelType1GEVD(true_loc, 1.0)
maxima = truth.sample(jr.key(0))

# today (pyrox-gp)
# k_θ, Matérn-3/2 with σ_f², ℓ ~ LogNormal(0, 0.5²)
hyperprior = dist.LogNormal(0.0, 0.5)
kernel = Matern(nu=1.5)
kernel.set_prior("variance", hyperprior)
kernel.set_prior("lengthscale", hyperprior)
# today (pyrox-nn)
# φ_1, …, φ_M, sines on [−4.5, 4.5] decades, scaled by the spectral density of k_θ
features = HSGPFeatures.init(1, 20, 4.5, kernel=kernel)
# today (pyrox-lgm)
# f_n − f_{n−1} ~ 𝒩(0, 1 / (c τ)), with τ under its default PC prior
walk = RW1(60, name="walk")


# user
# f ~ 𝒩(0, K_θ), with the dense Gram matrix K_θ
def covariance() -> Float[Array, " N"]:
    # today (pyrox-gp)
    return gp_sample("f", GPPrior(kernel, times[:, None]), whitened=True)


# user
# the same prior k_θ, as a recursion in time
def state_space() -> Float[Array, " N"]:
    variance = numpyro.sample("variance", hyperprior)
    lengthscale = numpyro.sample("lengthscale", hyperprior)
    # today (pyrox-ssm)
    sde = MaternSDE(variance, lengthscale, order=1)
    return markov_gp_innovations_sample("f", MarkovGPPrior(sde, times))


# user
# f ~ 𝒩(0, Q_τ⁻¹), with the sparse precision Q_τ
def precision() -> Float[Array, " N"]:
    # today (pyrox-lgm)
    return walk.sample(years)


# user
# f = Φ w, with w ~ 𝒩(0, S_θ)
def basis() -> Float[Array, " N"]:
    # today (pyrox-nn)
    return features(times[:, None] - times.mean())


# user
# f = 0, the stationary fit
def no_trend() -> Float[Array, " N"]:
    return jnp.zeros(60)


# user
# μ_0 ~ 𝒩(30, 5²), σ ~ LogNormal(0, 0.5²), y_n ~ Gumbel(μ_0 + f_n, σ)
def model(trend: Callable[[], Float[Array, " N"]], maxima: Float[Array, " N"]) -> None:
    loc = numpyro.sample("loc", dist.Normal(30.0, 5.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 0.5))
    # today (pyrox-xtreme)
    # z_n = F_n⁻¹(1 − 1/100), the 100-year return level in year n
    gumbel = GumbelType1GEVD(loc + trend(), scale)
    numpyro.deterministic("z100", gumbel.return_level(100.0))
    numpyro.sample("obs", gumbel.to_event(1), obs=maxima)


# z_n of the truth
# → (N,)
true_z100 = truth.return_level(100.0)
for trend in (covariance, state_space, precision, basis, no_trend):
    # p(θ, f | y), by NUTS
    nuts = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(nuts, num_warmup=300, num_samples=500, progress_bar=False)
    mcmc.run(jr.key(1), trend, maxima)
    # 𝔼[z_n | y], and its RMSE over the N years
    # (S, N) → (N,)
    z100 = mcmc.get_samples()["z100"].mean(axis=0)
    rmse = jnp.sqrt(jnp.mean((z100 - true_z100) ** 2))
    print(f"{trend.__name__:>11}  RMSE {rmse:.2f}  last year {z100[-1]:.1f}")
print(f"{'truth':>11}  RMSE 0.00  last year {true_z100[-1]:.1f}")
```

This prints

```
 covariance  RMSE 0.50  last year 34.0
state_space  RMSE 0.51  last year 34.0
  precision  RMSE 0.52  last year 34.0
      basis  RMSE 0.47  last year 34.1
   no_trend  RMSE 1.17  last year 36.6
      truth  RMSE 0.00  last year 34.6
```

that is, the root-mean-square error of the posterior mean of $z_n$ over the sixty years, and its value in the last one. The four priors agree to within 0.05, and they cut the error of the stationary fit (1.17) to less than half. The stationary fit is two units too high in the last year, because it keeps one level for sixty years and explains the swings with a wider scale (a posterior mean of 1.27, where the truth is 1). The covariance and the recursion are the same prior, and the basis approximates it, which is why they land together. The hyperparameters never appear in `model`, since the kernel and the layer declare their own sites. The whole block takes about a minute and a half on a shared 16-core CPU, much of it compilation.

To use another prior, write one more function that returns $\mathbf{f}$; to use another likelihood, change the last two lines of `model` and keep the priors. The [methane map](methane_overpass.md) takes one GP from its evidence to a map with uncertainties, and the [spatial extremes tutorials](https://jejjohnson.github.io/pyrox/spatial-extremes/) put trends like this one on station data.
