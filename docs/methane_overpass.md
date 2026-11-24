# Map methane from one overpass

A satellite overpass leaves holes wherever there was a cloud, and every pixel it does return is noisy. Here we map XCH₄ (the column-averaged methane) over a basin from the cloud-free pixels of one TROPOMI overpass, with an uncertainty for every cell, the cloudy ones included. The overpass is simulated, so that we know the truth, and a real L2 product's pixels drop in for its three arrays.

The state $\mathbf{x} \in \mathbb{R}^{N}$ is XCH₄ in ppb on a $20 \times 20$ grid over the Permian Basin, so $N = 400$ cells. The overpass leaves $M$ of them cloud-free, and $\mathbf{y} \in \mathbb{R}^{M}$ holds their values. The background is $x_b = 1880$ ppb. Cell centres are positions $\mathbf{s}_n \in \mathbb{R}^{3}$ in km on a sphere of radius $R = 6371$ km, so $\Vert \mathbf{s}_n - \mathbf{s}_{n'} \Vert$ is a chordal distance in km. They are stacked in $\mathbf{S} \in \mathbb{R}^{N \times 3}$, and the cloud-free ones in $\mathbf{S}_o \in \mathbb{R}^{M \times 3}$. The stand-in truth is a 40 ppb plume on the background, the pixel noise has a standard deviation of 8 ppb, and a pixel is cloud-free with probability 0.6.

```python
import geonnax as gnx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pyrox_gp as pgp
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam


jax.config.update("jax_enable_x64", True)

# Shapes: N = 400 grid cells; M cloud-free cells; S = 500 draws

# (λ_n, φ_n), the cell centres in degrees
# → (N, 2)
lon, lat = jnp.meshgrid(jnp.linspace(-104.2, -103.4, 20), jnp.linspace(31.6, 32.4, 20))
lonlat = jnp.stack([lon.ravel(), lat.ravel()], axis=-1)
# today (geonnax)
# s_n = R (cos φ_n cos λ_n, cos φ_n sin λ_n, sin φ_n), in km
# (N, 2) → (N, 3)
cells = 6371.0 * gnx.geo.lonlat_to_cartesian3d(lonlat, input_unit="degrees")
# x_n = x_b + 40 exp(−½ ‖(λ_n, φ_n) − (λ_0, φ_0)‖² / 0.08²), in ppb
# → (N,)
background = 1880.0
sq_dist = jnp.sum((lonlat - jnp.array([-103.8, 32.0])) ** 2, axis=-1)
truth = background + 40.0 * jnp.exp(-0.5 * sq_dist / 0.08**2)
# which cells are cloud-free
# → (N,)
mask_key, noise_key = jr.split(jr.key(0))
mask = jr.bernoulli(mask_key, 0.6, (400,))
# y_m = x_m + ε_m, with ε_m ~ 𝒩(0, 8²), at the cloud-free cells
# (N,) → (M,)
obs = (truth + 8.0 * jr.normal(noise_key, (400,)))[mask]
# (N, 3) → (M, 3)
obs_cells = cells[mask]
```

## Stage 1: the covariance, by its evidence

We choose the field's correlation length and variance, and the pixel noise, by asking which values make the observed pixels most probable once the unknown field is integrated out. The anomaly $x - x_b$ is a Gaussian process with a Matérn-3/2 kernel $k_{\boldsymbol{\theta}}$, and each observed pixel adds independent noise of standard deviation $\sigma_\varepsilon$. The parameters are $\boldsymbol{\theta} = (\ell, \sigma_f^2, \sigma_\varepsilon)$, where $\ell$ is the lengthscale in km and $\sigma_f^2$ the variance in ppb². Integrating out the field gives the evidence

```{math}
\log p(\mathbf{y} \mid \boldsymbol{\theta}) = \log \mathcal{N}(\mathbf{y} \mid x_b \mathbf{1}, \mathbf{K}_{oo} + \sigma_\varepsilon^2 \mathbf{I})
```

where $\mathbf{K}_{oo} = k_{\boldsymbol{\theta}}(\mathbf{S}_o, \mathbf{S}_o) \in \mathbb{R}^{M \times M}$ is the kernel matrix of the observed centres and $\mathbf{1} \in \mathbb{R}^{M}$ is a vector of ones. The target is the posterior $p(\boldsymbol{\theta} \mid \mathbf{y}) \propto p(\mathbf{y} \mid \boldsymbol{\theta}) \ p(\boldsymbol{\theta})$, sampled by NUTS, or a mean-field approximation $q(\boldsymbol{\theta})$ fitted by SVI from the same model.

```python
# Shapes: M cloud-free cells; S = 500 draws


# user
# p(y, θ) = p(y | θ) p(θ), with the field integrated out
def evidence(obs_cells: Float[Array, "M 3"], obs: Float[Array, " M"]) -> None:
    # today (pyrox-gp)
    # k_θ(r) = σ_f² (1 + √3 r / ℓ) exp(−√3 r / ℓ), r in km
    kernel = pgp.Matern(nu=1.5)
    # ℓ ~ LogNormal(3, 0.5²), median 20 km
    kernel.set_prior("lengthscale", dist.LogNormal(3.0, 0.5))
    # σ_f² ~ LogNormal(5, 1), median 150 ppb²
    kernel.set_prior("variance", dist.LogNormal(5.0, 1.0))
    # σ_ε ~ LogNormal(2.3, 0.5²), median 10 ppb
    noise_std = numpyro.sample("noise_std", dist.LogNormal(2.3, 0.5))
    # today (pyrox-gp)
    # log p(y | θ) = log 𝒩(y − x_b 1; 0, K_oo + σ_ε² I)
    prior = pgp.GPPrior(kernel=kernel, X=obs_cells)
    pgp.gp_factor("y", prior, obs - background, noise_std**2)


# θ ~ p(θ | y), by NUTS
# → (S,) per site
nuts = NUTS(evidence, init_strategy=init_to_median)
mcmc = MCMC(nuts, num_warmup=500, num_samples=500, progress_bar=False)
mcmc.run(jr.key(1), obs_cells, obs)
draws = mcmc.get_samples()
# q(θ) ≈ p(θ | y), by SVI with a mean-field guide on the same model
svi = SVI(evidence, AutoNormal(evidence), Adam(1e-2), Trace_ELBO())
svi_result = svi.run(jr.key(2), 2000, obs_cells, obs, progress_bar=False)

low, mid, high = jnp.quantile(draws["noise_std"], jnp.array([0.05, 0.5, 0.95]))
lengthscale = jnp.median(draws["Matern.lengthscale"])
print(f"M {obs.size}  noise {mid:.1f} [{low:.1f}, {high:.1f}]  ell {lengthscale:.0f}")
```

This prints `M 227  noise 8.0 [7.3, 8.7]  ell 18`: the number of cloud-free cells, the posterior median of $\sigma_\varepsilon$ with its 90 % interval (the truth is 8 ppb), and the posterior median of $\ell$ in km. `set_prior` turns each hyperparameter into a NumPyro site named `Matern.<name>`, so one model function serves both NUTS and SVI. Starting at the prior medians, with a prior on $\sigma_\varepsilon$ that stays away from zero, keeps the warm-up away from a near-singular $\mathbf{K}_{oo} + \sigma_\varepsilon^2 \mathbf{I}$.

## Stage 2: the map and its uncertainty

With $\boldsymbol{\theta}$ fixed at its posterior median $\hat{\boldsymbol{\theta}}$, the map of all $N$ cells is a closed-form conditional. With $\mathbf{K}_{So} = k_{\hat{\boldsymbol{\theta}}}(\mathbf{S}, \mathbf{S}_o) \in \mathbb{R}^{N \times M}$, and $\mathbf{k}_{no} \in \mathbb{R}^{M}$ its $n$-th row, the posterior mean $\boldsymbol{\mu} \in \mathbb{R}^{N}$ and the variance $\sigma_n^2$ of cell $n$ are

```{math}
\boldsymbol{\mu} = x_b \mathbf{1} + \mathbf{K}_{So} (\mathbf{K}_{oo} + \hat{\sigma}_\varepsilon^2 \mathbf{I})^{-1} (\mathbf{y} - x_b \mathbf{1})
```

```{math}
\sigma_n^2 = k_{\hat{\boldsymbol{\theta}}}(\mathbf{s}_n, \mathbf{s}_n) - \mathbf{k}_{no}^\top (\mathbf{K}_{oo} + \hat{\sigma}_\varepsilon^2 \mathbf{I})^{-1} \mathbf{k}_{no}
```

```python
# Shapes: N = 400 grid cells; M cloud-free cells

# θ̂, the posterior median of each site
# → ()
variance = float(jnp.median(draws["Matern.variance"]))
noise_std = float(jnp.median(draws["noise_std"]))
# today (pyrox-gp)
# k_θ̂
kernel = pgp.Matern(nu=1.5, init_lengthscale=float(lengthscale), init_variance=variance)
# today (pyrox-gp)
# μ − x_b 1 and σ_n², at every cell
# (N, 3) → (N,), (N,)
prior = pgp.GPPrior(kernel=kernel, X=obs_cells)
posterior = prior.condition(obs - background, jnp.array(noise_std**2))
anomaly, var = posterior.predict(cells)
mean = background + anomaly
std = jnp.sqrt(var)

# RMSE of the map and of the raw pixels, and the cells within ±2σ of the truth
map_rmse = jnp.sqrt(jnp.mean((mean - truth) ** 2))
pixel_rmse = jnp.sqrt(jnp.mean((obs - truth[mask]) ** 2))
covered = jnp.mean(jnp.abs(mean - truth) <= 2.0 * std)
print(
    f"RMSE map {map_rmse:.1f}  pixels {pixel_rmse:.1f}  covered {100 * covered:.1f} %"
)
print(f"peak map {mean.max():.0f}  truth {truth.max():.0f}")
```

This prints

```
RMSE map 3.4  pixels 8.2  covered 95.5 %
peak map 1905  truth 1917
```

The map more than halves the error of the raw pixels, although it also fills the cloudy cells, and 95.5 % of the cells lie within two standard deviations of the truth. The peak of the plume comes out low, because a kernel this smooth flattens a narrow peak. To carry the uncertainty of $\boldsymbol{\theta}$ into the map, condition on each NUTS draw instead of on the median and pool the results.

## The same model, three ways

Stage 1 used a ready-made `Parameterized` kernel. The [three ways to write a model](notebooks/regression_masterclass_treeat.ipynb) all give plain NumPyro sites, so each can write the same evidence, from the lightest machinery to the heaviest.

**A. Plain sites and `eqx.tree_at`.** When a field of an existing Equinox module becomes random, no pyrox class is needed. Sample the value in the model and splice it into the module.

```python
import equinox as eqx
import kernellib as kl


# Shapes: M cloud-free cells


# user
# p(y, θ), with θ spliced into a kernellib kernel
def evidence_a(obs_cells: Float[Array, "M 3"], obs: Float[Array, " M"]) -> None:
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(3.0, 0.5))
    variance = numpyro.sample("variance", dist.LogNormal(5.0, 1.0))
    # today (kernellib)
    # k_θ with the sampled ℓ and σ_f²
    kernel = eqx.tree_at(
        lambda k: (k.lengthscale, k.variance),
        kl.Matern(nu=1.5),
        (lengthscale, variance),
    )
    noise_std = numpyro.sample("noise_std", dist.LogNormal(2.3, 0.5))
    # today (pyrox-gp)
    # log p(y | θ)
    prior = pgp.GPPrior(kernel=kernel, X=obs_cells)
    pgp.gp_factor("y", prior, obs - background, noise_std**2)
```

**B. A `PyroxModule` owns its sites.** The module samples its own parameters, under names `<pyrox_name>.<site>` that are cached per call and stable across `jit`, `eqx.tree_at` and checkpoints. Two instances of one class in the same model need distinct `pyrox_name`s, otherwise the trace rejects the duplicate sites.

```python
import kernellib.functional as klf
from pyrox._core import PyroxModule, pyrox_method


# Shapes: M cloud-free cells


# user
# k_θ, which samples its own θ when it is called
class MaternSites(pgp.Kernel, PyroxModule):
    pyrox_name: str = "MaternSites"

    @pyrox_method
    def __call__(
        self, X1: Float[Array, "N1 3"], X2: Float[Array, "N2 3"]
    ) -> Float[Array, "N1 N2"]:
        # today (pyrox)
        lengthscale = self.pyrox_sample("lengthscale", dist.LogNormal(3.0, 0.5))
        variance = self.pyrox_sample("variance", dist.LogNormal(5.0, 1.0))
        # today (kernellib)
        # k_θ(r) = σ_f² (1 + √3 r / ℓ) exp(−√3 r / ℓ)
        return klf.matern_kernel(X1, X2, variance, lengthscale, nu=1.5)


# user
# p(y, θ), with θ inside the kernel
def evidence_b(obs_cells: Float[Array, "M 3"], obs: Float[Array, " M"]) -> None:
    noise_std = numpyro.sample("noise_std", dist.LogNormal(2.3, 0.5))
    # today (pyrox-gp)
    # log p(y | θ)
    prior = pgp.GPPrior(kernel=MaternSites(), X=obs_cells)
    pgp.gp_factor("y", prior, obs - background, noise_std**2)
```

**C. `Parameterized`, for constraints, priors and guides.** A module with constrained hyperparameters declares them once in `setup()`: `register_param` puts each value on its support, `set_prior` attaches $p(\boldsymbol{\theta})$ and `autoguide` attaches a guide $q(\boldsymbol{\theta})$ per parameter. `set_mode` then decides whether `get_param` samples the prior or the guide, so the module is its own SVI guide and `__call__` never changes. `pgp.Matern` above is a ready-made one, and the block below writes the same kernel out.

```python
from pyrox._core import Parameterized


# Shapes: M cloud-free cells


# user
# k_θ, with supports, priors and guides declared once
class MaternPriors(Parameterized, pgp.Kernel):
    pyrox_name: str = "MaternPriors"

    def setup(self) -> None:
        # today (pyrox)
        # ℓ > 0 and σ_f² > 0
        positive = dist.constraints.positive
        self.register_param("lengthscale", jnp.array(20.0), constraint=positive)
        self.register_param("variance", jnp.array(150.0), constraint=positive)
        # p(θ)
        self.set_prior("lengthscale", dist.LogNormal(3.0, 0.5))
        self.set_prior("variance", dist.LogNormal(5.0, 1.0))
        # q(θ), a normal on each unconstrained value, mapped back to ℓ, σ_f² > 0
        self.autoguide("lengthscale", "normal")
        self.autoguide("variance", "normal")

    @pyrox_method
    def __call__(
        self, X1: Float[Array, "N1 3"], X2: Float[Array, "N2 3"]
    ) -> Float[Array, "N1 N2"]:
        # today (pyrox)
        # θ ~ p(θ) in model mode, θ ~ q(θ) in guide mode
        lengthscale = self.get_param("lengthscale")
        variance = self.get_param("variance")
        # today (kernellib)
        # k_θ(r) = σ_f² (1 + √3 r / ℓ) exp(−√3 r / ℓ)
        return klf.matern_kernel(X1, X2, variance, lengthscale, nu=1.5)


kernel = MaternPriors()


# user
# p(y, θ), with the kernel in model mode
def evidence_c(obs_cells: Float[Array, "M 3"], obs: Float[Array, " M"]) -> None:
    kernel.set_mode("model")
    noise_std = numpyro.sample("noise_std", dist.LogNormal(2.3, 0.5))
    # today (pyrox-gp)
    # log p(y | θ)
    prior = pgp.GPPrior(kernel=kernel, X=obs_cells)
    pgp.gp_factor("y", prior, obs - background, noise_std**2)


# user
# q(θ), the kernel's own guide, and a log-normal for σ_ε, which is not the kernel's
def guide_c(obs_cells: Float[Array, "M 3"], obs: Float[Array, " M"]) -> None:
    kernel.set_mode("guide")
    # one call registers q(ℓ) and q(σ_f²)
    kernel(obs_cells[:1], obs_cells[:1])
    loc = numpyro.param("noise_loc", 2.3)
    positive = dist.constraints.positive
    scale = numpyro.param("noise_scale", 0.1, constraint=positive)
    numpyro.sample("noise_std", dist.LogNormal(loc, scale))


# ℒ_ELBO, the mean of the last 100 of 2000 SVI steps, for each way
for name, model, guide in (
    ("A", evidence_a, AutoNormal(evidence_a)),
    ("B", evidence_b, AutoNormal(evidence_b)),
    ("C", evidence_c, guide_c),
):
    svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())
    result = svi.run(jr.key(2), 2000, obs_cells, obs, progress_bar=False)
    print(f"{name}  {result.losses[-100:].mean():.1f}")
```

This prints

```
A  826.6
B  827.2
C  826.6
```

that is, $\mathcal{L}_\text{ELBO}$ (the negative ELBO, so lower is better) at the end of each fit. The three agree to within 0.6, since they write the same joint density. Which to use is a question of what the model already has: A for an existing Equinox module, B for a module that is itself probabilistic, and C when the parameters have constraints and guides of their own.
