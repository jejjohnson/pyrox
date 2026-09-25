---
status: draft
version: 0.2.0
---

# Layer 0 — Primitives

Pure JAX functions. Stateless, differentiable, composable. No classes, no NumPyro dependency, no xarray — just `(Array, ...) → Array`.

All functions are compatible with `jax.jit`, `jax.grad`, and `jax.vmap`. They implement the mathematical core of extreme value theory; the NumPyro `Distribution` wrappers at Layer 1 compose these into the standard probabilistic interface.

---

## GEV Functions (`xtremax._src.gev`)

The Generalized Extreme Value distribution unifies three classical types under one parameterization.

**CDF:**

$$F(x; \mu, \sigma, \xi) = \exp\!\left\{-\left[1 + \xi\frac{x - \mu}{\sigma}\right]^{-1/\xi}\right\}$$

with the convention $\xi = 0 \Rightarrow F(x) = \exp\{-\exp(-(x-\mu)/\sigma)\}$ (Gumbel limit).

**Support:** $\{x : 1 + \xi(x-\mu)/\sigma > 0\}$.

### `gev_log_prob(x, loc, scale, shape)`

**Mathematical definition:**

$$\log f(x; \mu, \sigma, \xi) = -\log\sigma - (1 + 1/\xi)\log t - t^{-1/\xi}$$

where $t = 1 + \xi(x - \mu)/\sigma$. For $\xi = 0$: $\log f = -\log\sigma - s - e^{-s}$ where $s = (x-\mu)/\sigma$.

**Numerical method:** Uses `jnp.where` to branch between $\xi \neq 0$ and $\xi \approx 0$ (Gumbel) to avoid division by zero. Gradient-safe via JAX autodiff.

**Complexity:** $O(N)$ element-wise.

```python
def gev_log_prob(
    x: Float[Array, "..."],
    loc: Float[Array, "..."],
    scale: Float[Array, "..."],
    shape: Float[Array, "..."],
) -> Float[Array, "..."]:
```

### `gev_cdf(x, loc, scale, shape)`

$$F(x) = \exp\!\left\{-t^{-1/\xi}\right\}, \quad t = 1 + \xi(x - \mu)/\sigma$$

```python
def gev_cdf(x, loc, scale, shape) -> Array:
```

### `gev_icdf(q, loc, scale, shape)`

**Quantile function (inverse CDF):**

$$F^{-1}(q) = \mu + \frac{\sigma}{\xi}\left[(-\log q)^{-\xi} - 1\right]$$

For $\xi = 0$: $F^{-1}(q) = \mu - \sigma \log(-\log q)$.

```python
def gev_icdf(q, loc, scale, shape) -> Array:
```

### `gev_return_level(period, loc, scale, shape)`

**$T$-year return level:**

$$z_T = F^{-1}(1 - 1/T) = \mu + \frac{\sigma}{\xi}\left[(-\log(1 - 1/T))^{-\xi} - 1\right]$$

This is the level expected to be exceeded on average once every $T$ years.

```python
def gev_return_level(period, loc, scale, shape) -> Array:
```

### `gev_mean(loc, scale, shape)`

$$\mathbb{E}[X] = \mu + \sigma \frac{\Gamma(1 - \xi) - 1}{\xi}, \quad \xi < 1$$

Undefined for $\xi \geq 1$ (infinite mean — heavy tail).

### `gev_variance(loc, scale, shape)`

$$\operatorname{Var}(X) = \frac{\sigma^2}{\xi^2}\left[\Gamma(1 - 2\xi) - \Gamma^2(1 - \xi)\right], \quad \xi < 1/2$$

---

## GPD Functions (`xtremax._src.gpd`)

The Generalized Pareto Distribution models threshold exceedances $Y = X - u \mid X > u$.

**CDF:**

$$G(y; \sigma, \xi) = 1 - \left(1 + \xi \frac{y}{\sigma}\right)^{-1/\xi}, \quad y > 0$$

For $\xi = 0$: $G(y) = 1 - \exp(-y/\sigma)$ (exponential).

### `gpd_log_prob(y, scale, shape)`

$$\log g(y; \sigma, \xi) = -\log\sigma - (1 + 1/\xi)\log\left(1 + \xi y/\sigma\right)$$

```python
def gpd_log_prob(y, scale, shape) -> Array:
```

### `gpd_cdf(y, scale, shape)`

### `gpd_survival(y, scale, shape)`

**Survival function:** $\bar{G}(y) = (1 + \xi y/\sigma)^{-1/\xi}$

### `gpd_return_level(period, scale, shape, rate)`

**$T$-year return level for POT:**

$$z_T = u + \frac{\sigma}{\xi}\left[(\lambda T)^{\xi} - 1\right]$$

where $u$ is the threshold and $\lambda$ is the exceedance rate (expected exceedances per year).

```python
def gpd_return_level(period, scale, shape, rate) -> Array:
```

### `gpd_mean_excess(scale, shape)`

**Mean excess function:** $\mathbb{E}[Y - u \mid Y > u] = \frac{\sigma + \xi u}{1 - \xi}$ for $\xi < 1$.

---

## Point Process Functions (`xtremax._src.point_process`)

### `poisson_log_likelihood(event_times, rate, T)`

**Homogeneous Poisson process log-likelihood:**

$$\log L = \sum_{i=1}^n \log \lambda - \lambda T$$

where $\lambda$ is the rate and $T$ is the observation window.

### `inhomogeneous_poisson_log_likelihood(event_times, log_intensity_fn, T)`

**Inhomogeneous Poisson process:**

$$\log L = \sum_{i=1}^n \log \lambda(t_i) - \int_0^T \lambda(t)\,dt$$

The integral is approximated numerically (trapezoidal or quadrature).

### `hawkes_log_likelihood(event_times, background_rate, alpha, beta)`

**Hawkes self-exciting process:**

$$\lambda(t) = \mu + \alpha \sum_{t_i < t} \beta e^{-\beta(t - t_i)}$$

$$\log L = \sum_{i=1}^n \log \lambda(t_i) - \mu T - \frac{\alpha}{\beta}\sum_{i=1}^n \left(1 - e^{-\beta(T - t_i)}\right)$$

### `extremal_index(exceedances, threshold, run_length)`

**Extremal index** $\theta \in (0, 1]$ — measures clustering of threshold exceedances. $\theta = 1$ means no clustering (independent exceedances). Estimated via the runs method:

$$\hat{\theta} = \frac{\text{number of clusters}}{\text{number of exceedances}}$$

---

## Max-Stable Functions (`xtremax._src.max_stable`)

### `brown_resnick_extremal_coeff(h, variogram_fn)`

**Brown-Resnick extremal coefficient:**

$$\theta(h) = 2\Phi\!\left(\sqrt{\gamma(h)/2}\right)$$

where $\gamma(h)$ is the semivariogram and $\Phi$ is the standard normal CDF.

### `smith_extremal_coeff(h, cov_matrix)`

$$\theta(h) = 2\Phi\!\left(\sqrt{h^T \Sigma^{-1} h / 2}\right)$$

### `schlather_extremal_coeff(h, correlation_fn)`

$$\theta(h) = 1 + \sqrt{(1 - \rho(h))/2}$$

### `extremal_t_extremal_coeff(h, correlation_fn, df)`

$$\theta(h) = 2\,T_{\nu+1}\!\left(\sqrt{(\nu+1)(1 - \rho(h))/(1 + \rho(h))}\right)$$

### `pairwise_log_likelihood(z, extremal_coeff_fn, coordinates)`

**Composite pairwise log-likelihood** for max-stable processes (Padoan et al. 2010):

$$\ell_c = \sum_{i < j} \log f_{ij}(z_i, z_j)$$

where $f_{ij}$ is the bivariate density derived from the bivariate CDF:

$$F_{ij}(z_i, z_j) = \exp\!\left(-\frac{1}{z_i}A\!\left(\frac{\log z_j / z_i}{\log z_j / z_i + \log z_i / z_j}\right) - \frac{1}{z_j}A\!\left(\frac{\log z_i / z_j}{\log z_j / z_i + \log z_i / z_j}\right)\right)$$

and $A$ is the Pickands dependence function determined by $\theta$.

### `madogram(data, coordinates, n_bins)`

**F-madogram** for nonparametric extremal coefficient estimation:

$$\hat{\nu}(h) = \frac{1}{2}\operatorname{mean}\left|F_n(Z(s_1)) - F_n(Z(s_2))\right|, \quad \|s_1 - s_2\| \approx h$$

$$\hat{\theta}(h) = \frac{1 + 2\hat{\nu}(h)}{1 - 2\hat{\nu}(h)}$$

---

## Variogram Functions (`xtremax._src.variogram`)

Used by max-stable models and spatial GP layers.

### `power_variogram(h, range, smoothness)`

$$\gamma(h) = \left(\|h\| / \lambda\right)^\alpha, \quad \alpha \in (0, 2]$$

### `matern_variogram(h, range, smoothness, sill)`

$$\gamma(h) = \sigma^2 \left(1 - \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\|h\|}{\lambda}\right)^\nu K_\nu\!\left(\frac{\|h\|}{\lambda}\right)\right)$$

where $K_\nu$ is the modified Bessel function of the second kind.
