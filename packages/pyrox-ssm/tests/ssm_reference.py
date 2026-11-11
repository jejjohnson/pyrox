"""Closed-form references and minimal stand-ins for pyrox-ssm's tests.

pyrox-ssm's tests import nothing from pyrox-gp or kernellib, so the dense
references (the Matérn kernel in closed form) and the non-Gaussian
likelihoods and the full-rank guide the Markov models are exercised with
are written out here, against the `pyrox.likelihoods` protocols.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from pyrox.likelihoods import Guide, Likelihood


def matern_kernel(X1, X2, variance, lengthscale, *, nu):
    r"""Closed-form Matérn kernel for $\nu \in \{1/2, 3/2, 5/2\}$.

    $$
    k_{1/2}(r) = \sigma^2 e^{-r/\ell}, \quad
    k_{3/2}(r) = \sigma^2 (1 + \sqrt{3} r/\ell) e^{-\sqrt{3} r/\ell}, \quad
    k_{5/2}(r) = \sigma^2 (1 + \sqrt{5} r/\ell + 5 r^2 / 3\ell^2) e^{-\sqrt{5} r/\ell}.
    $$
    """
    r = jnp.sqrt(jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    s = r / lengthscale
    if nu == 0.5:
        return variance * jnp.exp(-s)
    if nu == 1.5:
        return variance * (1.0 + jnp.sqrt(3.0) * s) * jnp.exp(-jnp.sqrt(3.0) * s)
    if nu == 2.5:
        poly = 1.0 + jnp.sqrt(5.0) * s + 5.0 * s**2 / 3.0
        return variance * poly * jnp.exp(-jnp.sqrt(5.0) * s)
    raise ValueError(f"nu must be 0.5, 1.5 or 2.5; got {nu}")


class BernoulliLikelihood(Likelihood):
    """Bernoulli with a logit link, as in pyrox-gp."""

    def log_prob(self, f, y, X=None):
        return dist.Bernoulli(logits=f).log_prob(y).sum()


class PoissonLikelihood(Likelihood):
    """Poisson with a log link, as in pyrox-gp."""

    def log_prob(self, f, y, X=None):
        return dist.Poisson(rate=jnp.exp(f)).log_prob(y).sum()


class StudentTLikelihood(Likelihood):
    """Student-t centred on the latent, as in pyrox-gp."""

    df: float
    scale: float

    def log_prob(self, f, y, X=None):
        return dist.StudentT(df=self.df, loc=f, scale=self.scale).log_prob(y).sum()


class MultiLatentLikelihood(Likelihood):
    """A likelihood that declares several latents per point (never evaluated)."""

    latent_dim: int = eqx.field(static=True)

    def log_prob(self, f, y, X=None):
        raise NotImplementedError


class FullRankGuide(Guide):
    r"""$q(u) = N(m, L L^\top)$ over inducing values, with dense predictions."""

    mean: jax.Array
    scale_tril: jax.Array

    @classmethod
    def init(cls, num_inducing: int, *, scale: float = 1.0) -> FullRankGuide:
        return cls(jnp.zeros(num_inducing), scale * jnp.eye(num_inducing))

    def sample(self, key):
        eps = jax.random.normal(key, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.scale_tril @ eps

    def log_prob(self, u):
        cov = self.scale_tril @ self.scale_tril.T
        return dist.MultivariateNormal(self.mean, cov).log_prob(u)

    def kl_divergence(self, prior_cov: lx.AbstractLinearOperator) -> jax.Array:
        cov = lx.MatrixLinearOperator(
            self.scale_tril @ self.scale_tril.T, lx.positive_semidefinite_tag
        )
        return gaussx.gaussian_kl(self.mean, cov, jnp.zeros_like(self.mean), prior_cov)

    def predict(self, K_xz, K_zz_op, K_xx_diag):
        # μ = K_xz K_zz⁻¹ m,  σ² = k_xx − diag(K_xz K_zz⁻¹ K_zx) + diag(A S Aᵀ)
        A = jnp.linalg.solve(K_zz_op.as_matrix(), K_xz.T).T
        S = self.scale_tril @ self.scale_tril.T
        var = K_xx_diag - jnp.sum(A * K_xz, axis=1) + jnp.sum((A @ S) * A, axis=1)
        return A @ self.mean, var


def site_expectation(integrator, fn, mean, var, y):
    r"""$\mathbb{E}_{f_n \sim N(m_n, v_n)}[\text{fn}(f_n, y_n)]$ at every site."""

    def one(m, v, d):
        cov = lx.MatrixLinearOperator(v[None, None], lx.positive_semidefinite_tag)
        state = gaussx.GaussianState(mean=m[None], cov=cov)
        out = gaussx.mean_expectation(
            lambda f: jnp.atleast_1d(fn(f[0], d)), state, integrator
        )
        return out[0]

    return jax.vmap(one)(mean, var, y)
