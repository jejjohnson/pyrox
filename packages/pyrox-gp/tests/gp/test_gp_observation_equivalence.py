"""pyrox-gp's observation models against the implementations they replaced.

The six likelihoods now build gaussx likelihoods (gh-375). Each ``_Old*``
class below is the previous ``log_prob`` (numpyro, summed over points) copied
verbatim, so it is both the reference formula and, as a plain `Likelihood`,
a driver of the inference strategies' protocol path, which is the code the
strategies ran before. Every comparison is at ``rtol=1e-10`` under x64: the
two sides evaluate the same density through different but exact formulas, so
they agree to float64 round-off (about 1e-15 for log densities and 1e-13 for
site derivatives on these inputs).
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as nd
import pytest
from pyrox.likelihoods import (
    AbstractObservation,
    GaussianLikelihood as CoreGaussianLikelihood,
)
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    ConjugateVI,
    ExpectationPropagation,
    FullRankGuide,
    GaussianLikelihood,
    GaussNewtonInference,
    GPPrior,
    HeteroscedasticGaussianLikelihood,
    LaplaceInference,
    PoissonLikelihood,
    PosteriorLinearization,
    SoftmaxLikelihood,
    SparseGPPrior,
    StudentTLikelihood,
    svgp_elbo,
)
from pyrox_gp._inference import _ell_numerical
from pyrox_gp._inference_nongauss import (
    _log_prob_per_point_factory,
    _per_point_grad_hess,
    _site_data,
)
from pyrox_gp._protocols import Likelihood


RTOL = 1e-10
N = 40


# --- the old implementations, copied -----------------------------------------


class _OldGaussian(Likelihood):
    noise_var: float

    def log_prob(self, f, y, X=None):
        return nd.Normal(f, jnp.sqrt(self.noise_var)).log_prob(y).sum()


class _OldBernoulli(Likelihood):
    def log_prob(self, f, y, X=None):
        return nd.Bernoulli(logits=f).log_prob(y).sum()


class _OldPoisson(Likelihood):
    def log_prob(self, f, y, X=None):
        return nd.Poisson(rate=jnp.exp(f)).log_prob(y).sum()


class _OldStudentT(Likelihood):
    df: float
    scale: float

    def log_prob(self, f, y, X=None):
        return nd.StudentT(df=self.df, loc=f, scale=self.scale).log_prob(y).sum()


def _old_softmax(f, y):
    return nd.Categorical(logits=f).log_prob(y).sum()


def _old_heteroscedastic(f, y):
    loc = f[..., 0]
    log_scale = f[..., 1]
    return nd.Normal(loc=loc, scale=jnp.exp(log_scale)).log_prob(y).sum()


# --- random problems -----------------------------------------------------------


def _latents(key, shape=(N,)):
    return 2.0 * jr.normal(key, shape)


def _scalar_cases():
    """``(new, old, y)`` on random targets in each likelihood's support."""
    k1, k2, k3, k4 = jr.split(jr.key(0), 4)
    y_real = jr.normal(k1, (N,))
    y_binary = jr.bernoulli(k2, 0.4, (N,)).astype(jnp.float64)
    y_count = jr.poisson(k3, 3.0, (N,)).astype(jnp.float64)
    y_heavy = jr.t(k4, 3.0, (N,))
    return {
        "gaussian": (GaussianLikelihood(noise_var=0.3), _OldGaussian(0.3), y_real),
        "bernoulli": (BernoulliLikelihood(), _OldBernoulli(), y_binary),
        "poisson": (PoissonLikelihood(), _OldPoisson(), y_count),
        "student_t": (
            StudentTLikelihood(df=3.5, scale=0.7),
            _OldStudentT(3.5, 0.7),
            y_heavy,
        ),
    }


SCALAR = sorted(_scalar_cases())


def _case(name):
    return _scalar_cases()[name]


# --- log_prob ---------------------------------------------------------------


@pytest.mark.parametrize("name", SCALAR)
def test_log_prob_equals_the_old_implementation(name) -> None:
    new, old, y = _case(name)
    f = _latents(jr.key(1))
    assert isinstance(new, AbstractObservation)
    assert jnp.allclose(new.log_prob(f, y), old.log_prob(f, y), rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("name", SCALAR)
def test_log_prob_broadcasts_a_batch_of_latents_as_before(name) -> None:
    new, old, y = _case(name)
    f = _latents(jr.key(2), (3, N))
    assert jnp.allclose(new.log_prob(f, y), old.log_prob(f, y), rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("name", SCALAR)
def test_log_prob_is_build_with_x_in_data(name) -> None:
    new, _, y = _case(name)
    f = _latents(jr.key(3))
    X = jr.normal(jr.key(4), (N, 2))
    built = new.build(y, {}, {"X": X})
    assert isinstance(built, gx.AbstractLikelihood)
    assert type(built).__name__ == type(new).__name__
    assert jnp.allclose(new.log_prob(f, y, X), built.log_prob(f), rtol=RTOL, atol=0.0)


def test_softmax_log_prob_equals_the_old_implementation() -> None:
    lik = SoftmaxLikelihood(num_classes=4)
    f = _latents(jr.key(5), (N, 4))
    y = jr.randint(jr.key(6), (N,), 0, 4)
    assert jnp.allclose(lik.log_prob(f, y), _old_softmax(f, y), rtol=RTOL, atol=0.0)
    batch = _latents(jr.key(7), (3, N, 4))
    assert jnp.allclose(
        lik.log_prob(batch, y), _old_softmax(batch, y), rtol=RTOL, atol=0.0
    )


def test_heteroscedastic_log_prob_equals_the_old_implementation() -> None:
    lik = HeteroscedasticGaussianLikelihood()
    mean = _latents(jr.key(8))
    log_scale = jr.normal(jr.key(9), (N,))
    f = jnp.stack([mean, log_scale], axis=-1)
    y = jr.normal(jr.key(10), (N,))
    assert jnp.allclose(
        lik.log_prob(f, y), _old_heteroscedastic(f, y), rtol=RTOL, atol=0.0
    )
    grad_new = jax.grad(lik.log_prob)(f, y)
    grad_old = jax.grad(_old_heteroscedastic)(f, y)
    assert jnp.allclose(grad_new, grad_old, rtol=RTOL, atol=1e-12)


def test_multi_latent_declarations_are_kept() -> None:
    softmax = SoftmaxLikelihood(3)
    assert softmax.latent_dim == 3 and softmax.num_classes == 3
    assert HeteroscedasticGaussianLikelihood().latent_dim == 2
    assert BernoulliLikelihood().latent_dim == 1
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        SoftmaxLikelihood(num_classes=1)


def test_gaussian_is_the_core_class_for_closed_form_dispatch() -> None:
    assert issubclass(GaussianLikelihood, CoreGaussianLikelihood)
    assert GaussianLikelihood(0.1).noise_var == 0.1


# --- what the inference strategies evaluate ---------------------------------


@pytest.mark.parametrize("name", SCALAR)
def test_site_derivatives_equal_the_old_implementation(name) -> None:
    """Laplace / GN / QN's per-site gradient and curvature."""
    new, old, y = _case(name)
    f = _latents(jr.key(11))
    X = jnp.zeros((N, 1))
    g_new, h_new = _per_point_grad_hess(
        _log_prob_per_point_factory(new), f, _site_data(new, y, X)
    )
    g_old, h_old = _per_point_grad_hess(
        _log_prob_per_point_factory(old), f, _site_data(old, y, X)
    )
    # g = y - exp(f) and the like cross zero, so a 1e-12 floor stands in
    # for the relative test there.
    assert jnp.allclose(g_new, g_old, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(h_new, h_old, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", SCALAR)
def test_expected_log_likelihood_equals_the_old_implementation(name) -> None:
    """The SVGP ELL, through the observation path and the protocol path."""
    new, old, y = _case(name)
    mean = _latents(jr.key(12))
    var = jnp.exp(jr.normal(jr.key(13), (N,)) - 1.0)
    X = jr.normal(jr.key(14), (N, 2))
    rule = gx.GaussHermiteIntegrator(order=20)
    got = _ell_numerical(new, y, mean, var, rule, X)
    expected = _ell_numerical(old, y, mean, var, rule, X)
    assert jnp.allclose(got, expected, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("name", ["bernoulli", "poisson", "student_t"])
def test_cvi_site_gradients_equal_the_old_implementation(name) -> None:
    new, old, y = _case(name)
    mean = _latents(jr.key(15))
    var = jnp.exp(jr.normal(jr.key(16), (N,)) - 1.0)
    cvi = ConjugateVI(integrator=gx.GaussHermiteIntegrator(order=20))
    got = cvi._site_gradients(new, y, mean, var, jnp.zeros((N, 1)))
    expected = cvi._site_gradients(old, y, mean, var)
    for a, b in zip(got, expected, strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)


def _classification_prior():
    X = jnp.linspace(-3.0, 3.0, 12)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float64)
    return GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X), y


@pytest.mark.parametrize(
    "strategy",
    [
        LaplaceInference(max_iter=5),
        GaussNewtonInference(max_iter=5),
        PosteriorLinearization(max_iter=3),
        ExpectationPropagation(max_iter=3),
    ],
    ids=["laplace", "gauss_newton", "pl", "ep"],
)
def test_dense_fit_equals_the_old_implementation(strategy) -> None:
    prior, y = _classification_prior()
    new = strategy.fit(prior, BernoulliLikelihood(), y)
    old = strategy.fit(prior, _OldBernoulli(), y)
    assert jnp.allclose(new.q_mean, old.q_mean, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(new.q_var, old.q_var, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(
        new.log_marginal_approx, old.log_marginal_approx, rtol=RTOL, atol=0.0
    )


def test_svgp_elbo_equals_the_old_implementation() -> None:
    Z = jnp.linspace(-2.0, 2.0, 5)[:, None]
    X = jnp.linspace(-3.0, 3.0, N)[:, None]
    prior = SparseGPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5), Z=Z, jitter=1e-4
    )
    guide = FullRankGuide.init(num_inducing=5)
    y = jr.poisson(jr.key(17), 2.0, (N,)).astype(jnp.float64)
    rule = gx.GaussHermiteIntegrator(order=20)
    got = svgp_elbo(prior, guide, PoissonLikelihood(), X, y, integrator=rule)
    expected = svgp_elbo(prior, guide, _OldPoisson(), X, y, integrator=rule)
    assert jnp.allclose(got, expected, rtol=RTOL, atol=0.0)


# --- an input-dependent observation model ------------------------------------


class _InputNoiseGaussian(AbstractObservation):
    r"""$y_i \sim N(f_i, \sigma(x_i)^2)$ with $\sigma(x) = a + b \lvert x \rvert$."""

    a: float = 0.2
    b: float = 0.5
    name: str = eqx.field(static=True, default="input_noise")

    def noise_std(self, X):
        return self.a + self.b * jnp.abs(X[:, 0])

    def build(self, y, theta, data):
        std = self.noise_std(data["X"])
        return gx.CallableLikelihood(
            lambda f_i, d: nd.Normal(f_i, d["std"]).log_prob(d["y"]),
            {"y": y, "std": std},
        )


def _input_noise_problem():
    X = jnp.linspace(-2.0, 2.0, 7)[:, None]
    y = jnp.sin(X[:, 0])
    return _InputNoiseGaussian(), X, y


def test_heteroscedastic_model_reads_x_from_data() -> None:
    lik, X, y = _input_noise_problem()
    f = 0.3 * jnp.ones_like(y)
    std = lik.noise_std(X)
    expected = nd.Normal(f, std).log_prob(y).sum()
    got = lik.build(y, {}, {"X": X}).log_prob(f)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_heteroscedastic_model_gets_each_site_its_input_in_the_ell() -> None:
    lik, X, y = _input_noise_problem()
    mean = jnp.zeros_like(y)
    var = 0.4 * jnp.ones_like(y)
    std = lik.noise_std(X)
    got = _ell_numerical(lik, y, mean, var, gx.GaussHermiteIntegrator(order=10), X)
    # E[log N(y; f, s²)] = log N(y; m, s²) - v / (2 s²), per site
    expected = (nd.Normal(mean, std).log_prob(y) - var / (2.0 * std**2)).sum()
    assert jnp.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_heteroscedastic_model_gets_each_site_its_input_in_a_dense_fit() -> None:
    lik, X, y = _input_noise_problem()
    f = jnp.zeros_like(y)
    std = lik.noise_std(X)
    g, h = _per_point_grad_hess(
        _log_prob_per_point_factory(lik), f, _site_data(lik, y, X)
    )
    assert jnp.allclose(g, (y - f) / std**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(h, -1.0 / std**2, rtol=1e-12, atol=0.0)
    prior = GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X)
    cond = LaplaceInference(max_iter=3).fit(prior, lik, y)
    # Gaussian sites: one Newton step is exact, with precision 1 / s²
    assert jnp.allclose(cond.site_nat2, 1.0 / std**2, rtol=1e-10, atol=0.0)
