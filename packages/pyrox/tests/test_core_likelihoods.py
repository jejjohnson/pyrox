"""``pyrox.likelihoods``: the observation-model and guide protocols."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as nd
import pyrox
import pytest
from pyrox.likelihoods import (
    GaussianLikelihood,
    Guide,
    Likelihood,
    likelihood_accepts_inputs,
)


def test_likelihoods_is_a_public_namespace_of_the_core():
    assert "likelihoods" in pyrox.__all__
    assert set(pyrox.likelihoods.__all__) == {
        "GaussianLikelihood",
        "Guide",
        "Likelihood",
        "likelihood_accepts_inputs",
    }


@pytest.mark.parametrize("protocol", [Likelihood, Guide])
def test_protocols_are_abstract(protocol):
    with pytest.raises(TypeError):
        protocol()


def test_gaussian_log_prob_is_the_summed_normal_density():
    f = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.1, 1.9, 3.2])
    lik = GaussianLikelihood(noise_var=0.25)
    assert isinstance(lik, Likelihood)
    expected = nd.Normal(f, 0.5).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), expected, rtol=1e-12)


class _TwoArgument(Likelihood):
    def log_prob(self, f, y):
        return jnp.sum(f - y)


class _VarArgs(Likelihood):
    def log_prob(self, f, y, *rest):
        return jnp.sum(f - y)


@pytest.mark.parametrize(
    ("lik", "expected"),
    [
        (GaussianLikelihood(noise_var=1.0), True),
        (_TwoArgument(), False),
        (_VarArgs(), True),
    ],
)
def test_likelihood_accepts_inputs_reads_the_signature(lik, expected):
    assert likelihood_accepts_inputs(lik) is expected
