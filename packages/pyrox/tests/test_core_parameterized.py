"""Tests for pyrox._core.Parameterized: registry, priors, guides, modes."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox._core import Parameterized, pyrox_method
from pyrox._core.parameterized import _State


class RBFKernel(Parameterized):
    pyrox_name = "RBFKernel"

    @pyrox_method
    def __call__(self, X1, X2):
        v = self.get_param("variance")
        ls = self.get_param("lengthscale")
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ls**2, axis=-1)
        return v * jnp.exp(-0.5 * sq)

    def setup(self):
        self.register_param(
            "variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))


# --- setup() and registry --------------------------------------------------


def test_setup_is_invoked_on_construction():
    k = RBFKernel()
    state = k._state()
    assert set(state.params) == {"variance", "lengthscale"}
    assert state.params["variance"].prior is not None
    assert state.params["lengthscale"].prior is None
    assert state.mode == "model"


def test_register_param_before_set_prior_raises_keyerror():
    class Empty(Parameterized):
        pyrox_name = "Empty"

    k = Empty()
    with pytest.raises(KeyError, match="not registered"):
        k.set_prior("missing", dist.Normal(0.0, 1.0))


# --- mode switching --------------------------------------------------------


def test_model_mode_with_prior_registers_sample_site():
    k = RBFKernel()
    X = jnp.array([[0.0], [1.0]])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert tr["RBFKernel.variance"]["type"] == "sample"
    # lengthscale has no prior — stays as param
    assert tr["RBFKernel.lengthscale"]["type"] == "param"


def test_guide_mode_delta_registers_param_backed_delta_sample():
    """The `delta` guide must expose the latent as a *sample* site (a
    ``Delta`` backed by a constrained ``{name}_loc`` param) so NumPyro's
    ``replay`` handler conditions the model on it. A bare param site is
    invisible to SVI. Regression for #182.
    """
    k = RBFKernel()
    k.set_mode("guide")
    X = jnp.array([[0.0], [1.0]])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    # variance has a prior → delta guide: param-backed Delta sample site.
    assert tr["RBFKernel.variance"]["type"] == "sample"
    assert isinstance(tr["RBFKernel.variance"]["fn"], dist.Delta)
    assert tr["RBFKernel.variance_loc"]["type"] == "param"
    # lengthscale has no prior → stays a plain param in guide mode too.
    assert tr["RBFKernel.lengthscale"]["type"] == "param"


def test_svi_delta_guide_converges_to_posterior_mode():
    """Real SVI with the default `delta` guide must run and recover the
    posterior mode — the bug in #182 crashed on the first step.
    """
    import jax

    class K(Parameterized):
        pyrox_name = "K"

        @pyrox_method
        def __call__(self):
            return self.get_param("mu")

        def setup(self):
            self.register_param("mu", jnp.array(0.0))
            self.set_prior("mu", dist.Normal(0.0, 10.0))
            self.autoguide("mu", "delta")

    from numpyro.infer import SVI, Trace_ELBO

    k = K()
    y = jnp.full((200,), 3.0)

    def model(y):
        k.set_mode("model")
        numpyro.sample("obs", dist.Normal(k(), 1.0), obs=y)

    def guide(y):
        k.set_mode("guide")
        k()

    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), 2000, y, progress_bar=False)
    assert bool(jnp.isfinite(res.losses[-1]))
    # posterior mode of Normal-Normal ≈ data mean (prior is near-flat)
    assert float(res.params["K.mu_loc"]) == pytest.approx(3.0, abs=0.05)


def test_svi_delta_guide_respects_constraint():
    """A positive-constrained delta point estimate stays in support."""
    import jax

    class K(Parameterized):
        pyrox_name = "K"

        @pyrox_method
        def __call__(self):
            return self.get_param("sigma")

        def setup(self):
            self.register_param(
                "sigma", jnp.array(1.0), constraint=dist.constraints.positive
            )
            self.set_prior("sigma", dist.LogNormal(0.0, 1.0))
            self.autoguide("sigma", "delta")

    from numpyro.infer import SVI, Trace_ELBO

    k = K()
    y = jax.random.normal(jax.random.PRNGKey(1), (400,)) * 2.5

    def model(y):
        k.set_mode("model")
        numpyro.sample("obs", dist.Normal(0.0, k()), obs=y)

    def guide(y):
        k.set_mode("guide")
        k()

    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), 2000, y, progress_bar=False)
    assert bool(jnp.isfinite(res.losses[-1]))
    # The Delta point estimate is drawn in-support at every step.
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0), k._get_context():
        k.load_pyro_samples()
    assert float(tr["K.sigma"]["value"]) > 0.0


def test_guide_mode_normal_adds_variational_params():
    class NK(Parameterized):
        pyrox_name = "NK"

        @pyrox_method
        def __call__(self):
            return self.get_param("v")

        def setup(self):
            self.register_param("v", jnp.array(1.0))
            self.set_prior("v", dist.Normal(0.0, 1.0))
            self.autoguide("v", "normal")

    k = NK()
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k()
    assert "NK.v_loc" in tr
    assert "NK.v_scale" in tr
    assert tr["NK.v"]["type"] == "sample"


def test_guide_mode_normal_respects_positive_constraint():
    """Regression for PR #57 review: the `normal` guide must keep draws
    inside the prior's support. A positive-support param with a LogNormal
    prior and `autoguide("normal")` must never sample negative values.
    """

    class PosK(Parameterized):
        pyrox_name = "PosK"

        @pyrox_method
        def __call__(self):
            return self.get_param("v")

        def setup(self):
            self.register_param(
                "v",
                jnp.array(1.0),
                constraint=dist.constraints.positive,
            )
            self.set_prior("v", dist.LogNormal(0.0, 1.0))
            self.autoguide("v", "normal")

    k = PosK()
    k.set_mode("guide")
    samples = []
    for seed in range(30):
        with handlers.trace() as tr, handlers.seed(rng_seed=seed):
            _ = k()
        samples.append(float(tr["PosK.v"]["value"]))
    # Every guide draw must stay in the prior's (positive) support.
    assert all(s > 0.0 for s in samples)
    # The guide distribution is a TransformedDistribution wrapping the
    # unconstrained Normal — not a bare Normal.
    site_fn = tr["PosK.v"]["fn"]
    assert isinstance(site_fn, dist.TransformedDistribution)


def test_autoguide_rejects_unknown_guide_type():
    k = RBFKernel()
    with pytest.raises(ValueError, match="guide_type must be"):
        k.autoguide("variance", "bogus")  # type: ignore[arg-type]


def test_set_mode_rejects_unknown_mode():
    k = RBFKernel()
    with pytest.raises(ValueError, match="mode must be"):
        k.set_mode("bogus")  # type: ignore[arg-type]


def test_delta_guide_uses_prior_support_not_registered_constraint():
    """The delta point estimate must live in the *prior's* support, even when
    the param was registered without a matching constraint — otherwise SVI
    replays the model prior outside its domain and diverges. Regression for
    the #186 Codex review (finding: constrain Delta to the prior support).
    """
    import jax
    from numpyro.infer import SVI, Trace_ELBO

    class K(Parameterized):
        pyrox_name = "K"

        @pyrox_method
        def __call__(self):
            return self.get_param("sigma")

        def setup(self):
            self.register_param("sigma", jnp.array(1.0))  # no constraint
            self.set_prior("sigma", dist.LogNormal(0.0, 1.0))  # positive support
            self.autoguide("sigma", "delta")

    k = K()
    y = jax.random.normal(jax.random.PRNGKey(1), (400,)) * 2.5

    def model(y):
        k.set_mode("model")
        numpyro.sample("obs", dist.Normal(0.0, k()), obs=y)

    def guide(y):
        k.set_mode("guide")
        k()

    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), 1500, y, progress_bar=False)
    # No NaN divergence, and the point estimate stays in the positive support.
    assert bool(jnp.isfinite(res.losses[-1]))
    assert float(res.params["K.sigma_loc"]) > 0.0
    # Recovers roughly the data scale (MLE ≈ 2.5, prior near-flat there).
    assert float(res.params["K.sigma_loc"]) == pytest.approx(2.5, abs=0.5)


def test_delta_guide_preserves_prior_event_dim():
    """The Delta's ``event_dim`` must match the prior's, so batched priors keep
    their plate/batch dimensions instead of collapsing the whole value into a
    single event. Regression for the #186 Codex review (finding: preserve
    prior batch dimensions in Delta guides).
    """

    class Batched(Parameterized):
        pyrox_name = "Batched"

        @pyrox_method
        def __call__(self):
            return self.get_param("w")

        def setup(self):
            self.register_param("w", jnp.zeros(5))
            self.set_prior("w", dist.Normal(jnp.zeros(5), 1.0))  # event_dim 0
            self.autoguide("w", "delta")

    k = Batched()
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        out = k()
    assert out.shape == (5,)
    assert tr["Batched.w"]["fn"].event_dim == 0  # matches batched prior
    assert tr["Batched.w_loc"]["value"].shape == (5,)

    class Joint(Parameterized):
        pyrox_name = "Joint"

        @pyrox_method
        def __call__(self):
            return self.get_param("w")

        def setup(self):
            self.register_param("w", jnp.zeros(3))
            self.set_prior("w", dist.Normal(jnp.zeros(3), 1.0).to_event(1))
            self.autoguide("w", "delta")

    j = Joint()
    j.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = j()
    assert tr["Joint.w"]["fn"].event_dim == 1  # matches to_event(1) prior


def test_autoguide_rejects_mvn_at_declaration():
    """`mvn` is not implemented; it must be rejected at ``autoguide()`` time
    (with a clear message) rather than deep inside a trace. Regression for
    the #185 late-failure nit.
    """
    k = RBFKernel()
    with pytest.raises(ValueError, match="guide_type must be"):
        k.autoguide("variance", "mvn")  # type: ignore[arg-type]


def test_guide_mode_normal_simplex_constraint():
    """The `normal` guide must handle shape-changing constraints (simplex:
    K -> K-1 in unconstrained space). Regression for #183 — the old code
    sized the variational scale from the constrained init and failed to
    broadcast against the unconstrained loc.
    """

    class SK(Parameterized):
        pyrox_name = "SK"

        @pyrox_method
        def __call__(self):
            return self.get_param("probs")

        def setup(self):
            self.register_param(
                "probs",
                jnp.ones(3) / 3,
                constraint=dist.constraints.simplex,
            )
            self.set_prior("probs", dist.Dirichlet(jnp.ones(3)))
            self.autoguide("probs", "normal")

    k = SK()
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        out = k()
    # Guide draw lands on the simplex: shape (3,), positive, sums to 1.
    assert out.shape == (3,)
    assert bool(jnp.all(out > 0.0))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-5)
    # Latent site is the TransformedDistribution over the simplex.
    assert isinstance(tr["SK.probs"]["fn"], dist.TransformedDistribution)
    # Variational params live in unconstrained space: shape (2,), not (3,).
    assert tr["SK.probs_loc"]["value"].shape == (2,)
    assert tr["SK.probs_scale"]["value"].shape == (2,)


def test_svi_normal_guide_simplex_end_to_end():
    """One SVI.run through a simplex `normal` guide must produce finite
    losses — guards against event_dim mistakes that only surface in the
    Trace_ELBO log-density reduction.
    """
    import jax
    from numpyro.infer import SVI, Trace_ELBO

    class SK(Parameterized):
        pyrox_name = "SK"

        @pyrox_method
        def __call__(self):
            return self.get_param("probs")

        def setup(self):
            self.register_param(
                "probs",
                jnp.ones(3) / 3,
                constraint=dist.constraints.simplex,
            )
            self.set_prior("probs", dist.Dirichlet(jnp.ones(3)))
            self.autoguide("probs", "normal")

    k = SK()
    counts = jnp.array([12, 30, 8])
    total = int(counts.sum())  # concrete, captured outside the traced model

    def model(counts):
        k.set_mode("model")
        numpyro.sample("obs", dist.Multinomial(total, k()), obs=counts)

    def guide(counts):
        k.set_mode("guide")
        k()

    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), 300, counts, progress_bar=False)
    assert bool(jnp.isfinite(res.losses[-1]))


# --- load_pyro_samples -----------------------------------------------------


def test_load_pyro_samples_touches_every_site():
    k = RBFKernel()
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        k._get_context(),
    ):
        k.load_pyro_samples()
    assert "RBFKernel.variance" in tr
    assert "RBFKernel.lengthscale" in tr


# --- teardown --------------------------------------------------------------


def test_teardown_removes_registry_entry():
    k = RBFKernel()
    _ = k._state()
    assert id(k) in Parameterized._registry
    k._teardown()
    assert id(k) not in Parameterized._registry


# --- instance isolation ----------------------------------------------------


def test_distinct_instances_have_distinct_state():
    k1 = RBFKernel()
    k2 = RBFKernel()
    assert isinstance(k1._state(), _State)
    assert k1._state() is not k2._state()
    k1.set_mode("guide")
    assert k2._state().mode == "model"
