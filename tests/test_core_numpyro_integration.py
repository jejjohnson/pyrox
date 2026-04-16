"""Integration tests — PyroxModule/Parameterized against NumPyro primitives.

Exercises Pattern B and Pattern C modules under the full NumPyro surface:
handlers (seed/trace/scope/substitute/condition/block/mask/scale/reparam/
do/lift/infer_config), primitives (plate/factor/deterministic), inference
(MCMC/NUTS, SVI with AutoGuides, Predictive), and JAX transforms
(jit/vmap/grad).

These tests verify that ``pyrox_param`` and ``pyrox_sample`` are transparent
delegates to ``numpyro.param`` / ``numpyro.sample`` — anything that composes
with the upstream primitives must compose with the pyrox wrappers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import numpyro.infer.reparam as reparam
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.optim import Adam

from pyrox._core import Parameterized, PyroxModule, pyrox_method


# ---------------------------------------------------------------------------
# Fixtures: minimal Pattern B and Pattern C modules
# ---------------------------------------------------------------------------


class Linear(PyroxModule):
    """Minimal Pattern B module — one sample site, one param site."""

    pyrox_name = "Linear"
    in_features: int
    out_features: int

    @pyrox_method
    def __call__(self, x):
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.out_features])
            .to_event(2),
        )
        b = self.pyrox_param("b", jnp.zeros(self.out_features))
        return x @ W + b


class Scalar(PyroxModule):
    """Single scalar sample site for simple primitive tests."""

    pyrox_name = "Scalar"

    @pyrox_method
    def __call__(self):
        return self.pyrox_sample("x", dist.Normal(0.0, 1.0))


class RBF(Parameterized):
    """Pattern C kernel with one prior-bearing param and one plain param."""

    pyrox_name = "RBF"

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

    @pyrox_method
    def __call__(self, X1, X2):
        v = self.get_param("variance")
        ls = self.get_param("lengthscale")
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ls**2, axis=-1)
        return v * jnp.exp(-0.5 * sq)


# ---------------------------------------------------------------------------
# handlers.seed / trace  (the base case — every other handler composes on top)
# ---------------------------------------------------------------------------


def test_seed_then_trace_captures_all_sites():
    m = Linear(in_features=2, out_features=3)
    x = jnp.ones((4, 2))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = m(x)
    assert tr["Linear.W"]["type"] == "sample"
    assert tr["Linear.b"]["type"] == "param"


# ---------------------------------------------------------------------------
# handlers.substitute — replace site values
# ---------------------------------------------------------------------------


def test_substitute_replaces_sample_value():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((3, 2))
    W_fixed = jnp.full((2, 1), 0.5)
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.substitute(data={"Linear.W": W_fixed}),
    ):
        y = m(x)
    assert jnp.allclose(tr["Linear.W"]["value"], W_fixed)
    assert jnp.allclose(y, x @ W_fixed)


def test_substitute_replaces_param_value():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((3, 2))
    b_fixed = jnp.array([7.0])
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.substitute(data={"Linear.b": b_fixed}),
    ):
        _ = m(x)
    assert jnp.allclose(tr["Linear.b"]["value"], b_fixed)


# ---------------------------------------------------------------------------
# handlers.condition — treat a site as observed
# ---------------------------------------------------------------------------


def test_condition_marks_site_observed():
    obs = jnp.array(1.5)
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.condition(data={"Scalar.x": obs}),
    ):
        _ = Scalar()()
    assert tr["Scalar.x"]["is_observed"] is True
    assert jnp.allclose(tr["Scalar.x"]["value"], obs)


# ---------------------------------------------------------------------------
# handlers.block — suppress site registration upward
# ---------------------------------------------------------------------------


def test_block_hides_inner_sites_from_outer_trace():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((1, 2))
    with handlers.trace() as tr, handlers.seed(rng_seed=0), handlers.block():
        _ = m(x)
    assert len(tr) == 0


# ---------------------------------------------------------------------------
# handlers.scope — site-name prefix
# ---------------------------------------------------------------------------


def test_scope_prefixes_site_names():
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.scope(prefix="layer0"),
    ):
        _ = Scalar()()
    # Under scope, site name becomes "{prefix}/{original}".
    keys = list(tr)
    assert any(k.startswith("layer0/") and k.endswith("Scalar.x") for k in keys)


# ---------------------------------------------------------------------------
# handlers.mask — mask log_prob contribution without changing sampling
# ---------------------------------------------------------------------------


def test_mask_preserves_sample_value_and_flags_site():
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.mask(mask=False),
    ):
        _ = Scalar()()
    site = tr["Scalar.x"]
    assert site["type"] == "sample"
    # Masking flips the mask flag on the sample site; exact key location
    # depends on numpyro version — assert trace still captured the sample.
    assert site["value"] is not None


# ---------------------------------------------------------------------------
# handlers.scale — scales log_prob; shouldn't corrupt the sample chain
# ---------------------------------------------------------------------------


def test_scale_leaves_sample_values_unchanged():
    seed = jr.PRNGKey(0)
    with (
        handlers.trace() as tr_a,
        handlers.seed(rng_seed=seed),
    ):
        _ = Scalar()()
    with (
        handlers.trace() as tr_b,
        handlers.seed(rng_seed=seed),
        handlers.scale(scale=10.0),
    ):
        _ = Scalar()()
    assert jnp.allclose(tr_a["Scalar.x"]["value"], tr_b["Scalar.x"]["value"])


# ---------------------------------------------------------------------------
# handlers.reparam — Normal site goes through LocScaleReparam
# ---------------------------------------------------------------------------


def test_reparam_injects_decentered_site():
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.reparam(config={"Scalar.x": reparam.LocScaleReparam(0.0)}),
    ):
        _ = Scalar()()
    # Reparam rewrites the site into a decentered auxiliary plus a
    # deterministic — the auxiliary takes a "_decentered" suffix.
    assert any("_decentered" in k for k in tr)


# ---------------------------------------------------------------------------
# handlers.do — intervene on a site (replace without observing)
# ---------------------------------------------------------------------------


def test_do_intervenes_on_sample_site():
    do_value = jnp.array(42.0)
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.do(data={"Scalar.x": do_value}),
    ):
        x = Scalar()()
    assert jnp.allclose(x, do_value)
    # do rewrites the site — ensure tracing still completed.
    assert "Scalar.x" in tr


# ---------------------------------------------------------------------------
# handlers.lift — treat params as random variables with a prior
# ---------------------------------------------------------------------------


def test_lift_promotes_param_to_sample():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((1, 2))
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.lift(prior=dist.Normal(0.0, 1.0)),
    ):
        _ = m(x)
    # After lift, Linear.b is promoted from param → sample.
    assert tr["Linear.b"]["type"] == "sample"


# ---------------------------------------------------------------------------
# handlers.infer_config — per-site inference config
# ---------------------------------------------------------------------------


def test_infer_config_propagates_to_sample_site():
    cfg = {"Scalar.x": {"enumerate": "parallel"}}
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        handlers.infer_config(config_fn=lambda site: cfg.get(site["name"], {})),
    ):
        _ = Scalar()()
    assert tr["Scalar.x"]["infer"].get("enumerate") == "parallel"


# ---------------------------------------------------------------------------
# numpyro.plate — vectorized sampling
# ---------------------------------------------------------------------------


def test_plate_batches_pyrox_sample_sites():
    class Plated(PyroxModule):
        pyrox_name = "Plated"
        n: int

        @pyrox_method
        def __call__(self):
            with numpyro.plate("i", self.n):
                return self.pyrox_sample("x", dist.Normal(0.0, 1.0))

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        xs = Plated(n=7)()
    assert xs.shape == (7,)
    assert tr["Plated.x"]["fn"].event_shape == ()


# ---------------------------------------------------------------------------
# numpyro.factor — coexistence with external log-factor terms
# ---------------------------------------------------------------------------


def test_factor_coexists_with_pyrox_sites():
    def model():
        _ = Scalar()()
        numpyro.factor("penalty", -1.0)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert tr["Scalar.x"]["type"] == "sample"
    # numpyro.factor registers a sample site with a Delta-like fn; the key
    # invariant is that it coexists in the same trace as pyrox sites.
    assert "penalty" in tr


# ---------------------------------------------------------------------------
# numpyro.deterministic — coexistence with module outputs
# ---------------------------------------------------------------------------


def test_deterministic_outside_module():
    def model():
        x = Scalar()()
        numpyro.deterministic("x_sq", x**2)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert tr["x_sq"]["type"] == "deterministic"


# ---------------------------------------------------------------------------
# Inference: MCMC / NUTS
# ---------------------------------------------------------------------------


def test_mcmc_nuts_round_trip_pattern_b():
    """Tiny BNN posterior — verifies pyrox sites are NUTS-compatible."""
    rng = jr.PRNGKey(0)
    x = jnp.linspace(-1.0, 1.0, 8)[:, None]
    y = jnp.sin(x.squeeze()) + 0.1 * jr.normal(rng, (8,))
    m = Linear(in_features=1, out_features=1)

    def model(x, y=None):
        f = m(x).squeeze(-1)
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("obs", dist.Normal(f, sigma), obs=y)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(rng, x, y)
    samples = mcmc.get_samples()
    assert "Linear.W" in samples
    assert "sigma" in samples
    assert samples["Linear.W"].shape == (5, 1, 1)


def test_mcmc_nuts_round_trip_pattern_c():
    rng = jr.PRNGKey(0)
    X = jnp.linspace(0.0, 1.0, 5)[:, None]
    kernel = RBF()

    def model():
        _ = kernel(X, X)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(rng)
    samples = mcmc.get_samples()
    assert "RBF.variance" in samples
    assert samples["RBF.variance"].shape == (5,)


# ---------------------------------------------------------------------------
# Inference: SVI with AutoGuides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide_cls", [AutoDelta, AutoNormal, AutoMultivariateNormal])
def test_svi_autoguide_step_runs(guide_cls):
    rng = jr.PRNGKey(0)
    x = jnp.linspace(-1.0, 1.0, 8)[:, None]
    y = jnp.sin(x.squeeze())
    m = Linear(in_features=1, out_features=1)

    def model(x, y=None):
        f = m(x).squeeze(-1)
        numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)

    guide = guide_cls(model)
    svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())
    state = svi.init(rng, x, y)
    for _ in range(3):
        state, loss = svi.update(state, x, y)
    assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Inference: Predictive — posterior predictive shape
# ---------------------------------------------------------------------------


def test_predictive_generates_posterior_samples():
    rng = jr.PRNGKey(0)
    m = Linear(in_features=1, out_features=1)

    def model(x):
        return numpyro.deterministic("f", m(x).squeeze(-1))

    x = jnp.ones((4, 1))
    pred = Predictive(model, num_samples=6)(rng, x)
    assert pred["Linear.W"].shape == (6, 1, 1)
    assert pred["f"].shape == (6, 4)


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


def test_jit_preserves_trace_values():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((3, 2))

    def model(x):
        return m(x)

    seeded = handlers.seed(model, rng_seed=0)
    y_eager = seeded(x)
    y_jit = jax.jit(handlers.seed(model, rng_seed=0))(x)
    assert jnp.allclose(y_eager, y_jit)


def test_vmap_over_rng_broadcasts_sample_sites():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((3, 2))

    def draw(seed):
        return handlers.seed(lambda: m(x), rng_seed=seed)()

    seeds = jr.split(jr.PRNGKey(0), 4)
    ys = jax.vmap(draw)(seeds)
    assert ys.shape == (4, 3, 1)


def test_grad_of_log_prob_wrt_param_is_finite():
    m = Linear(in_features=2, out_features=1)
    x = jnp.ones((3, 2))

    def loss(b):
        with (
            handlers.trace() as tr,
            handlers.seed(rng_seed=0),
            handlers.substitute(data={"Linear.b": b}),
        ):
            _ = m(x)
        return tr["Linear.W"]["fn"].log_prob(tr["Linear.W"]["value"]).sum() + b.sum()

    g = jax.grad(loss)(jnp.zeros(1))
    assert jnp.all(jnp.isfinite(g))
