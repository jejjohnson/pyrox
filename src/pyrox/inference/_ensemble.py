"""Ensemble-of-MAP / ensemble-of-VI primitives.

Three layers, exposed publicly so users can pick their level of control:

**Layer 1 — Functional primitives (free functions, pure):**

* :func:`ensemble_init` — initialize ``E`` ensemble members + per-member
  optimizer states by ``vmap``-ing ``init_fn`` over split keys.
* :func:`ensemble_step` — one ensemble update step on a batch.
* :func:`ensemble_loss` — vmapped negative-log-joint from a user-supplied
  ``log_joint``. Useful when the user wants to compose their own
  loss-and-grad outside ``ensemble_step``.

These primitives let users roll their own training loop with optax
schedules, custom batching, callbacks, early stopping, etc.

**Layer 2 — NumPyro-like inference ops (eqx.Module classes):**

* :class:`EnsembleMAP` — wraps Layer 1 with ``init`` / ``update`` / ``run``
  methods, mirroring :class:`numpyro.infer.SVI`'s API.
* :class:`EnsembleVI` — analogous wrapper around
  :class:`numpyro.infer.SVI` so VI surrogates fit the same shape.

**Layer 3 — One-shot sugar (top-level functions):**

* :func:`ensemble_map` — instantiate :class:`EnsembleMAP` and call
  ``.run`` in one line.
* :func:`ensemble_vi` — same for :class:`EnsembleVI`.
* :func:`ensemble_predict` — ``vmap`` a scalar-in predictive over the
  leading ensemble axis.

``optax`` is required for the MAP path and is the default optimizer for
the VI path (``pip install pyrox[optax]``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


if TYPE_CHECKING:
    import optax


def _require_optax() -> Any:
    try:
        import optax
    except ImportError as e:
        raise ImportError(
            "pyrox.inference ensemble primitives require `optax`. "
            "Install via `pip install pyrox[optax]`."
        ) from e
    return optax


# ============================================================================
# Layer 1 — Functional primitives
# ============================================================================


class EnsembleState(NamedTuple):
    """Stacked state for an ensemble of optimizer runs.

    Attributes:
        params: Stacked parameter PyTree. Array leaves carry a leading
            ``(E,)`` axis; non-array leaves (e.g. captured activation
            functions inside an :class:`equinox.Module`) are shared.
        opt_state: Stacked optax optimizer state with the same axis
            convention.
    """

    params: PyTree
    opt_state: PyTree


class EnsembleResult(NamedTuple):
    """Output of a full :meth:`EnsembleMAP.run` / :meth:`EnsembleVI.run`.

    Attributes:
        params: Final stacked parameters with leading ``(E,)`` axis.
        losses: ``(E, num_epochs)`` per-step loss history.
    """

    params: PyTree
    losses: Float[Array, "E T"]


def ensemble_init(
    init_fn: Callable[[PRNGKeyArray], PyTree],
    optimizer: optax.GradientTransformation,
    *,
    ensemble_size: int,
    seed: PRNGKeyArray,
) -> EnsembleState:
    """Initialize an ensemble of (params, opt_state) by vmap over keys.

    Args:
        init_fn: ``key -> params``. Called once per ensemble member.
        optimizer: ``optax.GradientTransformation``.
        ensemble_size: Number of independent ensemble members ``E``.
        seed: PRNG key, split into ``E`` per-member init keys.

    Returns:
        :class:`EnsembleState` with stacked ``params`` and ``opt_state``.
        Array leaves carry a leading ``(E,)`` axis.
    """
    keys = jr.split(seed, ensemble_size)

    @eqx.filter_vmap
    def _per_member(key: PRNGKeyArray) -> tuple[PyTree, Any]:
        params = init_fn(key)
        opt_state = optimizer.init(eqx.filter(params, eqx.is_inexact_array))
        return params, opt_state

    params, opt_state = _per_member(keys)
    return EnsembleState(params=params, opt_state=opt_state)


def ensemble_loss(
    log_joint: Callable[
        [PyTree, Array, Array],
        tuple[Float[Array, ""], Float[Array, ""]],
    ],
    *,
    prior_weight: float = 1.0,
    scale: float = 1.0,
) -> Callable[[PyTree, Array, Array], tuple[Float[Array, ""], PyTree]]:
    r"""Build the ``filter_value_and_grad`` loss from a log-joint.

    Returns a function ``loss_fn(params, x_batch, y_batch) -> (loss, grads)``
    that computes

    .. math::

        \mathcal{L}(\theta) = -\, \text{scale} \cdot \log p(y \mid x, \theta)
            \;-\; w_{\text{prior}} \cdot \log p(\theta).

    ``prior_weight=0`` short-circuits the prior term so the user may
    return a placeholder ``0.0`` for ``logprior``.

    Args:
        log_joint: ``(params, x_batch, y_batch) -> (loglik, logprior)``.
        prior_weight: Weight on the ``logprior`` term. ``0.0`` ⇒ MLE,
            ``1.0`` ⇒ MAP.
        scale: Multiplicative weight on the ``loglik`` term. Set to
            ``N / |B|`` for unbiased mini-batch SGD-MAP.
    """
    use_prior = prior_weight != 0.0

    @eqx.filter_value_and_grad
    def _loss(params: PyTree, xb: Array, yb: Array) -> Float[Array, ""]:
        ll, lp = log_joint(params, xb, yb)
        nll = -scale * ll
        if use_prior:
            return nll - prior_weight * lp
        return nll

    return _loss


def ensemble_step(
    state: EnsembleState,
    x_batch: Array,
    y_batch: Array,
    *,
    log_joint: Callable[
        [PyTree, Array, Array],
        tuple[Float[Array, ""], Float[Array, ""]],
    ],
    optimizer: optax.GradientTransformation,
    prior_weight: float = 1.0,
    scale: float = 1.0,
) -> tuple[EnsembleState, Float[Array, " E"]]:
    """Perform one ensemble update step on a batch.

    Each ensemble member computes ``∇L(θ_e)`` independently (vmapped),
    advances its optax state, and returns the updated params + state.

    Args:
        state: Current :class:`EnsembleState`.
        x_batch: Inputs.
        y_batch: Targets.
        log_joint: ``(params, x, y) -> (loglik, logprior)``.
        optimizer: Same ``optax.GradientTransformation`` passed to
            :func:`ensemble_init`. Stateless transforms can be re-built
            per call; stateful schedules need to share the same
            instance.
        prior_weight: Weight on ``logprior``; ``0`` ⇒ MLE.
        scale: Weight on ``loglik`` (use ``N / |B|`` for mini-batch).

    Returns:
        ``(new_state, per_member_losses)`` where losses has shape
        ``(E,)``.
    """
    loss_fn = ensemble_loss(log_joint, prior_weight=prior_weight, scale=scale)

    @eqx.filter_vmap
    def _per_member(
        params: PyTree, opt_state: Any
    ) -> tuple[PyTree, Any, Float[Array, ""]]:
        loss, grads = loss_fn(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(params, eqx.is_inexact_array)
        )
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    params, opt_state, losses = _per_member(state.params, state.opt_state)
    return EnsembleState(params=params, opt_state=opt_state), losses


# ============================================================================
# Layer 2 — NumPyro-like inference ops
# ============================================================================


class EnsembleMAP(eqx.Module):
    r"""NumPyro-like ensemble MAP/MLE runner.

    Mirrors :class:`numpyro.infer.SVI`'s ``init`` / ``update`` / ``run``
    triplet, but every operation is ensembled by ``vmap`` over the
    leading ``(E,)`` axis.

    Per-member objective is the tempered negative log-posterior

    .. math::

        \mathcal{L}_e(\theta_e) = -\frac{N}{|B|}\sum_{i \in B}
            \log p(y_i \mid x_i, \theta_e)
            \;-\; w_{\text{prior}} \cdot \log p(\theta_e),

    where :math:`w_{\text{prior}}` is :attr:`prior_weight`.

    Example:
        >>> runner = EnsembleMAP(
        ...     log_joint=log_joint,
        ...     init_fn=init_fn,
        ...     optimizer=optax.adam(5e-3),
        ...     ensemble_size=16,
        ... )
        >>> # numpyro-style three-method API
        >>> state = runner.init(jr.PRNGKey(0))
        >>> for _ in range(2000):
        ...     state, losses = runner.update(state, x, y)
        >>> # or one-shot
        >>> result = runner.run(jr.PRNGKey(0), 2000, x, y)
    """

    log_joint: Callable[
        [PyTree, Array, Array],
        tuple[Float[Array, ""], Float[Array, ""]],
    ]
    init_fn: Callable[[PRNGKeyArray], PyTree]
    optimizer: Any  # optax.GradientTransformation, but optax may not be installed
    ensemble_size: int = eqx.field(static=True, default=16)
    prior_weight: float = eqx.field(static=True, default=1.0)

    def init(self, seed: PRNGKeyArray) -> EnsembleState:
        """Initialize the ensemble. Mirrors ``numpyro.infer.SVI.init``."""
        return ensemble_init(
            self.init_fn,
            self.optimizer,
            ensemble_size=self.ensemble_size,
            seed=seed,
        )

    def update(
        self,
        state: EnsembleState,
        x_batch: Array,
        y_batch: Array,
        *,
        scale: float = 1.0,
    ) -> tuple[EnsembleState, Float[Array, " E"]]:
        """One ensemble update step. Mirrors ``numpyro.infer.SVI.update``.

        Args:
            state: Current :class:`EnsembleState`.
            x_batch: Batch inputs.
            y_batch: Batch targets.
            scale: ``N / |B|`` for mini-batch unbiased SGD-MAP. Defaults
                to ``1.0`` (full-batch).
        """
        return ensemble_step(
            state,
            x_batch,
            y_batch,
            log_joint=self.log_joint,
            optimizer=self.optimizer,
            prior_weight=self.prior_weight,
            scale=scale,
        )

    def run(
        self,
        seed: PRNGKeyArray,
        num_epochs: int,
        x: Array,
        y: Array,
        *,
        batch_size: int | None = None,
    ) -> EnsembleResult:
        """Fit the ensemble end-to-end. Mirrors ``numpyro.infer.SVI.run``.

        Internally drives :func:`ensemble_step` via ``lax.scan`` for
        speed; equivalent to a hand-written Python loop over
        :meth:`update`.

        Args:
            seed: PRNG key used for both init and (when applicable)
                mini-batch index permutation.
            num_epochs: Number of optimizer steps per member.
            x: Inputs.
            y: Targets.
            batch_size: Optional mini-batch size. ``None`` ⇒ full-batch.

        Returns:
            :class:`EnsembleResult` with stacked final params + loss
            history of shape ``(E, num_epochs)``.
        """
        n = y.shape[0]
        bsz = n if batch_size is None else batch_size
        scale = float(n) / float(bsz)
        use_minibatch = batch_size is not None and batch_size < n

        init_key, perm_key = jr.split(seed)
        state = self.init(init_key)

        # Partition non-array leaves out of the scan carry; lax.scan only
        # accepts JAX-typed carry, but eqx.Module trees may contain
        # captured Python callables (e.g. jax.nn.tanh). Static is shared
        # across ensemble members, so we recombine after the scan.
        arrays0, static = eqx.partition(state.params, eqx.is_inexact_array)

        def epoch(
            carry: tuple[PyTree, Any, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, Any, PRNGKeyArray], Float[Array, " E"]]:
            arrays, opt_state, key = carry
            if use_minibatch:
                key, sub = jr.split(key)
                idx = jr.permutation(sub, n)[:bsz]
                xb, yb = x[idx], y[idx]
            else:
                xb, yb = x, y
            params = eqx.combine(arrays, static)
            new_state, losses = ensemble_step(
                EnsembleState(params=params, opt_state=opt_state),
                xb,
                yb,
                log_joint=self.log_joint,
                optimizer=self.optimizer,
                prior_weight=self.prior_weight,
                scale=scale,
            )
            new_arrays, _ = eqx.partition(new_state.params, eqx.is_inexact_array)
            return (new_arrays, new_state.opt_state, key), losses

        (arrays_final, _, _), losses = jax.lax.scan(
            epoch, (arrays0, state.opt_state, perm_key), None, length=num_epochs
        )
        final_params = eqx.combine(arrays_final, static)
        # losses is (T, E) from scan; transpose to (E, T) per the public contract.
        return EnsembleResult(params=final_params, losses=losses.T)


class EnsembleVI(eqx.Module):
    r"""NumPyro-like ensemble variational-inference runner.

    Wraps :class:`numpyro.infer.SVI` + :class:`numpyro.infer.Trace_ELBO`
    with the same ensemble surface as :class:`EnsembleMAP`.

    Per-member objective is the tempered ELBO

    .. math::

        \mathrm{ELBO}(\phi_e) = \mathbb{E}_{q_{\phi_e}}\!
            \bigl[\log p(y \mid x, \theta)\bigr]
            - \beta\, \mathrm{KL}\!\bigl(q_{\phi_e}\,\|\,p\bigr),

    where :math:`\beta` is :attr:`kl_weight`.
    """

    model_fn: Callable[..., None]
    guide_fn: Callable[..., None]
    optimizer: Any  # numpyro.optim or optax.GradientTransformation
    ensemble_size: int = eqx.field(static=True, default=16)
    kl_weight: float = eqx.field(static=True, default=1.0)
    num_particles: int = eqx.field(static=True, default=1)

    def _build_svi(self) -> Any:
        import numpyro
        from numpyro.infer import SVI, Trace_ELBO

        opt = self.optimizer
        if not isinstance(opt, numpyro.optim._NumPyroOptim):
            opt = numpyro.optim.optax_to_numpyro(opt)
        if self.kl_weight == 1.0:
            elbo: Any = Trace_ELBO(num_particles=self.num_particles)
        else:
            elbo = _TemperedTraceELBO(
                kl_weight=self.kl_weight, num_particles=self.num_particles
            )
        return SVI(self.model_fn, self.guide_fn, opt, loss=elbo)

    def init(self, seed: PRNGKeyArray, *args: Any, **kwargs: Any) -> Any:
        """Initialize the ensemble of SVI states."""
        svi = self._build_svi()
        keys = jr.split(seed, self.ensemble_size)
        return jax.vmap(lambda k: svi.init(k, *args, **kwargs))(keys)

    def update(
        self, state: Any, *args: Any, **kwargs: Any
    ) -> tuple[Any, Float[Array, " E"]]:
        """One ensemble SVI update. Mirrors ``numpyro.infer.SVI.update``."""
        svi = self._build_svi()
        return jax.vmap(lambda s: svi.update(s, *args, **kwargs))(state)

    def run(
        self,
        seed: PRNGKeyArray,
        num_epochs: int,
        *args: Any,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Fit the ensemble end-to-end via vmapped ``svi.run``."""
        svi = self._build_svi()
        keys = jr.split(seed, self.ensemble_size)

        @jax.vmap
        def _run_one(
            k: PRNGKeyArray,
        ) -> tuple[PyTree, Float[Array, " T"]]:
            r = svi.run(k, num_epochs, *args, progress_bar=False, **kwargs)
            return r.params, r.losses

        params, losses = _run_one(keys)
        return EnsembleResult(params=params, losses=losses)


class _TemperedTraceELBO:
    """ELBO with a scalar weight on the KL term.

    Splits the standard ELBO into ``log_lik`` (observed-site log-prob)
    and ``KL = log q(z) - log p(z)``, then recombines as
    ``-(log_lik - kl_weight * KL)``. ``kl_weight=1`` recovers the
    standard ELBO; ``kl_weight<1`` is a cold posterior temper.
    """

    def __init__(self, kl_weight: float, num_particles: int = 1) -> None:
        self.kl_weight = kl_weight
        self.num_particles = num_particles

    def loss(
        self,
        rng_key: PRNGKeyArray,
        param_map: dict,
        model: Callable,
        guide: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        from numpyro.handlers import replay, seed, substitute, trace

        def _per_particle(key: PRNGKeyArray) -> Float[Array, ""]:
            guide_seeded = seed(substitute(guide, param_map), key)
            guide_tr = trace(guide_seeded).get_trace(*args, **kwargs)
            model_seeded = seed(replay(substitute(model, param_map), guide_tr), key)
            model_tr = trace(model_seeded).get_trace(*args, **kwargs)

            log_lik = jnp.zeros(())
            log_p_z = jnp.zeros(())
            for site in model_tr.values():
                if site["type"] != "sample":
                    continue
                lp = site["fn"].log_prob(site["value"]).sum()
                if site.get("is_observed", False):
                    log_lik = log_lik + lp
                else:
                    log_p_z = log_p_z + lp
            log_q_z = jnp.zeros(())
            for site in guide_tr.values():
                if site["type"] != "sample":
                    continue
                log_q_z = log_q_z + site["fn"].log_prob(site["value"]).sum()
            kl = log_q_z - log_p_z
            return -(log_lik - self.kl_weight * kl)

        keys = jr.split(rng_key, self.num_particles)
        return jax.vmap(_per_particle)(keys).mean()


# ============================================================================
# Layer 3 — One-shot sugar
# ============================================================================


def ensemble_map(
    log_joint: Callable[
        [PyTree, Array, Array],
        tuple[Float[Array, ""], Float[Array, ""]],
    ],
    init_fn: Callable[[PRNGKeyArray], PyTree],
    *,
    ensemble_size: int,
    num_epochs: int,
    data: tuple[Array, Array],
    seed: PRNGKeyArray,
    batch_size: int | None = None,
    learning_rate: float = 5e-3,
    prior_weight: float = 1.0,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[PyTree, Float[Array, "E T"]]:
    """One-shot wrapper around :class:`EnsembleMAP`.

    Equivalent to ``EnsembleMAP(log_joint, init_fn, optimizer,
    ensemble_size=E, prior_weight=w).run(seed, num_epochs, *data,
    batch_size=B)``.

    Args:
        log_joint: ``(params, x_batch, y_batch) -> (loglik, logprior)``.
        init_fn: ``key -> params``.
        ensemble_size: Number of independent MAP fits ``E``.
        num_epochs: Optimizer steps per member.
        data: ``(x, y)``.
        seed: PRNG key.
        batch_size: Mini-batch size. ``None`` ⇒ full-batch.
        learning_rate: Default-Adam learning rate. Ignored if
            ``optimizer`` is supplied.
        prior_weight: ``0`` ⇒ MLE, ``1`` ⇒ MAP.
        optimizer: Optional ``optax.GradientTransformation``. Defaults
            to ``optax.adam(learning_rate)``.

    Returns:
        ``(params_stacked, losses)`` — leading ``(E,)`` axis on
        ``params``; ``losses`` shape ``(E, num_epochs)``.

    Example:
        >>> params, losses = ensemble_map(
        ...     log_joint, init_fn,
        ...     ensemble_size=16, num_epochs=2000,
        ...     data=(X, y), seed=jr.PRNGKey(0),
        ... )
    """
    optax = _require_optax()
    opt = optimizer if optimizer is not None else optax.adam(learning_rate)
    runner = EnsembleMAP(
        log_joint=log_joint,
        init_fn=init_fn,
        optimizer=opt,
        ensemble_size=ensemble_size,
        prior_weight=prior_weight,
    )
    result = runner.run(seed, num_epochs, *data, batch_size=batch_size)  # ty: ignore[unresolved-attribute]
    return result.params, result.losses


def ensemble_vi(
    model_fn: Callable[..., None],
    guide_fn: Callable[..., None],
    *,
    ensemble_size: int,
    num_epochs: int,
    data: tuple[Array, Array],
    seed: PRNGKeyArray,
    kl_weight: float = 1.0,
    learning_rate: float = 5e-3,
    optimizer: Any = None,
    num_particles: int = 1,
) -> tuple[PyTree, Float[Array, "E T"]]:
    """One-shot wrapper around :class:`EnsembleVI`.

    Args:
        model_fn: NumPyro model ``(x, y) -> None``.
        guide_fn: NumPyro guide.
        ensemble_size: Number of SVI fits ``E``.
        num_epochs: Steps per member.
        data: ``(x, y)``.
        seed: PRNG key.
        kl_weight: ELBO temper ``β``. ``1.0`` ⇒ standard ELBO.
        learning_rate: Default-Adam learning rate.
        optimizer: Optional ``numpyro.optim`` or
            ``optax.GradientTransformation`` (auto-wrapped).
        num_particles: MC particles per ELBO estimate.

    Returns:
        ``(guide_params_stacked, losses)`` with leading ``(E,)`` axis.
    """
    import numpyro

    if optimizer is None:
        optimizer = numpyro.optim.Adam(learning_rate)
    runner = EnsembleVI(
        model_fn=model_fn,
        guide_fn=guide_fn,
        optimizer=optimizer,
        ensemble_size=ensemble_size,
        kl_weight=kl_weight,
        num_particles=num_particles,
    )
    result = runner.run(seed, num_epochs, *data)  # ty: ignore[unresolved-attribute]
    return result.params, result.losses


def ensemble_predict(
    params_stacked: PyTree,
    predict_fn: Callable[[PyTree, Array], Array],
    x_new: Array,
) -> Array:
    """Vmap ``predict_fn`` over the leading ensemble axis of params.

    Uses :func:`equinox.filter_vmap` so it works whether
    ``params_stacked`` is a pure-array PyTree or an
    :class:`equinox.Module` containing non-array leaves (e.g. captured
    ``jax.nn.tanh``). Array leaves are mapped over axis 0; non-array
    leaves are broadcast.

    Args:
        params_stacked: PyTree returned by :func:`ensemble_map` /
            :func:`ensemble_vi` / :class:`EnsembleMAP.run`; every array
            leaf has a leading ``(E,)`` axis.
        predict_fn: ``(params, x) -> y``.
        x_new: Inputs to predict at; shared across all members.

    Returns:
        Stacked predictions with leading ``(E,)`` axis.
    """
    return eqx.filter_vmap(predict_fn, in_axes=(eqx.if_array(0), None))(
        params_stacked, x_new
    )
