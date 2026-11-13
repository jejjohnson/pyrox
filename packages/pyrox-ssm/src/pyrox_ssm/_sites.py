r"""Ragged per-site series, padded to one shape with an observation mask.

A hierarchical model over $S$ sites observed at different times runs one
Markov-GP factor per site under `jax.vmap`, which needs every site on a grid
of the same length $N$. `pad_sites` pads each site's series after its last
observation and returns the mask that the `pyrox_ssm.MarkovGPPrior` methods
take, so the padded steps run the Kalman predict only and the log marginal of
every site is unchanged. `markov_gp_sites_factor` then scores all the padded
sites as one factor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpyro
from gaussx import SDEKernel
from jaxtyping import Array, Bool, Float

from pyrox_ssm._hierarchical import _site_axes
from pyrox_ssm._markov import MarkovGPPrior


def pad_sites(
    times: Sequence[Float[Array, " N_s"]],
    ys: Sequence[Float[Array, " N_s"]],
    *,
    masks: Sequence[Bool[Array, " N_s"]] | None = None,
    length: int | None = None,
) -> tuple[Float[Array, "S N"], Float[Array, "S N"], Bool[Array, "S N"]]:
    r"""Pad ragged site series to ``(S, N)`` with strictly increasing padded times.

    Site $s$ with $N_s$ observations at times $t_1 < \dots < t_{N_s}$ is
    extended to length $N$ by stepping on from its last observation at its
    median spacing,

    $$
    t_{N_s + j} = t_{N_s} + j\, \delta_s, \qquad
    \delta_s = \operatorname{median}(\operatorname{diff}(t_s)), \qquad
    j = 1, \dots, N - N_s,
    $$

    with $\delta_s = 1$ for a site with one observation. The padded steps
    are masked, and a masked step runs the Kalman predict only,

    $$
    \mu_k^+ = \mu_k^-, \qquad P_k^+ = P_k^-,
    $$

    so it adds nothing to the log marginal: padding after the last
    observation changes no site's log marginal or smoothed moments at its
    observed steps. The padded times stay strictly increasing, so the
    monotonicity check and the kernel discretisation are unchanged. Padded
    entries of ``y`` are zero.

    Missing steps inside a record are masked the same way, so a record on a
    regular grid with gaps can be passed as it is: a step is masked where
    ``masks`` is ``False`` or, by default, where ``y`` is ``NaN``. A masked
    step runs the predict step only, as a padded one does, so the site's log
    marginal equals that of the same site with those steps dropped. Masked
    entries of ``y`` are zero, as padded ones are.

    Args:
        times: One strictly increasing ``(N_s,)`` array of observation times
            per site.
        ys: One ``(N_s,)`` array of observations per site, matching
            ``times``. ``NaN`` marks a missing step when ``masks`` is not
            given.
        masks: Optional ``(N_s,)`` boolean array per site, ``True`` where the
            step is observed. Defaults to ``~jnp.isnan(y)`` per site.
        length: Padded length $N$. Defaults to the longest site,
            $\max_s N_s$.

    Returns:
        Tuple ``(t, y, mask)``, each of shape ``(S, N)``: the padded times,
        the padded observations and a boolean mask that is ``True`` at each
        row's observed steps: its first $N_s$ steps less any masked ones.

    Raises:
        ValueError: If no site is given, if ``times``, ``ys`` and ``masks``
            differ in length or shape, if a site is not a non-empty 1-D series, if a
            site's (concrete) times are not strictly increasing, or if
            ``length`` is shorter than the longest site.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import pyrox_ssm as pss
        >>> times = [jnp.array([0.0, 1.0, 3.0]), jnp.array([0.5, 1.0])]
        >>> ys = [jnp.array([0.2, 0.1, -0.3]), jnp.array([1.0, 0.8])]
        >>> t, y, mask = pss.pad_sites(times, ys)
        >>> t.tolist()
        [[0.0, 1.0, 3.0], [0.5, 1.0, 1.5]]
        >>> mask.tolist()
        [[True, True, True], [True, True, False]]
        >>> def site_log_marginal(t_s, y_s, m_s):
        ...     prior = pss.MarkovGPPrior(pss.MaternSDE(1.0, 1.0, order=1), t_s)
        ...     return prior.log_marginal(y_s, jnp.asarray(0.1), mask=m_s)
        >>> jax.vmap(site_log_marginal)(t, y, mask).shape
        (2,)
    """
    if len(times) != len(ys):
        raise ValueError(
            f"times has {len(times)} sites but ys has {len(ys)}; pass one "
            "series of each per site."
        )
    if len(times) == 0:
        raise ValueError("pad_sites needs at least one site.")
    if masks is not None and len(masks) != len(times):
        raise ValueError(
            f"masks has {len(masks)} sites but times has {len(times)}; pass "
            "one mask per site."
        )
    t_dtype = jnp.result_type(*times, 0.0)
    y_dtype = jnp.result_type(*ys)
    sites = []
    site_masks = [None] * len(times) if masks is None else list(masks)
    for s, (t_s, y_s, m_s) in enumerate(zip(times, ys, site_masks, strict=True)):
        t_s = jnp.asarray(t_s, dtype=t_dtype)
        y_s = jnp.asarray(y_s, dtype=y_dtype)
        if t_s.ndim != 1 or t_s.shape[0] == 0:
            raise ValueError(
                f"site {s}: times must be a non-empty 1-D series, got shape "
                f"{tuple(t_s.shape)!r}."
            )
        if y_s.shape != t_s.shape:
            raise ValueError(
                f"site {s}: ys has shape {tuple(y_s.shape)!r} but times has "
                f"{tuple(t_s.shape)!r}; each observation needs one time."
            )
        m_s = ~jnp.isnan(y_s) if m_s is None else jnp.asarray(m_s, dtype=bool)
        if m_s.shape != t_s.shape:
            raise ValueError(
                f"site {s}: masks has shape {tuple(m_s.shape)!r} but times has "
                f"{tuple(t_s.shape)!r}; pass one flag per observation."
            )
        if t_s.shape[0] >= 2:
            # Concrete times only; a tracer cannot be checked and is trusted,
            # as in MarkovGPPrior.
            try:
                if not bool(jnp.all(jnp.diff(t_s) > 0)):
                    raise ValueError(f"site {s}: times must be strictly increasing.")
            except jax.errors.TracerBoolConversionError:
                pass
        sites.append((t_s, jnp.where(m_s, y_s, 0), m_s))

    longest = max(t_s.shape[0] for t_s, _, _ in sites)
    n = longest if length is None else int(length)
    if n < longest:
        raise ValueError(
            f"length={n} is shorter than the longest site ({longest} "
            "observations); padding cannot drop observations."
        )

    t_rows, y_rows, mask_rows = [], [], []
    for t_s, y_s, m_s in sites:
        n_s = t_s.shape[0]
        # δₛ = median(diff(tₛ)), one time unit for a single observation
        step = jnp.median(jnp.diff(t_s)) if n_s >= 2 else jnp.ones((), dtype=t_dtype)
        # t_{Nₛ + j} = t_{Nₛ} + j δₛ,  j = 1, …, N − Nₛ
        padded_times = t_s[-1] + step * jnp.arange(1, n - n_s + 1, dtype=t_dtype)
        t_rows.append(jnp.concatenate([t_s, padded_times]))
        y_rows.append(jnp.concatenate([y_s, jnp.zeros(n - n_s, dtype=y_dtype)]))
        mask_rows.append(jnp.concatenate([m_s, jnp.zeros(n - n_s, dtype=bool)]))
    return jnp.stack(t_rows), jnp.stack(y_rows), jnp.stack(mask_rows)


def markov_gp_sites_factor(
    name: str,
    sde_kernel: SDEKernel,
    times: Float[Array, "S N"],
    y: Float[Array, "S N"],
    noise_var: Float[Array, ""] | Float[Array, " S"],
    *,
    mask: Bool[Array, "S N"] | None = None,
    mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    obs_noise_floor: float = 0.0,
) -> None:
    r"""Register the summed Markov-GP log marginal of ``S`` sites as one factor.

    The multi-site form of `markov_gp_factor`, for the ``(S, N)`` padded
    times, observations and mask that `pyrox_ssm.pad_sites` returns. Site
    $s$ has its own kernel $\theta_s$ and noise variance $\sigma_s^2$, and the
    sites are independent given them, so the factor is

    $$
    \log p(Y \mid \theta) = \sum_{s=1}^{S} \log p(y_s \mid \theta_s, \sigma_s^2),
    $$

    each term the masked Kalman log marginal of `MarkovGPPrior.log_marginal`.
    It replaces the hand-written ``jax.vmap`` over sites shown in
    `markov_gp_factor`'s example, and is one ``numpyro.factor`` site, so it
    composes with NUTS and SVI as that pattern does.

    One kernel per site is one kernel whose array leaves carry a leading
    ``(S,)`` axis, such as ``MaternSDE(variance, lengthscale, order=1)`` with
    ``(S,)`` arrays sampled under a ``numpyro.plate`` over sites. A 0-d leaf
    (a Python float or a scalar array) is shared by every site.

    Args:
        name: The factor site name.
        sde_kernel: An `SDEKernel` whose non-scalar array leaves have a
            leading site axis of length ``S``.
        times: ``(S, N)`` strictly increasing times per row.
        y: ``(S, N)`` observations; masked entries may be ``NaN``.
        noise_var: Observation noise variance, shared ``()`` or per site
            ``(S,)``.
        mask: Optional ``(S, N)`` boolean mask, ``True`` where observed.
        mean_fn: Optional mean function shared by every site, as in
            `MarkovGPPrior`.
        obs_noise_floor: Extra observation variance, as in `MarkovGPPrior`.

    Raises:
        ValueError: If ``times``, ``y`` or ``mask`` is not ``(S, N)``, or if a
            non-scalar kernel leaf or ``noise_var`` has a leading axis whose
            length is not ``S``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> t, y, mask = pss.pad_sites(
        ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
        ... )
        >>> def model():
        ...     with numpyro.plate("site", 2):
        ...         ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        ...     kernel = pss.MaternSDE(1.0, ell, order=1)
        ...     noise = jnp.asarray(0.1)
        ...     pss.markov_gp_sites_factor("y", kernel, t, y, noise, mask=mask)
    """
    times = jnp.asarray(times)
    if times.ndim != 2:
        raise ValueError(
            f"times must be (S, N), one row per site, got shape {tuple(times.shape)!r}."
        )
    n_sites = times.shape[0]
    y = jnp.asarray(y)
    if y.shape != times.shape:
        raise ValueError(
            f"y has shape {tuple(y.shape)!r} but times has {tuple(times.shape)!r}."
        )
    if mask is not None and jnp.shape(mask) != times.shape:
        raise ValueError(
            f"mask has shape {tuple(jnp.shape(mask))!r} but times has "
            f"{tuple(times.shape)!r}."
        )

    kernel_axes = _site_axes(sde_kernel, n_sites)
    noise_var = jnp.asarray(noise_var)
    if noise_var.ndim not in (0, 1) or (
        noise_var.ndim == 1 and noise_var.shape[0] != n_sites
    ):
        raise ValueError(
            f"noise_var must be () or (S,)=({n_sites},), got shape "
            f"{tuple(noise_var.shape)!r}."
        )
    noise_axis = None if noise_var.ndim == 0 else 0

    def _site(
        kernel_s: SDEKernel,
        t_s: Float[Array, " N"],
        y_s: Float[Array, " N"],
        m_s: Bool[Array, " N"] | None,
        noise_s: Float[Array, ""],
    ) -> Float[Array, ""]:
        prior = MarkovGPPrior(
            kernel_s, t_s, mean_fn=mean_fn, obs_noise_floor=obs_noise_floor
        )
        return prior.log_marginal(y_s, noise_s, mask=m_s)

    mask_axis = None if mask is None else 0
    log_marginals = jax.vmap(_site, in_axes=(kernel_axes, 0, 0, mask_axis, noise_axis))(
        sde_kernel, times, y, mask, noise_var
    )
    numpyro.factor(name, jnp.sum(log_marginals))
