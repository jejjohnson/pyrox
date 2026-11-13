r"""Ragged per-site series, padded to one shape with an observation mask.

A hierarchical model over $S$ sites observed at different times runs one
Markov-GP factor per site under `jax.vmap`, which needs every site on a grid
of the same length $N$. `pad_sites` pads each site's series after its last
observation and returns the mask that the `pyrox_ssm.MarkovGPPrior` methods
take, so the padded steps run the Kalman predict only and the log marginal of
every site is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


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
