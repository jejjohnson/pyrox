r"""Ragged per-site series, padded to one shape with an observation mask.

A hierarchical model over $S$ sites observed at different times runs one
Markov-GP factor per site under `jax.vmap`, which needs every site on a grid
of the same length $N$. `pad_sites` pads each site's series after its last
observation and returns the mask that the `pyrox_ssm.MarkovGPPrior` methods
take, so the padded steps run the Kalman predict only and the log marginal of
every site is unchanged. `MarkovGPSitesPrior` holds the padded sites as one
prior, with a summed log marginal and a `condition` whose result predicts every
site at any times; `markov_gp_sites_factor` scores all the padded sites as one
factor, and `noncentred_log_normal` draws per-site kernel hyperparameters
around a population value without NUTS's funnel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from gaussx import SDEKernel
from jaxtyping import Array, Bool, Float

from pyrox_ssm._hierarchical import _site_axes
from pyrox_ssm._markov import ConditionedMarkovGP, MarkovGPPrior, _observation_mask


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


class MarkovGPSitesPrior(eqx.Module):
    r"""A Markov GP prior over $S$ padded sites that share one SDE kernel.

    Site $s$ is an independent Markov GP over its own row of ``times``, with
    kernel $\theta_s$ and noise variance $\sigma_s^2$, so the log marginal of
    the sites is the sum of theirs,

    $$
    \log p(Y \mid \theta) = \sum_{s=1}^{S} \log p(y_s \mid \theta_s, \sigma_s^2),
    $$

    each term `pyrox_ssm.MarkovGPPrior.log_marginal` over the site's masked
    padded series. ``times`` and ``mask`` are the ``(S, N)`` arrays that
    `pyrox_ssm.pad_sites` returns; padding after a site's last observation
    adds predict-only steps, so it changes neither the site's log marginal
    nor its predictions.

    One kernel per site is one kernel whose array leaves carry a leading
    ``(S,)`` axis, such as ``MaternSDE(variance, lengthscale, order=1)`` with
    ``(S,)`` arrays; a 0-d leaf is shared by every site. The mean function
    follows the same rule: a plain function, or a module whose array leaves
    are 0-d, is shared, and a module (an ``eqx.Module``) whose array leaves
    carry a leading ``(S,)`` axis gives each site its own mean.
    ``obs_noise_floor`` is shared by every site. The prior registers no
    NumPyro sites; `markov_gp_sites_factor` adds its log marginal as one
    factor.

    Attributes:
        sde_kernel: An `SDEKernel` whose non-scalar array leaves have a
            leading site axis of length ``S``.
        times: ``(S, N)`` padded times, strictly increasing along each row.
        mask: Optional ``(S, N)`` boolean mask, ``True`` where observed.
        mean_fn: Optional mean function ``times -> (N,)``, as in
            `pyrox_ssm.MarkovGPPrior`; shared, or one per site through array
            leaves with a leading ``(S,)`` axis.
        obs_noise_floor: Extra observation variance, as in
            `pyrox_ssm.MarkovGPPrior`.

    Raises:
        ValueError: If ``times`` is not ``(S, N)``, ``mask`` is not shaped
            like it, or a non-scalar array leaf of the kernel or the mean
            function has a leading axis whose length is not ``S``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_ssm as pss
        >>> t, y, mask = pss.pad_sites(
        ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
        ... )
        >>> kernel = pss.MaternSDE(1.0, jnp.array([0.5, 2.0]), order=1)
        >>> prior = pss.MarkovGPSitesPrior(kernel, t, mask=mask)
        >>> log_marg = prior.log_marginal(y, jnp.asarray(0.1))
        >>> cond = prior.condition(y, jnp.asarray(0.1))
        >>> mean, var = cond.predict(jnp.array([1.5, 6.0]))
        >>> mean.shape
        (2, 2)
    """

    sde_kernel: SDEKernel
    times: Float[Array, "S N"]
    mask: Bool[Array, "S N"] | None = None
    mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None
    obs_noise_floor: float = eqx.field(static=True, default=0.0)

    def __init__(
        self,
        sde_kernel: SDEKernel,
        times: Float[Array, "S N"],
        mask: Bool[Array, "S N"] | None = None,
        mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
        obs_noise_floor: float = 0.0,
    ) -> None:
        times = jnp.asarray(times, dtype=jnp.result_type(times, 0.0))
        if times.ndim != 2:
            raise ValueError(
                "times must be (S, N), one row per site, got shape "
                f"{tuple(times.shape)!r}."
            )
        if mask is not None:
            if jnp.shape(mask) != times.shape:
                raise ValueError(
                    f"mask has shape {tuple(jnp.shape(mask))!r} but times has "
                    f"{tuple(times.shape)!r}."
                )
            mask = jnp.asarray(mask).astype(bool)
        # Validate the site axis of the kernel's and the mean's leaves now.
        _site_axes(sde_kernel, times.shape[0])
        _site_axes(eqx.filter(mean_fn, eqx.is_array), times.shape[0])
        if obs_noise_floor < 0:
            raise ValueError(
                f"obs_noise_floor must be non-negative, got {obs_noise_floor!r}"
            )
        self.sde_kernel = sde_kernel
        self.times = times
        self.mask = mask
        self.mean_fn = mean_fn
        self.obs_noise_floor = float(obs_noise_floor)

    @property
    def n_sites(self) -> int:
        """Number of sites $S$."""
        return self.times.shape[0]

    def _mean_parts(self) -> tuple[Any, Any, Any]:
        """The mean's array leaves, their vmap axes, and the rest of it."""
        dynamic, static = eqx.partition(self.mean_fn, eqx.is_array)
        return dynamic, _site_axes(dynamic, self.n_sites), static

    def _site_prior(
        self, kernel_s: SDEKernel, t_s: Float[Array, " N"], mean_s: Any, static: Any
    ) -> MarkovGPPrior:
        mean_fn = None if self.mean_fn is None else eqx.combine(mean_s, static)
        return MarkovGPPrior(
            kernel_s, t_s, mean_fn=mean_fn, obs_noise_floor=self.obs_noise_floor
        )

    def _check_y(self, y: Float[Array, "S N"]) -> Float[Array, "S N"]:
        y = jnp.asarray(y)
        if y.shape != self.times.shape:
            raise ValueError(
                f"y has shape {tuple(y.shape)!r} but times has "
                f"{tuple(self.times.shape)!r}."
            )
        return y

    def _noise(
        self, noise_var: Float[Array, ""] | Float[Array, " S"]
    ) -> tuple[Float[Array, ""] | Float[Array, " S"], int | None]:
        """``noise_var`` as an array and its vmap axis, validated against S."""
        noise_var = jnp.asarray(noise_var)
        if noise_var.ndim not in (0, 1) or (
            noise_var.ndim == 1 and noise_var.shape[0] != self.n_sites
        ):
            raise ValueError(
                f"noise_var must be () or (S,)=({self.n_sites},), got shape "
                f"{tuple(noise_var.shape)!r}."
            )
        return noise_var, None if noise_var.ndim == 0 else 0

    def _site_log_marginals(
        self,
        y: Float[Array, "S N"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
    ) -> Float[Array, " S"]:
        y = self._check_y(y)
        noise_var, noise_axis = self._noise(noise_var)
        mean_dynamic, mean_axes, mean_static = self._mean_parts()

        def site(kernel_s, mean_s, t_s, y_s, m_s, noise_s):
            prior_s = self._site_prior(kernel_s, t_s, mean_s, mean_static)
            return prior_s.log_marginal(y_s, noise_s, mask=m_s)

        in_axes = (
            _site_axes(self.sde_kernel, self.n_sites),
            mean_axes,
            0,
            0,
            None if self.mask is None else 0,
            noise_axis,
        )
        return jax.vmap(site, in_axes=in_axes)(
            self.sde_kernel, mean_dynamic, self.times, y, self.mask, noise_var
        )

    def log_marginal(
        self,
        y: Float[Array, "S N"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
    ) -> Float[Array, ""]:
        r"""Summed log marginal of the sites' observed steps.

        $$
        \log p(Y \mid \theta) = \sum_{s=1}^{S} \log p(y_s \mid \theta_s, \sigma_s^2).
        $$

        Args:
            y: ``(S, N)`` observations; masked entries may be ``NaN``.
            noise_var: Observation noise variance, shared ``()`` or per site
                ``(S,)``.

        Returns:
            The scalar log marginal.

        Raises:
            ValueError: If ``y`` is not shaped like ``times``, or
                ``noise_var`` is neither ``()`` nor ``(S,)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> t, y, mask = pss.pad_sites(
            ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
            ... )
            >>> prior = pss.MarkovGPSitesPrior(pss.MaternSDE(1.0, 1.0), t, mask=mask)
            >>> log_marg = prior.log_marginal(y, jnp.array([0.1, 0.2]))
        """
        return jnp.sum(self._site_log_marginals(y, noise_var))

    def condition(
        self,
        y: Float[Array, "S N"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
    ) -> ConditionedMarkovGPSites:
        """Condition every site on its observations, by its filter and smoother.

        Args:
            y: ``(S, N)`` observations; masked entries may be ``NaN``.
            noise_var: Observation noise variance, shared ``()`` or per site
                ``(S,)``.

        Returns:
            The `ConditionedMarkovGPSites`, holding each site's smoothed
            moments, with ``y``'s masked entries set to zero.

        Raises:
            ValueError: If ``y`` is not shaped like ``times``, or
                ``noise_var`` is neither ``()`` nor ``(S,)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> t, y, mask = pss.pad_sites(
            ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
            ... )
            >>> prior = pss.MarkovGPSitesPrior(pss.MaternSDE(1.0, 1.0), t, mask=mask)
            >>> cond = prior.condition(y, jnp.asarray(0.1))
            >>> cond.smoothed_means.shape
            (2, 5, 1)
        """
        y = self._check_y(y)
        noise_arr, noise_axis = self._noise(noise_var)
        mean_dynamic, mean_axes, mean_static = self._mean_parts()

        def site(kernel_s, mean_s, t_s, y_s, m_s, noise_s):
            prior_s = self._site_prior(kernel_s, t_s, mean_s, mean_static)
            cond = prior_s.condition(y_s, noise_s, mask=m_s)
            return cond.y, cond.smoothed_means, cond.smoothed_covs, cond.log_marginal

        in_axes = (
            _site_axes(self.sde_kernel, self.n_sites),
            mean_axes,
            0,
            0,
            None if self.mask is None else 0,
            noise_axis,
        )
        y_clean, m_smooth, P_smooth, site_lml = jax.vmap(site, in_axes=in_axes)(
            self.sde_kernel, mean_dynamic, self.times, y, self.mask, noise_arr
        )
        return ConditionedMarkovGPSites(
            prior=self,
            y=y_clean,
            noise_var=noise_arr,
            smoothed_means=m_smooth,
            smoothed_covs=P_smooth,
            site_log_marginals=site_lml,
        )


class ConditionedMarkovGPSites(eqx.Module):
    r"""A `MarkovGPSitesPrior` conditioned on every site's observations.

    Returned by `MarkovGPSitesPrior.condition`. Holds each site's smoothed
    moments on its padded grid; `predict` gives every site's marginal mean and
    variance at any times, as `pyrox_ssm.ConditionedMarkovGP.predict` does for
    one site.

    Attributes:
        prior: The originating `MarkovGPSitesPrior`.
        y: ``(S, N)`` observations, with masked entries set to zero.
        noise_var: The noise variance conditioned on, ``()`` or ``(S,)``.
        smoothed_means: ``(S, N, d)`` smoothed state means.
        smoothed_covs: ``(S, N, d, d)`` smoothed state covariances.
        site_log_marginals: ``(S,)`` log marginal of each site.
    """

    prior: MarkovGPSitesPrior
    y: Float[Array, "S N"]
    noise_var: Float[Array, ""] | Float[Array, " S"]
    smoothed_means: Float[Array, "S N d"]
    smoothed_covs: Float[Array, "S N d d"]
    site_log_marginals: Float[Array, " S"]

    @property
    def log_marginal(self) -> Float[Array, ""]:
        """The summed log marginal, equal to `MarkovGPSitesPrior.log_marginal`."""
        return jnp.sum(self.site_log_marginals)

    def predict(
        self,
        t_star: Float[Array, " M"] | Float[Array, "S M"],
        *,
        include_noise: bool = False,
    ) -> tuple[Float[Array, "S M"], Float[Array, "S M"]]:
        r"""Every site's predictive marginals ``(mean, var)`` at ``t_star``.

        Site $s$ is predicted as `pyrox_ssm.ConditionedMarkovGP.predict`
        predicts one site: its filter and smoother rerun over the merged grid
        of its padded times and the test times, with the test times
        unobserved. With ``include_noise=True`` the variance is that of a new
        observation, $\sigma_*^2 + \sigma_s^2$ plus ``obs_noise_floor``.

        Args:
            t_star: Test times, ``(M,)`` for the same times at every site or
                ``(S, M)`` for one row per site.
            include_noise: Add the site's observation variance.

        Returns:
            ``(mean, var)``, each ``(S, M)``.

        Raises:
            ValueError: If ``t_star`` is neither ``(M,)`` nor ``(S, M)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> t, y, mask = pss.pad_sites(
            ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
            ... )
            >>> prior = pss.MarkovGPSitesPrior(pss.MaternSDE(1.0, 1.0), t, mask=mask)
            >>> cond = prior.condition(y, jnp.asarray(0.1))
            >>> mean, var = cond.predict(jnp.linspace(0.0, 6.0, 7))
            >>> mean.shape
            (2, 7)
        """
        prior = self.prior
        t_star = jnp.asarray(t_star)
        if t_star.ndim == 1:
            t_axis = None
        elif t_star.ndim == 2 and t_star.shape[0] == prior.n_sites:
            t_axis = 0
        else:
            raise ValueError(
                f"t_star must be (M,) or (S, M)=({prior.n_sites}, M), got shape "
                f"{tuple(t_star.shape)!r}."
            )
        noise_axis = None if self.noise_var.ndim == 0 else 0
        mean_dynamic, mean_axes, mean_static = prior._mean_parts()

        def site(kernel_s, mean_s, t_s, y_s, m_s, noise_s, m_smooth, P_smooth, lml, ts):
            site_prior = prior._site_prior(kernel_s, t_s, mean_s, mean_static)
            cond = ConditionedMarkovGP(
                prior=site_prior,
                y=y_s,
                noise_var=noise_s,
                smoothed_means=m_smooth,
                smoothed_covs=P_smooth,
                log_marginal=lml,
                mask=None if m_s is None else _observation_mask(m_s, t_s),
            )
            return cond.predict(ts, include_noise=include_noise)

        in_axes = (
            _site_axes(prior.sde_kernel, prior.n_sites),
            mean_axes,
            0,
            0,
            None if prior.mask is None else 0,
            noise_axis,
            0,
            0,
            0,
            t_axis,
        )
        return jax.vmap(site, in_axes=in_axes)(
            prior.sde_kernel,
            mean_dynamic,
            prior.times,
            self.y,
            prior.mask,
            self.noise_var,
            self.smoothed_means,
            self.smoothed_covs,
            self.site_log_marginals,
            t_star,
        )


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
    composes with NUTS and SVI as that pattern does. It registers
    ``MarkovGPSitesPrior(sde_kernel, times, mask=mask, ...).log_marginal(y,
    noise_var)``; build that prior to condition and predict after a fit.

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
        mean_fn: Optional mean function, as in `MarkovGPPrior`; shared, or
            one per site through array leaves with a leading ``(S,)`` axis
            (see `MarkovGPSitesPrior`).
        obs_noise_floor: Extra observation variance, as in `MarkovGPPrior`.

    Raises:
        ValueError: If ``times``, ``y`` or ``mask`` is not ``(S, N)``, or if a
            non-scalar array leaf of the kernel or the mean function, or
            ``noise_var``, has a leading axis whose length is not ``S``.

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
    prior = MarkovGPSitesPrior(
        sde_kernel, times, mask=mask, mean_fn=mean_fn, obs_noise_floor=obs_noise_floor
    )
    numpyro.factor(name, prior.log_marginal(y, noise_var))


def noncentred_log_normal(
    name: str,
    loc: Float[Array, " *batch"] | float,
    scale: Float[Array, " *batch"] | float,
) -> Float[Array, " *batch"]:
    r"""A positive per-site parameter around a population value, non-centred.

    Site $n$'s parameter $\theta_n$ (a lengthscale, a variance) is log-normal
    around a population log-mean $m$ with spread $\tau$,

    $$
    \log \theta_n = m + \tau z_n, \qquad z_n \sim \mathcal{N}(0, 1),
    $$

    which has the distribution of the centred form
    $\log \theta_n \sim \mathcal{N}(m, \tau^2)$. The sampler sees $z_n$, which
    is independent of $\tau$ a priori, instead of $\log \theta_n$, whose
    spread shrinks with $\tau$. That removes the funnel NUTS meets when $\tau$
    is small, the usual case for sites that differ little. NumPyro's
    ``LocScaleReparam(centered=0)`` does the same rewrite on a centred
    ``Normal`` site through a reparam handler; this helper writes it out and
    names its sites predictably.

    Call it inside the ``numpyro.plate`` over sites, so there is one $z_n$ per
    site; ``loc`` and ``scale`` broadcast against the plate.

    Registers the sample site ``f"{name}_z"`` (standard normal) and the
    deterministic site ``name`` ($\theta_n$, so its posterior is in the
    trace).

    Args:
        name: Name of the deterministic site; the sample site is
            ``f"{name}_z"``.
        loc: Population log-mean $m$.
        scale: Population log-spread $\tau > 0$.

    Returns:
        $\theta_n = \exp(m + \tau z_n)$, shaped by the enclosing plates.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> t, y, mask = pss.pad_sites(
        ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
        ... )
        >>> def model():
        ...     loc = numpyro.sample("lengthscale_loc", dist.Normal(jnp.log(7.0), 1.0))
        ...     scale = numpyro.sample("lengthscale_scale", dist.HalfNormal(0.5))
        ...     with numpyro.plate("site", 2):
        ...         ell = pss.noncentred_log_normal("lengthscale", loc, scale)
        ...     kernel = pss.MaternSDE(1.0, ell, order=1)
        ...     noise = jnp.asarray(0.1)
        ...     pss.markov_gp_sites_factor("y", kernel, t, y, noise, mask=mask)
    """
    z = numpyro.sample(f"{name}_z", dist.Normal(0.0, 1.0))
    # log θ_n = m + τ z_n
    log_value = jnp.asarray(loc) + jnp.asarray(scale) * z
    return numpyro.deterministic(  # ty: ignore[invalid-return-type]
        name, jnp.exp(log_value)
    )
