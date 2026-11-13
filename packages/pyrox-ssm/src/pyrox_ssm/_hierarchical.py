r"""A shared GP plus one GP per site, as one state-space model on a merged grid.

$S$ sites observed on their own days often share a regional signal. The
model

$$
y_s(t) = g(t) + h_s(t) + \varepsilon_s(t), \qquad
g \sim \mathcal{GP}(0, k_g), \quad h_s \sim \mathcal{GP}(0, k_h), \quad
\varepsilon_s(t) \sim \mathcal{N}(0, \sigma_{n,s}^2),
$$

with $g$ and every $h_s$ independent, is a linear-Gaussian state-space
model on the union of the site times: the state stacks the SDE states of
$g$ and of each $h_s$, the transitions are block diagonal, and site $s$
observes the sum of the $g$ block and its own block.
`SharedPlusSiteMarkovGP` builds that model as a
`pyrox_ssm.LinearGaussianSSMPrior` with a ``(T, S)`` observation mask, so
the marginal likelihood and the posterior split into $g$ and $h_s$ are
exact Kalman filtering and RTS smoothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from gaussx import SDEKernel
from jaxtyping import Array, ArrayLike, Bool, Float

from pyrox_ssm._lgssm import LinearGaussianSSMPrior
from pyrox_ssm._markov import _build_dt_full, _kernel_initial_covariance


def _site_axes(site: SDEKernel, n_sites: int) -> SDEKernel:
    """The `jax.vmap` axes of the site kernel's leaves, validated against S.

    A scalar leaf is pooled across sites (axis ``None``); any other leaf must
    carry a leading axis of length S and is mapped over it (axis ``0``).
    Shapes are static, so the check also runs on traced leaves.

    Raises:
        ValueError: If a non-scalar leaf's leading axis is not S.
    """

    def axis(leaf: ArrayLike) -> int | None:
        shape = jnp.shape(leaf)
        if len(shape) == 0:
            return None
        if shape[0] != n_sites:
            raise ValueError(
                "every array leaf of the site kernel must be a scalar (pooled "
                f"across sites) or carry a leading axis of length S = {n_sites} "
                f"(one value per site), got a leaf of shape {tuple(shape)!r}"
            )
        return 0

    return jax.tree.map(axis, site)


def _per_site(fn: Callable[[SDEKernel], Any], site: SDEKernel, n_sites: int) -> Any:
    """Evaluate ``fn(site_s)`` for every site, stacked on a leading S axis.

    A pooled site kernel (every leaf scalar) is evaluated once and broadcast;
    otherwise ``fn`` runs under `jax.vmap` over the batched leaves.
    """
    in_axes = _site_axes(site, n_sites)
    if all(a is None for a in jax.tree.leaves(in_axes, is_leaf=lambda a: a is None)):
        out = fn(site)
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n_sites, *x.shape)), out)
    return jax.vmap(fn, in_axes=(in_axes,))(site)


def _stack_block_diag(
    shared: Float[Array, "*T dg dg"], site: Float[Array, "*T S dh dh"]
) -> Float[Array, "*T D D"]:
    r"""Block-diagonal stack $\mathrm{blockdiag}(B_g, B_{h,1}, \dots, B_{h,S})$.

    Args:
        shared: The shared block, ``(*T, d_g, d_g)``.
        site: One block per site, ``(*T, S, d_h, d_h)``.

    Returns:
        The ``(*T, D, D)`` block-diagonal matrices, $D = d_g + S d_h$.
    """
    *batch, n_sites, d_h, _ = site.shape
    d_g = shared.shape[-1]
    eye = jnp.eye(n_sites, dtype=site.dtype)
    # blockdiag(B_{h,1}, …, B_{h,S}) = Σ_s e_s e_sᵀ ⊗ B_{h,s}
    sites = einx.multiply("... s i j, s r -> ... s i r j", site, eye)
    sites = sites.reshape(*batch, n_sites * d_h, n_sites * d_h)
    out = jnp.zeros((*batch, d_g + n_sites * d_h, d_g + n_sites * d_h), site.dtype)
    out = out.at[..., :d_g, :d_g].set(shared)
    return out.at[..., d_g:, d_g:].set(sites)


def _shared_plus_site_rows(
    H_shared: Float[Array, " dg"], H_site: Float[Array, "S dh"]
) -> Float[Array, "S D"]:
    r"""Observation rows $H[s] = (H_g, 0, \dots, H_{h,s}, \dots, 0)$.

    Args:
        H_shared: The shared kernel's observation row, ``(d_g,)``.
        H_site: Each site kernel's observation row, ``(S, d_h)``.

    Returns:
        The ``(S, D)`` observation matrix: row $s$ reads the shared block and
        block $s$ of the site states.
    """
    n_sites = H_site.shape[0]
    eye = jnp.eye(n_sites, dtype=H_site.dtype)
    sites = einx.multiply("s j, s r -> s r j", H_site, eye).reshape(n_sites, -1)
    shared = jnp.broadcast_to(H_shared, (n_sites, H_shared.shape[0]))
    return jnp.concatenate([shared.astype(sites.dtype), sites], axis=-1)


class SharedPlusSiteMarkovGP(eqx.Module):
    r"""A shared GP plus one independent GP per site, on a merged time grid.

    Site $s = 1, \dots, S$ is observed on its own days as

    $$
    y_s(t) = \mu_s(t) + g(t) + h_s(t) + \varepsilon_s(t), \qquad
    g \sim \mathcal{GP}(0, k_g), \quad
    h_s \sim \mathcal{GP}(0, k_{h,s}), \quad
    \varepsilon_s(t) \sim \mathcal{N}(0, \sigma_{n,s}^2),
    $$

    with $g$, $h_1, \dots, h_S$ independent and $\mu_s$ an optional known
    per-site mean (``mean_fn``, zero by default). Every kernel is an SDE kernel
    with state $x_g \in \mathbb{R}^{d_g}$ or $x_{h,s} \in \mathbb{R}^{d_h}$,
    and the joint state on the merged grid $t_1 < \dots < t_T$ stacks them,

    $$
    x = (x_g, x_{h,1}, \dots, x_{h,S}) \in \mathbb{R}^{D}, \qquad
    D = d_g + S\, d_h .
    $$

    `as_lgssm` returns that model as a `pyrox_ssm.LinearGaussianSSMPrior`
    with block-diagonal transitions, process noise and initial covariance,

    $$
    A_k = \mathrm{blockdiag}(A_k^{(g)}, A_k^{(h,1)}, \dots, A_k^{(h,S)}),
    \qquad Q_k \text{ likewise}, \qquad
    P_0 = \mathrm{blockdiag}(P_0^{(g)}, P_0^{(h,1)}, \dots, P_0^{(h,S)}),
    $$

    observation rows $H[s] = (H_g, 0, \dots, H_h \text{ at block } s,
    \dots, 0)$ and $R = \mathrm{diag}(\sigma_{n,1}^2, \dots,
    \sigma_{n,S}^2)$. ``mask[k, s]`` says whether site $s$ was observed at
    $t_k$; an unobserved entry is marginalised out exactly, so the log
    marginal is that of the observed $(t, s)$ pairs under the dense
    Gaussian with covariance $K = K_g \otimes \mathbf{1}\mathbf{1}^\top +
    \mathrm{blockdiag}_s(K_{h,s}) + \mathrm{diag}(\sigma_{n,s}^2)$.

    The site kernel's array leaves are either scalars, shared by every site
    (pooled hyperparameters), or carry a leading axis of length $S$
    (per-site hyperparameters); the two may be mixed, for example a
    per-site variance with a pooled lengthscale. $S$ is ``mask.shape[1]``.
    Each site's discretisation runs once, under `jax.vmap` over the
    batched leaves, or once in total when every leaf is pooled.

    Cost: each filter or smoother pass is $O(T D^3)$, cubic in the number
    of sites. Beyond a few dozen sites $D^3$ dominates. Two cheaper routes
    exist: gaussx's ``meanfield_kalman_filter`` and
    ``meanfield_rts_smoother`` run independent filters on the diagonal
    state blocks at $O(T L d^3)$ for $L$ blocks of size $d$, dropping every
    cross-block covariance; that projection is exact only when the blocks
    do not couple, and here every site couples $g$ to $h_s$ through the
    observation, so it is an approximation that ignores the posterior
    correlation between the shared and the site parts. An ensemble Kalman
    filter (filterax) keeps that correlation at the price of Monte Carlo
    error in the covariances.

    The per-site mean $\mu_s(t)$ (a station climatology, an offset) enters
    `as_lgssm` as the observation offset, so it shifts the observations and
    leaves every covariance unchanged: the log marginal is that of the
    anomalies $y - \mu$, and gradients flow to whatever parameters
    ``mean_fn`` closes over. That is a fixed shift, given its parameters; a
    `pyrox_ssm.ConstantSDE` term in the site kernel instead integrates an
    unknown level out.

    To predict at times no site observed (a gap, a forecast), condition once
    and call ``predict``: ``model.condition(y, noise_var).predict(t_star)``
    adds the test times to the grid as unobserved rows. `from_sites` takes
    the union of the site times with ``jnp.unique``, so it needs concrete
    times and runs outside `jax.jit`. A model rebuilt from sampled kernels
    inside `jax.jit` or a NumPyro model under NUTS uses the constructor with
    a grid and mask computed once outside, for example the ``times`` and
    ``mask`` of one `from_sites` call made before sampling.

    The model registers no NumPyro sites; wrap `log_marginal` in
    ``numpyro.factor`` (or pass `as_lgssm` to `pyrox_ssm.lgssm_factor`).
    Shapes are checked at construction, and concrete ``times`` must be
    strictly increasing (a traced ``times`` is trusted, as in
    `pyrox_ssm.MarkovGPPrior`).

    Attributes:
        shared: The SDE kernel $k_g$ of the shared GP.
        site: The SDE kernel $k_h$ of the site GPs; each array leaf is a
            scalar (pooled) or has a leading axis of length $S$ (per site).
        times: Merged, strictly increasing grid, shaped ``(T,)``.
        mask: Boolean ``(T, S)`` mask, ``True`` where site $s$ was observed
            at ``times[k]``.
        mean_fn: Optional callable mapping times ``(T,)`` to the per-site
            means $\mu_s(t_k)$, ``(T, S)``. ``None`` (the default) is the
            zero mean.

    Examples:
        Three sites on ragged days, a per-site variance and a pooled
        lengthscale:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> t_per_site = [jnp.array([0.0, 1.0, 3.0]), jnp.array([1.0, 2.0]),
        ...               jnp.array([0.0, 2.0, 3.0])]
        >>> y_per_site = [jnp.array([0.2, 0.1, -0.3]), jnp.array([0.4, 0.0]),
        ...               jnp.array([0.1, -0.2, 0.3])]
        >>> shared = pss.MaternSDE(variance=1.0, lengthscale=2.0, order=1)
        >>> site = pss.MaternSDE(variance=jnp.array([0.3, 0.5, 0.2]),
        ...                      lengthscale=0.5, order=1)
        >>> model, y_grid = pss.SharedPlusSiteMarkovGP.from_sites(
        ...     shared, site, t_per_site, y_per_site
        ... )
        >>> y_grid.shape, model.state_dim
        ((4, 3), 8)
        >>> log_marg = model.log_marginal(y_grid, jnp.asarray(0.01))
        >>> g_mean, h_mean = model.decompose(y_grid, jnp.asarray(0.01))
        >>> g_mean.shape, h_mean.shape
        ((4,), (4, 3))

        Inside a NumPyro model, the log marginal is a factor:

        >>> def numpyro_model(y_grid):
        ...     noise_var = numpyro.sample("noise_var", dist.LogNormal(-3.0, 1.0))
        ...     numpyro.factor("y", model.log_marginal(y_grid, noise_var))
    """

    shared: SDEKernel
    site: SDEKernel
    times: Float[Array, " T"]
    mask: Bool[Array, "T S"]
    mean_fn: Callable[[Float[Array, " T"]], Float[Array, "T S"]] | None = None

    def __init__(
        self,
        shared: SDEKernel,
        site: SDEKernel,
        times: Float[Array, " T"],
        mask: Bool[Array, "T S"],
        mean_fn: Callable[[Float[Array, " T"]], Float[Array, "T S"]] | None = None,
    ) -> None:
        times_arr = jnp.asarray(times, dtype=jnp.result_type(times, 0.0))
        if times_arr.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {tuple(times_arr.shape)!r}")
        if times_arr.shape[0] >= 2:
            # Concrete times only; a tracer cannot be checked and is trusted.
            try:
                if not bool(jnp.all(jnp.diff(times_arr) > 0)):
                    raise ValueError("times must be strictly increasing")
            except jax.errors.TracerBoolConversionError:
                pass
        mask_arr = jnp.asarray(mask).astype(bool)
        if mask_arr.ndim != 2 or mask_arr.shape[0] != times_arr.shape[0]:
            raise ValueError(
                f"mask must have shape (T, S) with T = {times_arr.shape[0]} "
                f"steps, got {tuple(mask_arr.shape)!r}"
            )
        if mask_arr.shape[1] == 0:
            raise ValueError("mask must cover at least one site (S >= 1)")
        _site_axes(site, mask_arr.shape[1])
        self.shared = shared
        self.site = site
        self.times = times_arr
        self.mask = mask_arr
        self.mean_fn = mean_fn

    @property
    def n_sites(self) -> int:
        """Number of sites $S$, ``mask.shape[1]``."""
        return self.mask.shape[1]

    @property
    def state_dim(self) -> int:
        """Joint state dimension $D = d_g + S d_h$."""
        return self.shared.state_dim + self.n_sites * self.site.state_dim

    @classmethod
    def from_sites(
        cls,
        shared: SDEKernel,
        site: SDEKernel,
        times_per_site: Sequence[Float[Array, " N_s"]],
        y_per_site: Sequence[Float[Array, " N_s"]],
        mean_fn: Callable[[Float[Array, " T"]], Float[Array, "T S"]] | None = None,
    ) -> tuple[SharedPlusSiteMarkovGP, Float[Array, "T S"]]:
        r"""Merge ragged site series onto one grid and build the model.

        The grid is the sorted union of the site times,
        $\{t_1 < \dots < t_T\} = \bigcup_s \{t_{s,1}, \dots, t_{s,N_s}\}$;
        site $s$'s observations land in column $s$ at their own times, and
        every other entry is ``NaN`` and masked. Where `pyrox_ssm.pad_sites`
        pads each site on its own axis for a vmapped per-site factor, this
        lays every site on the shared axis that the shared GP needs. The
        union is taken with ``jnp.unique``, so the times must be concrete
        (call it outside `jax.jit`).

        Args:
            shared: The SDE kernel $k_g$ of the shared GP.
            site: The SDE kernel $k_h$ of the site GPs (scalar leaves pooled,
                leaves with a leading axis of length $S$ per site).
            times_per_site: One strictly increasing ``(N_s,)`` array of
                observation times per site.
            y_per_site: One ``(N_s,)`` array of observations per site.
            mean_fn: Optional per-site mean, as in the class.

        Returns:
            ``(model, y_grid)``: the model over the merged ``(T,)`` grid with
            its ``(T, S)`` mask, and the ``(T, S)`` observations, ``NaN``
            where a site was not observed.

        Raises:
            ValueError: If no site is given, if the two sequences differ in
                length, if a site is not a non-empty 1-D series matching its
                observations, or if a site's times are not strictly
                increasing.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=0)
            >>> model, y_grid = pss.SharedPlusSiteMarkovGP.from_sites(
            ...     kernel, kernel,
            ...     [jnp.array([0.0, 2.0]), jnp.array([1.0, 2.0])],
            ...     [jnp.array([0.5, 0.1]), jnp.array([0.3, 0.2])],
            ... )
            >>> model.times.tolist(), model.mask.tolist()
            ([0.0, 1.0, 2.0], [[True, False], [False, True], [True, True]])
        """
        if len(times_per_site) != len(y_per_site):
            raise ValueError(
                f"times_per_site has {len(times_per_site)} sites but y_per_site "
                f"has {len(y_per_site)}; pass one series of each per site."
            )
        if len(times_per_site) == 0:
            raise ValueError("from_sites needs at least one site.")
        t_dtype = jnp.result_type(*times_per_site, 0.0)
        y_dtype = jnp.result_type(*y_per_site, 0.0)
        sites = []
        for s, (t_s, y_s) in enumerate(zip(times_per_site, y_per_site, strict=True)):
            t_s = jnp.asarray(t_s, dtype=t_dtype)
            y_s = jnp.asarray(y_s, dtype=y_dtype)
            if t_s.ndim != 1 or t_s.shape[0] == 0:
                raise ValueError(
                    f"site {s}: times must be a non-empty 1-D series, got shape "
                    f"{tuple(t_s.shape)!r}."
                )
            if y_s.shape != t_s.shape:
                raise ValueError(
                    f"site {s}: y has shape {tuple(y_s.shape)!r} but times has "
                    f"{tuple(t_s.shape)!r}; each observation needs one time."
                )
            if t_s.shape[0] >= 2 and not bool(jnp.all(jnp.diff(t_s) > 0)):
                raise ValueError(f"site {s}: times must be strictly increasing.")
            sites.append((t_s, y_s))

        times = jnp.unique(jnp.concatenate([t_s for t_s, _ in sites]))
        n_steps, n_sites = times.shape[0], len(sites)
        y_grid = jnp.full((n_steps, n_sites), jnp.nan, dtype=y_dtype)
        mask = jnp.zeros((n_steps, n_sites), dtype=bool)
        for s, (t_s, y_s) in enumerate(sites):
            rows = jnp.searchsorted(times, t_s)
            y_grid = y_grid.at[rows, s].set(y_s)
            mask = mask.at[rows, s].set(True)
        return cls(shared, site, times, mask, mean_fn), y_grid

    def _noise_vector(
        self, noise_var: Float[Array, ""] | Float[Array, " S"]
    ) -> Float[Array, " S"]:
        """``noise_var`` as the ``(S,)`` per-site observation variances."""
        noise_var = jnp.asarray(noise_var)
        if noise_var.shape not in ((), (self.n_sites,)):
            raise ValueError(
                f"noise_var must be a scalar or have shape ({self.n_sites},), "
                f"one variance per site, got {tuple(noise_var.shape)!r}"
            )
        return jnp.broadcast_to(noise_var, (self.n_sites,))

    def as_lgssm(
        self, noise_var: Float[Array, ""] | Float[Array, " S"]
    ) -> LinearGaussianSSMPrior:
        r"""This model as a `pyrox_ssm.LinearGaussianSSMPrior` over $D$ states.

        With $\Delta t_1 = 0$ and $\Delta t_k = t_k - t_{k-1}$, as in
        `pyrox_ssm.MarkovGPPrior.as_lgssm`,

        $$
        A_k = \mathrm{blockdiag}(A_k^{(g)}, A_k^{(h,1)}, \dots,
        A_k^{(h,S)}), \qquad
        Q_k = \mathrm{blockdiag}(Q_k^{(g)}, Q_k^{(h,1)}, \dots,
        Q_k^{(h,S)}),
        $$

        each block from its kernel's discretisation over $\Delta t_k$, so
        $A_1 = I$ and $Q_1 = 0$ and the first observation sees the initial
        state. $m_0 = 0$, $P_0$ is the block diagonal of each kernel's
        ``initial_covariance()`` ($P_\infty$ for a stationary kernel), the
        time-invariant observation matrix is the ``(S, D)`` stack of rows
        $H[s] = (H_g, 0, \dots, H_h \text{ at block } s, \dots, 0)$ and
        $R = \mathrm{diag}(\sigma_{n,1}^2, \dots, \sigma_{n,S}^2)$. With a
        ``mean_fn``, its ``(T, S)`` values on the grid are the observation
        offset $b_k$.

        Args:
            noise_var: Observation noise variance, a scalar shared by every
                site or one per site, shaped ``(S,)``.

        Returns:
            The `pyrox_ssm.LinearGaussianSSMPrior` with $T$ steps, state
            dimension $D = d_g + S d_h$ and $S$ outputs; filter it with
            ``mask=self.mask``.

        Raises:
            ValueError: If ``noise_var`` is neither a scalar nor ``(S,)``, or
                if ``mean_fn`` does not return ``(T, S)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=1)
            >>> model = pss.SharedPlusSiteMarkovGP(
            ...     kernel, kernel, jnp.arange(5.0), jnp.ones((5, 3), dtype=bool)
            ... )
            >>> lgssm = model.as_lgssm(jnp.array([0.1, 0.2, 0.1]))
            >>> lgssm.transition.shape, lgssm.obs_model.shape
            ((5, 8, 8), (3, 8))
        """
        n_sites = self.n_sites
        dt = _build_dt_full(self.times)
        A_g, Q_g = self.shared.discretise_sequence(dt)
        P_g = _kernel_initial_covariance(self.shared)
        H_g = self.shared.sde_params().H

        def site_model(kernel: SDEKernel):
            A_h, Q_h = kernel.discretise_sequence(dt)
            return A_h, Q_h, _kernel_initial_covariance(kernel), kernel.sde_params().H

        A_h, Q_h, P_h, H_h = _per_site(site_model, self.site, n_sites)
        # (S, T, d, d) → (T, S, d, d): one block per site at every step
        A_h = einx.id("s t i j -> t s i j", A_h)
        Q_h = einx.id("s t i j -> t s i j", Q_h)
        dtype = jnp.result_type(A_g, A_h)

        noise = self._noise_vector(noise_var).astype(dtype)
        init_cov = _stack_block_diag(P_g.astype(dtype), P_h.astype(dtype))
        offset = None if self.mean_fn is None else self.mean_fn(self.times)
        return LinearGaussianSSMPrior(
            transition=_stack_block_diag(A_g.astype(dtype), A_h.astype(dtype)),
            obs_model=_shared_plus_site_rows(H_g[0], H_h[:, 0]).astype(dtype),
            process_noise=_stack_block_diag(Q_g.astype(dtype), Q_h.astype(dtype)),
            obs_noise=jnp.diag(noise),
            init_mean=jnp.zeros(init_cov.shape[0], dtype=dtype),
            init_cov=init_cov,
            obs_offset=offset,
        )

    def log_marginal(
        self,
        y: Float[Array, "T S"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
    ) -> Float[Array, ""]:
        r"""Marginal log-likelihood of the observed $(t, s)$ pairs.

        $$
        \log p(y_{\mathrm{obs}} \mid \theta) = \sum_{k=1}^{T}
          \log \mathcal{N}\!\left(y_{k, o_k};\, H_{o_k} m_k^-,\,
          H_{o_k} P_k^- H_{o_k}^\top + R_{o_k o_k}\right),
        $$

        with $o_k$ the sites observed at $t_k$ (``mask[k]``), equal to the
        dense Gaussian over the observed pairs.

        Args:
            y: Observations, shaped ``(T, S)``; unobserved entries may be
                ``NaN`` and reach neither the value nor a gradient.
            noise_var: Observation noise variance, scalar or ``(S,)``.

        Returns:
            The scalar log marginal.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=0)
            >>> mask = jnp.array([[True, False], [True, True]])
            >>> times = jnp.arange(2.0)
            >>> model = pss.SharedPlusSiteMarkovGP(kernel, kernel, times, mask)
            >>> y = jnp.array([[0.3, jnp.nan], [0.1, -0.2]])
            >>> log_marg = model.log_marginal(y, jnp.asarray(0.1))
        """
        return self.as_lgssm(noise_var).log_marginal(y, mask=self.mask)

    def decompose(
        self,
        y: Float[Array, "T S"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
        *,
        return_var: bool = False,
    ) -> (
        tuple[Float[Array, " T"], Float[Array, "T S"]]
        | tuple[
            Float[Array, " T"],
            Float[Array, "T S"],
            Float[Array, " T"],
            Float[Array, "T S"],
        ]
    ):
        r"""Posterior means, and optionally variances, of the shared and site GPs.

        Runs the Kalman filter and RTS smoother on `as_lgssm` and reads the
        blocks of the smoothed state mean $\hat{x}_k = \mathbb{E}[x_k \mid y]$,

        $$
        \mathbb{E}[g(t_k) \mid y] = H_g \hat{x}_{g,k}, \qquad
        \mathbb{E}[h_s(t_k) \mid y] = H_h \hat{x}_{h,s,k},
        $$

        so the two sum to the posterior mean of site $s$'s signal,
        $\mathbb{E}[g(t_k) + h_s(t_k) \mid y] = H[s] \hat{x}_k$. With
        ``return_var=True`` the marginal variances come from the same
        smoothed state covariance $\hat{P}_k$,

        $$
        \mathbb{V}[g(t_k) \mid y] = H_g \hat{P}_{gg,k} H_g^\top, \qquad
        \mathbb{V}[h_s(t_k) \mid y] = H_h \hat{P}_{h_s h_s,k} H_h^\top .
        $$

        The two parts are correlated a posteriori (one observation informs
        both), so their variances do not add to the variance of
        $g + h_s$, $H[s] \hat{P}_k H[s]^\top$. The RTS smoother solves
        against the predicted state covariance, so a kernel of exactly zero
        variance (a singular block) gives ``NaN`` here; `log_marginal` is unaffected.

        Args:
            y: Observations, shaped ``(T, S)``; unobserved entries may be
                ``NaN``.
            noise_var: Observation noise variance, scalar or ``(S,)``.
            return_var: Also return the marginal variances.

        Returns:
            ``(g_mean, h_mean)``: $\mathbb{E}[g \mid y]$ on the grid,
            ``(T,)``, and $\mathbb{E}[h_s \mid y]$, ``(T, S)``. With
            ``return_var=True``, ``(g_mean, h_mean, g_var, h_var)``, the
            variances shaped like the means.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=0)
            >>> mask = jnp.array([[True, False], [True, True]])
            >>> times = jnp.arange(2.0)
            >>> model = pss.SharedPlusSiteMarkovGP(kernel, kernel, times, mask)
            >>> y = jnp.array([[0.3, jnp.nan], [0.1, -0.2]])
            >>> g_mean, h_mean = model.decompose(y, jnp.asarray(0.1))
            >>> g_mean.shape, h_mean.shape
            ((2,), (2, 2))
        """
        lgssm = self.as_lgssm(noise_var)
        means, covs, _ = lgssm.smooth(y, mask=self.mask)
        d_g, d_h = self.shared.state_dim, self.site.state_dim
        n_steps, n_sites = means.shape[0], self.n_sites
        H = lgssm.obs_model
        # H[s] = (H_g, 0, …, H_h at block s, …, 0): read each block separately
        H_g = H[0, :d_g]
        g_mean = einx.dot("t i, i -> t", means[:, :d_g], H_g)
        site_means = means[:, d_g:].reshape(n_steps, n_sites, d_h)
        H_h = H[:, d_g:].reshape(n_sites, n_sites, d_h)
        H_h = jnp.diagonal(H_h, axis1=0, axis2=1).T  # (S, d_h), row s = H_h,s
        h_mean = einx.dot("t s i, s i -> t s", site_means, H_h)
        if not return_var:
            return g_mean, h_mean
        # V[g] = H_g P̂_gg H_gᵀ
        g_var = einx.dot("t i j, i, j -> t", covs[:, :d_g, :d_g], H_g, H_g)
        # the (d_h, d_h) diagonal block of each site: P̂_{h_s h_s}
        site_cov = covs[:, d_g:, d_g:].reshape(n_steps, n_sites, d_h, n_sites, d_h)
        site_cov = jnp.diagonal(site_cov, axis1=1, axis2=3)  # (T, d_h, d_h, S)
        # V[h_s] = H_h,s P̂_{h_s h_s} H_h,sᵀ
        h_var = einx.dot("t i j s, s i, s j -> t s", site_cov, H_h, H_h)
        return g_mean, h_mean, g_var, h_var

    def condition(
        self,
        y: Float[Array, "T S"],
        noise_var: Float[Array, ""] | Float[Array, " S"],
    ) -> ConditionedSharedPlusSiteMarkovGP:
        r"""Condition on the observations, to predict $g$ and $h_s$ at any times.

        Args:
            y: Observations on the grid, ``(T, S)``; unobserved entries may be
                ``NaN``.
            noise_var: Observation noise variance, scalar or ``(S,)``.

        Returns:
            A `ConditionedSharedPlusSiteMarkovGP`, whose ``predict(t_star)``
            gives the posterior marginals of the shared and site signals.

        Raises:
            ValueError: If ``y`` is not ``(T, S)`` or ``noise_var`` is neither
                a scalar nor ``(S,)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=0)
            >>> mask = jnp.array([[True, False], [True, True]])
            >>> times = jnp.arange(2.0)
            >>> model = pss.SharedPlusSiteMarkovGP(kernel, kernel, times, mask)
            >>> y = jnp.array([[0.3, jnp.nan], [0.1, -0.2]])
            >>> cond = model.condition(y, jnp.asarray(0.1))
            >>> g_mean, h_mean, g_var, h_var = cond.predict(jnp.array([0.5, 3.0]))
        """
        y = jnp.asarray(y)
        if y.shape != self.mask.shape:
            raise ValueError(
                f"y must have the mask's shape {tuple(self.mask.shape)!r}, got "
                f"{tuple(y.shape)!r}"
            )
        noise_var = jnp.asarray(noise_var)
        self._noise_vector(noise_var)  # validates the shape
        return ConditionedSharedPlusSiteMarkovGP(self, y, noise_var)


class ConditionedSharedPlusSiteMarkovGP(eqx.Module):
    r"""A `SharedPlusSiteMarkovGP` conditioned on its observations.

    Returned by `SharedPlusSiteMarkovGP.condition`. `predict` gives the
    posterior marginals of the shared signal $g$ and of each site's signal
    $h_s$ at any times: observed days, gaps no site observed, and forecasts
    or backcasts beyond the grid.

    Attributes:
        model: The conditioned `SharedPlusSiteMarkovGP`.
        y: Observations on the model's grid, ``(T, S)``; unobserved entries
            may be ``NaN``.
        noise_var: Observation noise variance, scalar or ``(S,)``.
    """

    model: SharedPlusSiteMarkovGP
    y: Float[Array, "T S"]
    noise_var: Float[Array, ""] | Float[Array, " S"]

    def predict(
        self, t_star: Float[Array, " M"]
    ) -> tuple[
        Float[Array, " M"], Float[Array, "M S"], Float[Array, " M"], Float[Array, "M S"]
    ]:
        r"""Posterior marginals of $g$ and every $h_s$ at the test times.

        The test times join the grid as rows with no observed site, the
        sorted union $t_{1:T} \cup t^*_{1:M}$, and one filter and smoother
        pass over that grid gives `SharedPlusSiteMarkovGP.decompose`'s means
        and variances at every row; the test rows are read off. An
        unobserved row runs the Kalman predict only, so the log marginal and
        the posterior at the observed rows are unchanged, and a test time
        equal to a grid time sees that time's observations. Cost is
        $O((T + M) D^3)$. The test times need not be sorted or distinct, and
        the method traces, so it runs under `jax.jit` given a model built
        with the constructor.

        Args:
            t_star: Test times, ``(M,)``.

        Returns:
            ``(g_mean, h_mean, g_var, h_var)`` at ``t_star``: the shared
            signal's mean and variance, ``(M,)`` each, and each site's,
            ``(M, S)`` each.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> kernel = pss.MaternSDE(variance=1.0, lengthscale=1.0, order=0)
            >>> mask = jnp.array([[True, False], [True, True]])
            >>> times = jnp.arange(2.0)
            >>> model = pss.SharedPlusSiteMarkovGP(kernel, kernel, times, mask)
            >>> y = jnp.array([[0.3, jnp.nan], [0.1, -0.2]])
            >>> cond = model.condition(y, jnp.asarray(0.1))
            >>> g_mean, h_mean, g_var, h_var = cond.predict(jnp.array([0.5, 3.0]))
            >>> g_mean.shape, h_mean.shape
            ((2,), (2, 2))
        """
        model = self.model
        times = model.times
        t_star = jnp.asarray(t_star, dtype=times.dtype)
        if t_star.ndim != 1:
            raise ValueError(f"t_star must be 1-D, got shape {tuple(t_star.shape)!r}")
        n_steps, n_test, n_sites = times.shape[0], t_star.shape[0], model.n_sites
        # Stable sort: a test time equal to a grid time sorts after it, so it
        # sees that step's update.
        order = jnp.argsort(jnp.concatenate([times, t_star]), stable=True)
        merged_times = jnp.concatenate([times, t_star])[order]
        no_obs = jnp.zeros((n_test, n_sites), dtype=bool)
        merged_mask = jnp.concatenate([model.mask, no_obs])[order]
        merged_y = jnp.concatenate(
            [self.y, jnp.zeros((n_test, n_sites), dtype=self.y.dtype)]
        )[order]
        # tree_at skips __init__, whose strictly-increasing check a test time
        # equal to a grid time would fail; a zero step is A = I, Q = 0.
        merged = eqx.tree_at(
            lambda m: (m.times, m.mask), model, (merged_times, merged_mask)
        )
        g_mean, h_mean, g_var, h_var = merged.decompose(
            merged_y, self.noise_var, return_var=True
        )
        test_rows = jnp.argsort(order, stable=True)[n_steps:]
        return g_mean[test_rows], h_mean[test_rows], g_var[test_rows], h_var[test_rows]
