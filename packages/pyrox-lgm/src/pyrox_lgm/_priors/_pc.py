r"""Penalised-complexity (PC) priors (Simpson et al., 2017).

A PC prior measures how far a component is from a simpler *base model*
$\xi_0$ (no effect, no spatial structure, infinite range) by the distance
$d(\xi) = \sqrt{2\,\mathrm{KLD}(\pi_\xi\,\|\,\pi_{\xi_0})}$ and puts an
exponential prior on that distance:

$$
\pi(\xi) = \lambda\,e^{-\lambda d(\xi)}\,\Big|\frac{\partial d}{\partial\xi}\Big|,
$$

with the rate $\lambda$ fixed by a tail statement that a user can reason
about ("$P(\sigma > U) = \alpha$"). All priors here are NumPyro
distributions, so they work as hyperpriors under NUTS and in ``inla()``.

- `PCPrecision`: precision $\tau$, base $\sigma = 0$,
  $P(\sigma > U) = \alpha$.
- `PCAR1Rho`: AR(1) correlation, base $\rho = 0$,
  $P(\lvert\rho\rvert > U) = \alpha$.
- `PCBYM2Phi`: BYM2 mixing $\phi$, base $\phi = 0$ (no structure),
  $P(\phi < U) = \alpha$.
- `PCMatern`: Matérn range and sd, base infinite range and $\sigma = 0$,
  $P(\rho < \rho_0) = \alpha_\rho$ and $P(\sigma > \sigma_0) = \alpha_\sigma$.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.decomp
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample


__all__ = [
    "PCAR1Rho",
    "PCBYM2Phi",
    "PCMatern",
    "PCPrecision",
    "StructureSpectrum",
    "structure_spectrum",
]


# ---------------------------------------------------------------------------
# PCPrecision
# ---------------------------------------------------------------------------


class PCPrecision(dist.Distribution):
    r"""PC prior on the precision $\tau$ of a Gaussian effect.

    The distance to the base model $\sigma = 0$ is proportional to the
    standard deviation $\sigma = \tau^{-1/2}$, so $\sigma \sim
    \mathrm{Exp}(\lambda)$ with $\lambda = -\log\alpha / U$, i.e.
    $P(\sigma > U) = \alpha$. On $\tau$:

    $$
    \pi(\tau) = \tfrac{\lambda}{2}\,\tau^{-3/2}\,e^{-\lambda\tau^{-1/2}}.
    $$

    This is R-INLA's ``pc.prec``.

    Args:
        U: Upper reference value for $\sigma$, ``U > 0``.
        alpha: Tail probability $P(\sigma > U)$, in ``(0, 1)``.

    Examples:
        >>> import jax
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.PCPrecision(U=1.0, alpha=0.01)  # P(sigma > 1) = 0.01
        >>> tau = prior.sample(jax.random.key(0), (20_000,))
        >>> round(float((tau < 1.0).mean()), 2)  # sigma > 1  <=>  tau < 1
        0.01
    """

    arg_constraints = {  # noqa: RUF012 (NumPyro's class-level contract)
        "U": constraints.positive,
        "alpha": constraints.open_interval(0.0, 1.0),
    }
    support = constraints.positive
    reparametrized_params = ["U", "alpha"]  # noqa: RUF012

    def __init__(
        self,
        U: ArrayLike = 1.0,
        alpha: ArrayLike = 0.01,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.U, self.alpha = promote_shapes(U, alpha)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(U), jnp.shape(alpha))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def rate(self) -> Array:
        """The rate $\\lambda = -\\log\\alpha / U$ of the exponential on $\\sigma$."""
        return -jnp.log(self.alpha) / self.U

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        sigma = jax.random.exponential(key, shape) / self.rate
        return sigma**-2.0

    @validate_sample
    def log_prob(self, value):
        lam = self.rate
        return jnp.log(lam / 2.0) - 1.5 * jnp.log(value) - lam * jax.lax.rsqrt(value)

    def cdf(self, value):
        # P(tau <= t) = P(sigma >= t^{-1/2}) = exp(-lam t^{-1/2}).
        return jnp.exp(-self.rate * jax.lax.rsqrt(value))

    def icdf(self, q):
        return (-jnp.log(q) / self.rate) ** -2.0


# ---------------------------------------------------------------------------
# PCAR1Rho
# ---------------------------------------------------------------------------


def _safe_ratio_rho_over_d(rho: Array) -> Array:
    """``|rho| / d(rho)`` with ``d = sqrt(-log(1 - rho^2))``, finite at 0.

    The ratio tends to 1 as rho -> 0. The double ``where`` keeps both the
    value and its gradient finite there.
    """
    r2 = rho**2
    small = r2 < 1e-8
    r2_safe = jnp.where(small, 0.5, r2)
    exact = jnp.sqrt(r2_safe / -jnp.log1p(-r2_safe))
    # Series: r^2 / -log(1 - r^2) = 1 - r^2/2 - r^4/12 + O(r^6).
    series = jnp.sqrt(1.0 - r2 / 2.0)
    return jnp.where(small, series, exact)


class PCAR1Rho(dist.Distribution):
    r"""PC prior on the lag-one correlation $\rho$ of an AR(1) process.

    Base model $\rho = 0$ (Sørbye & Rue, 2017): the distance is
    $d(\rho) = \sqrt{-\log(1-\rho^2)}$, and the prior is symmetric,

    $$
    \pi(\rho) = \frac{\lambda}{2}\,e^{-\lambda d(\rho)}\,
        \frac{\lvert\rho\rvert}{(1-\rho^2)\,d(\rho)},\qquad
    \lambda = -\log\alpha / d(U),
    $$

    so that $P(\lvert\rho\rvert > U) = \alpha$. This is R-INLA's ``pc.cor0``
    (which states its calibration on $\lvert\rho\rvert$).

    Args:
        U: Reference correlation, in ``(0, 1)``.
        alpha: Tail probability $P(\lvert\rho\rvert > U)$, in ``(0, 1)``.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.PCAR1Rho(U=0.5, alpha=0.1)  # P(|rho| > 0.5) = 0.1
        >>> rho = prior.sample(jax.random.key(0), (20_000,))
        >>> round(float((jnp.abs(rho) > 0.5).mean()), 2)
        0.1
    """

    arg_constraints = {  # noqa: RUF012 (NumPyro's class-level contract)
        "U": constraints.open_interval(0.0, 1.0),
        "alpha": constraints.open_interval(0.0, 1.0),
    }
    support = constraints.open_interval(-1.0, 1.0)
    reparametrized_params = ["U", "alpha"]  # noqa: RUF012

    def __init__(
        self,
        U: ArrayLike = 0.5,
        alpha: ArrayLike = 0.5,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.U, self.alpha = promote_shapes(U, alpha)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(U), jnp.shape(alpha))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def rate(self) -> Array:
        """The rate $\\lambda = -\\log\\alpha / d(U)$."""
        return -jnp.log(self.alpha) / jnp.sqrt(-jnp.log1p(-(self.U**2)))

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        k_d, k_sign = jax.random.split(key)
        d = jax.random.exponential(k_d, shape) / self.rate
        magnitude = jnp.sqrt(-jnp.expm1(-(d**2)))
        sign = jnp.where(jax.random.bernoulli(k_sign, 0.5, shape), 1.0, -1.0)
        return sign * magnitude

    @validate_sample
    def log_prob(self, value):
        lam = self.rate
        ratio = _safe_ratio_rho_over_d(value)
        # d = |rho| / ratio: unlike sqrt(-log1p(-rho^2)) it has a finite
        # gradient at rho = 0.
        d = jnp.abs(value) / ratio
        return jnp.log(lam / 2.0) - lam * d + jnp.log(ratio) - jnp.log1p(-(value**2))


# ---------------------------------------------------------------------------
# Structure spectra (for PCBYM2Phi)
# ---------------------------------------------------------------------------


class StructureSpectrum(NamedTuple):
    r"""Spectral measure of $R_\ast^{+}$, the generalised inverse of a structure.

    ``eigenvalues`` are the eigenvalues $\gamma_i$ of $R_\ast^{+}$ (zero on
    the null space of $R_\ast$) and ``weights`` their multiplicities, which
    sum to $n$. A dense spectrum has unit weights; a Lanczos (SLQ) spectrum
    has quadrature nodes and weights. Anything linear in the spectrum
    ($\operatorname{tr}R_\ast^{+}$, $\log|(1-\phi)I + \phi R_\ast^{+}|$) is a
    weighted sum over it.
    """

    eigenvalues: Float[Array, " m"]
    weights: Float[Array, " m"]

    @property
    def size(self) -> Array:
        """The dimension $n$, the total weight."""
        return jnp.sum(self.weights)


def _dense_eigvals(operator: lx.AbstractLinearOperator) -> Float[Array, " n"]:
    """Eigenvalues of a symmetric operator, exactly from Kronecker-sum factors."""
    if isinstance(operator, gx.KroneckerSum):
        a = _dense_eigvals(operator.A)
        b = _dense_eigvals(operator.B)
        return jnp.ravel(a[:, None] + b[None, :])
    return jnp.linalg.eigvalsh(operator.as_matrix())


def _as_columns(null_space: ArrayLike) -> Float[Array, "n c"]:
    V = jnp.asarray(null_space)
    return V[:, None] if V.ndim == 1 else V


def structure_spectrum(
    structure: lx.AbstractLinearOperator,
    null_space: ArrayLike | None = None,
    *,
    method: Literal["auto", "dense", "lanczos"] = "auto",
    num_probes: int = 64,
    order: int = 50,
    deflate: int = 32,
    key: jax.Array | None = None,
) -> StructureSpectrum:
    r"""The spectrum of $R^{+}$ that `PCBYM2Phi` needs, computed once.

    - ``"dense"``: exact eigenvalues. A `gaussx.KroneckerSum` (the structure
      of a `kernellib.GridGraph`) is diagonalised factor by factor, so a grid
      costs only its 1-D eigenproblems; anything else is a dense ``eigh``.
    - ``"lanczos"``: a deflated stochastic Lanczos quadrature (SLQ) estimate.
      One Lanczos run of ``max(order, 8 * deflate)`` steps converges up to
      ``deflate`` of the largest eigenvalues of $R_\ast^{+}$ (the smallest
      of $R_\ast$), which are kept exactly: they dominate
      $\operatorname{tr}R_\ast^{+}$, and leaving them to the probes gives a
      large variance. The rest of the measure comes from ``num_probes``
      Rademacher probes, projected off the null space and the deflated
      eigenvectors, with ``order`` Lanczos steps each (full
      reorthogonalisation); the null space contributes exact zeros. Cost:
      about ``num_probes * order + 8 * deflate`` matvecs. Runs eagerly (it picks the
      converged Ritz pairs), so call it once, outside ``jit``.
    - ``"auto"``: ``"dense"`` for a `gaussx.KroneckerSum` or ``n <= 5000``,
      else ``"lanczos"``.

    Args:
        structure: The (scaled) structure matrix $R_\ast$, symmetric PSD.
        null_space: Orthonormal basis of its null space, ``(n, c)`` or
            ``(n,)``; ``None`` if it has none. Its columns are the
            eigenvalues reported as zero.
        method: See above.
        num_probes: SLQ probes (``"lanczos"`` only).
        order: Lanczos steps per probe (``"lanczos"`` only).
        deflate: Most eigenvalues to deflate exactly (``"lanczos"`` only).
        key: PRNG key for the probes; ``None`` means ``jax.random.key(0)``.

    Returns:
        The spectral measure of $R_\ast^{+}$.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> g = kl.grid_graph((4,))
        >>> R = kl.structure_matrix(g, scaled=True)
        >>> spec = lgm.structure_spectrum(R, kl.graph_null_space(g))
        >>> float(spec.size)
        4.0
        >>> int(jnp.sum(spec.eigenvalues == 0.0))  # one connected component
        1
    """
    n = structure.in_size()
    V = None if null_space is None else _as_columns(null_space)
    c = 0 if V is None else V.shape[1]
    if method == "auto":
        dense = isinstance(structure, gx.KroneckerSum) or n <= 5000
        method = "dense" if dense else "lanczos"
    if method == "dense":
        lam = jnp.sort(_dense_eigvals(structure))
        is_null = jnp.arange(n) < c  # the c smallest are the null space
        safe = jnp.where(is_null, 1.0, lam)
        gamma = jnp.where(is_null, 0.0, 1.0 / safe)
        return StructureSpectrum(gamma, jnp.ones(n, dtype=gamma.dtype))
    if method != "lanczos":
        raise ValueError(f"method must be 'auto', 'dense' or 'lanczos', got {method!r}")
    return _lanczos_spectrum(structure, V, n, c, num_probes, order, deflate, key)


def _lanczos_spectrum(structure, V, n, c, num_probes, order, deflate, key):
    key = jax.random.key(0) if key is None else key
    k_defl, k_probe = jax.random.split(key)
    dtype = jnp.result_type(float)

    def projector(basis):
        def project(v):
            return v if basis is None else v - basis @ (basis.T @ v)

        return project

    project = projector(V)

    def matvec(v):
        return project(structure.mv(project(v)))

    # 1. Deflation: one long Lanczos run converges the smallest eigenvalues of
    # R (the largest of R+, which dominate its trace and would otherwise give
    # the probes a large variance). Keep the converged Ritz pairs as exact.
    exact = jnp.zeros(0, dtype=dtype)
    W = V
    m = min(max(order, 8 * deflate), n - c)
    if deflate and m > 0:
        start = project(jax.random.normal(k_defl, (n,), dtype=dtype))
        out = matfree.decomp.tridiag_sym(m, reortho="full")(matvec, start)
        theta, S = jnp.linalg.eigh(out.J_small)
        resid = jnp.linalg.norm(out.residual) * jnp.abs(S[-1, :])
        scale = jnp.max(jnp.abs(theta))
        # Converged, and not a null-space direction leaking back in rounding.
        converged = (resid <= 1e-8 * scale) & (theta > 1e-8 * scale)
        idx = [int(i) for i in jnp.flatnonzero(converged)][:deflate]  # ascending
        if idx:
            Y = out.Q_tall.T @ S[:, jnp.asarray(idx)]
            exact = 1.0 / theta[jnp.asarray(idx)]
            W = Y if V is None else jnp.concatenate([V, Y], axis=1)
            W, _ = jnp.linalg.qr(W)
    k = int(exact.shape[0])
    rest = n - c - k

    # 2. SLQ on the remainder, orthogonal to the null space and the deflated
    # eigenvectors.
    nodes = jnp.zeros(0, dtype=dtype)
    weights = jnp.zeros(0, dtype=dtype)
    if rest > 0:
        project_rest = projector(W)

        def matvec_rest(v):
            return project_rest(structure.mv(project_rest(v)))

        tridiag = matfree.decomp.tridiag_sym(min(order, rest), reortho="full")

        def one_probe(kp):
            z = project_rest(jax.random.rademacher(kp, (n,), dtype=dtype))
            J = tridiag(matvec_rest, z).J_small
            th, U = jnp.linalg.eigh(J)
            # Drop Ritz values at rounding level: directions leaking back from
            # the projected-out subspace, whose reciprocals would explode.
            keep = th > 1e-8 * jnp.max(jnp.abs(th))
            nodes = jnp.where(keep, 1.0 / jnp.where(keep, th, 1.0), 0.0)
            return nodes, jnp.where(keep, jnp.sum(z**2) * U[0, :] ** 2, 0.0)

        nodes, weights = jax.vmap(one_probe)(jax.random.split(k_probe, num_probes))
        nodes, weights = jnp.ravel(nodes), jnp.ravel(weights)
        # Ratio estimator: the remainder has mass exactly n - c - k.
        weights = weights * rest / jnp.sum(weights)
    nodes = jnp.concatenate([jnp.zeros(c, dtype=dtype), exact, nodes])
    weights = jnp.concatenate([jnp.ones(c + k, dtype=dtype), weights])
    return StructureSpectrum(nodes, weights)


# ---------------------------------------------------------------------------
# PCBYM2Phi
# ---------------------------------------------------------------------------


def _h_over_x2(x: Array) -> Array:
    """``(x - log1p(x)) / x**2``, stable at ``x = 0`` (limit 1/2)."""
    small = jnp.abs(x) < 1e-3
    xs = jnp.where(small, 0.5, x)
    exact = (xs - jnp.log1p(xs)) / xs**2
    series = 0.5 - x / 3.0 + x**2 / 4.0 - x**3 / 5.0
    return jnp.where(small, series, exact)


class PCBYM2Phi(dist.Distribution):
    r"""PC prior on the BYM2 mixing parameter $\phi \in (0, 1)$.

    BYM2 (Riebler et al., 2016) mixes unstructured and scaled-structured
    effects, $b = \sigma(\sqrt{1-\phi}\,v + \sqrt{\phi}\,u_\ast)$. The base
    model is $\phi = 0$ (no spatial structure); with $\gamma_i$ the
    eigenvalues of $R_\ast^{+}$,

    $$
    2\,\mathrm{KLD}(\phi) = n\phi\big(\tfrac1n\operatorname{tr}R_\ast^{+} - 1\big)
        - \sum_i \log\big(1 - \phi + \phi\gamma_i\big),
    $$

    and $d(\phi) = \sqrt{2\,\mathrm{KLD}(\phi)}$ gets an exponential prior
    with $\lambda = -\log(1-\alpha)/d(U)$, so $P(\phi < U) = \alpha$. This is
    R-INLA's ``pc`` prior on BYM2's ``phi``. The null-space eigenvalues
    ($\gamma = 0$) send $d(\phi) \to \infty$ as $\phi \to 1$, which makes the
    prior proper on $(0, 1)$, so the spectrum must contain at least one (a
    BYM2 structure always has a null space).

    Args:
        U: Reference value of $\phi$, in ``(0, 1)``.
        alpha: $P(\phi < U)$, in ``(0, 1)``.
        structure_spectrum: The spectrum of $R_\ast^{+}$, from
            `structure_spectrum` (computed once per graph).

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> g = kl.grid_graph((6, 6))
        >>> R = kl.structure_matrix(g, scaled=True)
        >>> spec = lgm.structure_spectrum(R, kl.graph_null_space(g))
        >>> prior = lgm.PCBYM2Phi(U=0.5, alpha=2 / 3, structure_spectrum=spec)
        >>> phi = prior.sample(jax.random.key(0), (20_000,))
        >>> round(float((phi < 0.5).mean()), 2)  # P(phi < 0.5) = 2/3
        0.67
    """

    arg_constraints = {  # noqa: RUF012 (NumPyro's class-level contract)
        "U": constraints.open_interval(0.0, 1.0),
        "alpha": constraints.open_interval(0.0, 1.0),
    }
    support = constraints.open_interval(0.0, 1.0)
    pytree_data_fields = ("U", "alpha", "gamma", "weights")

    def __init__(
        self,
        U: ArrayLike = 0.5,
        alpha: ArrayLike = 2.0 / 3.0,
        *,
        structure_spectrum: StructureSpectrum,
        validate_args: bool | None = None,
    ) -> None:
        self.U = jnp.asarray(U)
        self.alpha = jnp.asarray(alpha)
        if self.U.ndim or self.alpha.ndim:
            raise ValueError("PCBYM2Phi takes scalar U and alpha")
        self.gamma = jnp.asarray(structure_spectrum.eigenvalues)
        self.weights = jnp.asarray(structure_spectrum.weights)
        try:
            has_null = bool(jnp.any((self.gamma == 0.0) & (self.weights > 0.0)))
        except jax.errors.ConcretizationTypeError:  # traced: checked eagerly
            has_null = True
        if not has_null:
            raise ValueError(
                "structure_spectrum has no null space (no zero eigenvalue): "
                "BYM2's structure is intrinsic, and the PC prior on phi is "
                "proper only through its null space"
            )
        super().__init__(batch_shape=(), validate_args=validate_args)

    def distance(self, phi: ArrayLike) -> Array:
        r"""$d(\phi) = \sqrt{2\,\mathrm{KLD}(\phi)}$."""
        phi = jnp.asarray(phi)
        return phi * jnp.sqrt(self._g(phi))

    def _g(self, phi):
        # 2 KLD = sum_i w_i [x_i - log1p(x_i)] with x_i = phi (gamma_i - 1),
        # = phi^2 sum_i w_i (gamma_i - 1)^2 h(x_i) / x_i^2: cancellation-free.
        a = self.gamma - 1.0
        x = jnp.expand_dims(phi, -1) * a
        return jnp.sum(self.weights * a**2 * _h_over_x2(x), axis=-1)

    def _ddistance(self, phi):
        a = self.gamma - 1.0
        x = jnp.expand_dims(phi, -1) * a
        num = jnp.sum(self.weights * a**2 / (1.0 + x), axis=-1)
        return num / (2.0 * jnp.sqrt(self._g(phi)))

    @property
    def rate(self) -> Array:
        """The rate $\\lambda = -\\log(1-\\alpha) / d(U)$."""
        return -jnp.log1p(-self.alpha) / self.distance(self.U)

    @validate_sample
    def log_prob(self, value):
        lam = self.rate
        return (
            jnp.log(lam) - lam * self.distance(value) + jnp.log(self._ddistance(value))
        )

    def cdf(self, value):
        return -jnp.expm1(-self.rate * self.distance(value))

    def icdf(self, q):
        return self._invert(-jnp.log1p(-q) / self.rate)

    def sample(self, key, sample_shape=()):
        q = jax.random.uniform(key, sample_shape, minval=1e-12, maxval=1.0)
        return self.icdf(q)

    def _invert(self, d: Array) -> Array:
        """Solve ``distance(phi) = d`` for phi by bisection (d is increasing)."""

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            below = self.distance(mid) < d
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lo = jnp.zeros_like(d)
        hi = jnp.ones_like(d)
        lo, hi = jax.lax.fori_loop(0, 60, body, (lo, hi))
        # Mass within rounding of 1 is real (d grows like sqrt(-log(1-phi)));
        # keep draws inside the open support.
        eps = jnp.finfo(lo.dtype).eps
        return jnp.clip(0.5 * (lo + hi), eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# PCMatern
# ---------------------------------------------------------------------------


class PCMatern(dist.Distribution):
    r"""Joint PC prior on the range $\rho$ and marginal sd $\sigma$ of a Matérn field.

    Fuglstad et al. (2019): in dimension $d$, the base model is infinite
    range and $\sigma = 0$, and

    $$
    \pi(\rho, \sigma) = \tfrac{d}{2}\,\lambda_\rho\,\rho^{-d/2-1}
        e^{-\lambda_\rho\rho^{-d/2}}\cdot\lambda_\sigma e^{-\lambda_\sigma\sigma},
    $$

    with $\lambda_\rho = -\log(\alpha_\rho)\,\rho_0^{d/2}$ and
    $\lambda_\sigma = -\log(\alpha_\sigma)/\sigma_0$, so
    $P(\rho < \rho_0) = \alpha_\rho$ and $P(\sigma > \sigma_0) = \alpha_\sigma$.
    The value is the pair ``[range, sigma]`` (event shape ``(2,)``). This is
    R-INLA's ``inla.spde2.pcmatern`` prior.

    Args:
        range0: Reference range $\rho_0 > 0$.
        alpha_range: $P(\rho < \rho_0)$, in ``(0, 1)``.
        sigma0: Reference standard deviation $\sigma_0 > 0$.
        alpha_sigma: $P(\sigma > \sigma_0)$, in ``(0, 1)``.
        d: Spatial dimension of the field.

    Examples:
        >>> import jax
        >>> import pyrox_lgm as lgm
        >>> prior = lgm.PCMatern(
        ...     range0=10.0, alpha_range=0.05, sigma0=2.0, alpha_sigma=0.05, d=2
        ... )
        >>> x = prior.sample(jax.random.key(0), (20_000,))
        >>> x.shape
        (20000, 2)
        >>> round(float((x[:, 0] < 10.0).mean()), 2)  # P(range < 10)
        0.05
        >>> round(float((x[:, 1] > 2.0).mean()), 2)  # P(sigma > 2)
        0.05
    """

    arg_constraints = {  # noqa: RUF012 (NumPyro's class-level contract)
        "range0": constraints.positive,
        "alpha_range": constraints.open_interval(0.0, 1.0),
        "sigma0": constraints.positive,
        "alpha_sigma": constraints.open_interval(0.0, 1.0),
    }
    support = constraints.independent(constraints.positive, 1)
    pytree_aux_fields = ("d",)

    def __init__(
        self,
        range0: ArrayLike = 1.0,
        alpha_range: ArrayLike = 0.05,
        sigma0: ArrayLike = 1.0,
        alpha_sigma: ArrayLike = 0.05,
        d: int = 2,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.range0 = jnp.asarray(range0)
        self.alpha_range = jnp.asarray(alpha_range)
        self.sigma0 = jnp.asarray(sigma0)
        self.alpha_sigma = jnp.asarray(alpha_sigma)
        if any(jnp.ndim(a) for a in (range0, alpha_range, sigma0, alpha_sigma)):
            raise ValueError("PCMatern takes scalar parameters")
        self.d = int(d)
        super().__init__(batch_shape=(), event_shape=(2,), validate_args=validate_args)

    @property
    def rate_range(self) -> Array:
        """$\\lambda_\\rho = -\\log(\\alpha_\\rho)\\,\\rho_0^{d/2}$."""
        return -jnp.log(self.alpha_range) * self.range0 ** (self.d / 2.0)

    @property
    def rate_sigma(self) -> Array:
        """$\\lambda_\\sigma = -\\log(\\alpha_\\sigma)/\\sigma_0$."""
        return -jnp.log(self.alpha_sigma) / self.sigma0

    def sample(self, key, sample_shape=()):
        k_r, k_s = jax.random.split(key)
        e = jax.random.exponential(k_r, sample_shape) / self.rate_range
        range_ = e ** (-2.0 / self.d)
        sigma = jax.random.exponential(k_s, sample_shape) / self.rate_sigma
        return jnp.stack([range_, sigma], axis=-1)

    @validate_sample
    def log_prob(self, value):
        range_, sigma = value[..., 0], value[..., 1]
        h = self.d / 2.0
        lr, ls = self.rate_range, self.rate_sigma
        log_range = jnp.log(h * lr) - (h + 1.0) * jnp.log(range_) - lr * range_**-h
        return log_range + jnp.log(ls) - ls * sigma
