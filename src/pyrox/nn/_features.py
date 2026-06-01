"""Pure-JAX feature helpers and Bayesian RFF feature layers.

This module hosts two related groups:

1. *Pure-JAX feature helpers* — :func:`fourier_features`,
   :func:`seasonal_frequencies`, :func:`seasonal_features`,
   :func:`interaction_features`, :func:`standardize`,
   :func:`unstandardize`. Stateless functions used by the deterministic
   coordinate-encoder layers in :mod:`pyrox.nn._layers`.
2. *Bayesian random-feature layers* — :class:`RBFFourierFeatures`,
   :class:`MaternFourierFeatures`, :class:`LaplaceFourierFeatures`,
   :class:`RandomKitchenSinks`, :class:`RBFCosineFeatures`,
   :class:`MaternCosineFeatures`, :class:`LaplaceCosineFeatures`,
   :class:`ArcCosineFourierFeatures`,
   :class:`VariationalFourierFeatures`, and :class:`HSGPFeatures`.
   These previously lived in :mod:`pyrox.nn._layers`. The deterministic
   RFF kernel is now consumed from :mod:`geonnax`
   (:func:`geonnax.rff_forward` and :func:`geonnax.rff_cosine_forward`),
   which is single-example ``(D,) -> (D,)``; the pyrox wrappers stay
   batched ``(*batch, D)`` and ``jax.vmap`` the geonnax core over the
   batch axis. The Bayesian site-registration (priors on ``W``, ``b``,
   ``lengthscale``) happens *once* per ``model()`` call and the sampled
   values are closed over the vmapped call — never sampled per data
   point.

Implementation uses :mod:`einx` (``einx.id`` for broadcasts/reshapes,
``einx.prod`` for axis reductions) for any non-trivial reshaping, per the
project convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import einx
import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Int

from pyrox._basis import fourier_basis, spectral_density
from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.gp._context import _kernel_context
from pyrox.gp._protocols import Kernel


def fourier_features(
    x: Float[Array, " N"],
    max_degree: int,
    *,
    rescale: bool = False,
) -> Float[Array, "N two_max_degree"]:
    r"""Cos/sin Fourier basis at dyadic frequencies.

    For each input element and each degree :math:`d \in \{0, \dots,
    D-1\}`, evaluates

    .. math::

        \phi_{d, \cos}(x) = \cos(2\pi \cdot 2^d \cdot x), \qquad
        \phi_{d, \sin}(x) = \sin(2\pi \cdot 2^d \cdot x).

    Returns the columns concatenated as ``[cos_0, ..., cos_{D-1},
    sin_0, ..., sin_{D-1}]``, matching Google's bayesnf layout.

    Args:
        x: Length-``N`` input vector.
        max_degree: Number of dyadic frequencies ``D``. Output has
            ``2 * max_degree`` columns.
        rescale: If ``True``, divide each ``(cos_d, sin_d)`` pair by
            ``d + 1`` to bias the prior toward lower-frequency basis
            functions.

    Returns:
        Array of shape ``(N, 2 * max_degree)``.
    """
    degrees = jnp.arange(max_degree)
    # Broadcast x to (N, D) frequencies without an explicit reshape.
    z = einx.id("n -> n d", x, d=max_degree) * (2.0 * jnp.pi * 2.0**degrees)
    feats = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
    if rescale:
        denom = jnp.concatenate([degrees + 1, degrees + 1])
        feats = feats / denom
    return feats


def seasonal_frequencies(
    periods: Sequence[float],
    harmonics: Sequence[int],
) -> tuple[list[int], list[float]]:
    r"""Flatten ``(period, harmonic_count)`` pairs into Python lists.

    For each period :math:`\tau_p` with :math:`H_p` harmonics, emits
    frequencies :math:`f_{p, h} = h / \tau_p` for :math:`h = 1, \dots,
    H_p`. The total length is :math:`F = \sum_p H_p`.

    Inputs are **Python sequences**, not JAX arrays, so this helper
    runs at trace time and never triggers a concretization error under
    ``jax.jit``. Most callers won't use it directly; it's exposed for
    symmetry with :func:`seasonal_features`.

    Args:
        periods: Period values.
        harmonics: Number of harmonics per period.

    Returns:
        ``(period_index, frequency)``: two Python lists of length
        :math:`F = \sum_p H_p`.
    """
    period_index: list[int] = []
    freqs: list[float] = []
    for p_idx, (period, n_h) in enumerate(zip(periods, harmonics, strict=True)):
        for h in range(1, int(n_h) + 1):
            period_index.append(p_idx)
            freqs.append(float(h) / float(period))
    return period_index, freqs


def seasonal_features(
    x: Float[Array, " N"],
    periods: Sequence[float],
    harmonics: Sequence[int],
    *,
    rescale: bool = False,
) -> Float[Array, "N two_F"]:
    r"""Cos/sin features at multiples of :math:`2\pi / \tau_p`.

    For each period :math:`\tau_p` with :math:`H_p` harmonics, evaluates

    .. math::

        \phi_{p, h, \cos}(x) = \cos(2\pi h x / \tau_p), \qquad
        \phi_{p, h, \sin}(x) = \sin(2\pi h x / \tau_p),

    for :math:`h = 1, \dots, H_p`. Returns the cos columns concatenated
    with the sin columns, length :math:`F = \sum_p H_p` each.

    ``periods`` and ``harmonics`` are **Python sequences** (tuples,
    lists, or 0-d JAX arrays wrapped at the call site). Keeping them as
    Python values lets the function run cleanly under ``jax.jit`` and
    ``lax.scan`` without triggering a concretization error.

    Args:
        x: Time/index input, shape ``(N,)``.
        periods: Period values.
        harmonics: Harmonics per period.
        rescale: If ``True``, divide each ``(cos, sin)`` pair by its
            within-period harmonic index, biasing the prior toward
            longer-wavelength modes within each period.

    Returns:
        Array of shape ``(N, 2 * F)``.
    """
    _, freq_list = seasonal_frequencies(periods, harmonics)
    if not freq_list:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    freqs = jnp.asarray(freq_list, dtype=jnp.float32)
    z = einx.id("n -> n f", x, f=freqs.shape[0]) * (2.0 * jnp.pi * freqs)
    feats = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
    if rescale:
        # Rescale by within-period harmonic index (1, 2, ..., H_p).
        h_within_list: list[float] = []
        for n_h in harmonics:
            h_within_list.extend(range(1, int(n_h) + 1))
        h_within = jnp.asarray(h_within_list, dtype=jnp.float32)
        denom = jnp.concatenate([h_within, h_within])
        feats = feats / denom
    return feats


def interaction_features(
    x: Float[Array, "N D"],
    pairs: Int[Array, "K 2"],
) -> Float[Array, "N K"]:
    r"""Element-wise products on selected pairs of input columns.

    For each pair :math:`(i, j)` and each row :math:`n`, computes
    :math:`x_{n, i} \cdot x_{n, j}`.

    Args:
        x: Input matrix, shape ``(N, D)``.
        pairs: Index pairs, shape ``(K, 2)``. Empty pairs yield an
            ``(N, 0)`` output.

    Returns:
        Array of shape ``(N, K)`` of pairwise products.
    """
    if pairs.shape[0] == 0:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    # x[:, pairs] has shape (N, K, 2); reduce the paired axis with prod.
    selected = x[:, pairs]
    return einx.prod("n k [two]", selected)


def standardize(
    x: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Affine standardize: ``(x - mu) / std``.

    Broadcasts ``mu`` and ``std`` against ``x`` per the JAX broadcasting
    rules. ``std`` is *not* clamped; pass a positive value or guard
    upstream.
    """
    return (x - mu) / std


def unstandardize(
    z: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Inverse of :func:`standardize`: ``z * std + mu``."""
    return z * std + mu


# ---------------------------------------------------------------------------
# Bayesian random Fourier feature layers
# ---------------------------------------------------------------------------
#
# These layers wrap the deterministic single-example RFF cores
# ``geonnax.rff_forward`` and ``geonnax.rff_cosine_forward`` behind a
# pyrox ``PyroxModule`` that registers priors on the frequency matrix
# ``W``, optional cosine bias ``b``, and lengthscale ``ls`` as
# ``pyrox_sample`` sites. The sites are registered **once per
# ``model()`` call** (never per data point); the sampled values are then
# closed over a ``jax.vmap`` of the geonnax core across the batch axis,
# preserving the ``(*batch, D)`` external API.


def _vmap_rff_forward(
    W: Float[Array, "D_in n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch D_rff"]:
    """Vmap ``geonnax.rff_forward`` over the leading batch axis of ``x``.

    The geonnax core is single-example ``(D_in,) -> (D_rff,)``; here we
    vmap with ``W``, ``lengthscale``, ``n_features`` closed over so the
    Bayesian site-registration happens exactly once outside the vmap.
    """
    return jax.vmap(lambda xi: geonnax.rff_forward(W, lengthscale, n_features, xi))(x)


def _vmap_rff_cosine_forward(
    W: Float[Array, "D_in n_features"],
    b: Float[Array, " n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch n_features"]:
    """Vmap ``geonnax.rff_cosine_forward`` over the leading batch axis of ``x``."""
    return jax.vmap(
        lambda xi: geonnax.rff_cosine_forward(W, b, lengthscale, n_features, xi)
    )(x)


class RBFFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with RBF spectral density.

    Both the spectral frequencies :math:`W` and the lengthscale
    :math:`\ell` are ``pyrox_sample`` sites — :math:`W` has a
    standard normal prior (the RBF spectral density) and :math:`\ell`
    has a ``LogNormal`` prior. Under SVI, the guide learns a posterior
    over both; under a seed handler, they are drawn from the prior.

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs (output dim
            ``2 * n_features``).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> RBFFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_forward(W, ls, self.n_features, x)


class MaternFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with Matern spectral density.

    Spectral frequencies :math:`W` have a ``StudentT(df=2\nu)`` prior
    (the Matern spectral density). The smoothness :math:`\nu` controls
    the regularity: ``nu=0.5`` (Laplace), ``nu=1.5`` (Matern-3/2),
    ``nu=2.5`` (Matern-5/2).

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs.
        nu: Smoothness parameter :math:`\nu`.
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    nu: float = eqx.field(static=True, default=1.5)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        nu: float = 1.5,
        lengthscale: float = 1.0,
    ) -> MaternFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            nu=nu,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=2.0 * self.nu, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_forward(W, ls, self.n_features, x)


class LaplaceFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with Laplace (Matern-1/2) spectral density.

    Spectral frequencies :math:`W` have a ``Cauchy`` prior (Student-t
    with ``df = 1``).

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs.
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> LaplaceFourierFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=1.0, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_forward(W, ls, self.n_features, x)


class RandomKitchenSinks(PyroxModule):
    r"""Random Kitchen Sinks: RFF + a learned linear head.

    Composes any RFF layer (:class:`RBFFourierFeatures`,
    :class:`MaternFourierFeatures`, :class:`LaplaceFourierFeatures`)
    with a trainable linear projection:

    .. math::

        y = \phi(x)\, \beta + b

    The linear head (``beta``, ``bias``) is registered via
    ``pyrox_sample`` with ``Normal`` priors.

    Attributes:
        rff: The underlying RFF feature layer.
        init_beta: Initial linear weights.
        init_bias: Initial bias vector.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    rff: RBFFourierFeatures | MaternFourierFeatures | LaplaceFourierFeatures
    init_beta: Float[Array, "D_rff D_out"]
    init_bias: Float[Array, " D_out"]
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        rff: RBFFourierFeatures | MaternFourierFeatures | LaplaceFourierFeatures,
        out_features: int,
    ) -> RandomKitchenSinks:
        """Construct from a pre-built RFF layer with zero-initialized head."""
        beta = jnp.zeros((2 * rff.n_features, out_features))
        bias = jnp.zeros(out_features)
        return cls(rff=rff, init_beta=beta, init_bias=bias)

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        phi = self.rff(x)
        beta = self.pyrox_sample(
            "beta",
            dist.Normal(self.init_beta, 1.0).to_event(2),
        )
        bias = self.pyrox_sample(
            "bias",
            dist.Normal(self.init_bias, 1.0).to_event(1),
        )
        return einx.dot("... r, r dout -> ... dout", phi, beta) + bias


class RBFCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the RBF kernel.

    Uses the single-cosine feature map with a bias term:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathcal{N}(0, I)` and
    :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. This variant produces
    ``n_features``-dimensional output (half the dimension of the
    ``[cos, sin]`` variant in :class:`RBFFourierFeatures`) and is
    commonly used in Random Kitchen Sinks implementations.

    All parameters (:math:`W`, :math:`b`, :math:`\ell`) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> RBFCosineFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "b",
            dist.Uniform(0.0, 2.0 * jnp.pi).expand([self.n_features]).to_event(1),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_cosine_forward(W, b, ls, self.n_features, x)


class MaternCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the Matern kernel.

    Single-cosine analogue of :class:`MaternFourierFeatures`:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathrm{StudentT}(2\nu)` (the Matern spectral
    density) and :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. Output dim is
    ``n_features`` (vs ``2 * n_features`` for the ``[cos, sin]``
    variant). Approximates the same kernel as
    :class:`MaternFourierFeatures` in expectation but with higher
    variance per draw — see Sutherland & Schneider (2015).

    All parameters (:math:`W`, :math:`b`, :math:`\ell`) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        nu: Smoothness parameter :math:`\nu`.
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    nu: float = eqx.field(static=True, default=1.5)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        nu: float = 1.5,
        lengthscale: float = 1.0,
    ) -> MaternCosineFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            nu=nu,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=2.0 * self.nu, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "b",
            dist.Uniform(0.0, 2.0 * jnp.pi).expand([self.n_features]).to_event(1),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_cosine_forward(W, b, ls, self.n_features, x)


class LaplaceCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the Laplace kernel.

    Single-cosine analogue of :class:`LaplaceFourierFeatures` (the
    Matern-1/2 kernel):

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathrm{Cauchy}(0, 1)` (Student-t with
    ``df = 1``) and :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. Output
    dim is ``n_features``.

    All parameters (:math:`W`, :math:`b`, :math:`\ell`) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> LaplaceCosineFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=1.0, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "b",
            dist.Uniform(0.0, 2.0 * jnp.pi).expand([self.n_features]).to_event(1),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_cosine_forward(W, b, ls, self.n_features, x)


class ArcCosineFourierFeatures(PyroxModule):
    r"""Random features for the arc-cosine kernel (Cho & Saul, 2009).

    The arc-cosine kernel of order :math:`p` corresponds to an
    infinite-width single-layer ReLU network. The random feature map
    is:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\max(0,\, x W / \ell)^p

    where :math:`W \sim \mathcal{N}(0, I)`.

    ``order=0`` gives the Heaviside (step) feature; ``order=1`` gives
    the ReLU feature (the most common); ``order=2`` gives the squared
    ReLU feature.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        order: Kernel order (0, 1, or 2).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    order: int = eqx.field(static=True, default=1)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        order: int = 1,
        lengthscale: float = 1.0,
    ) -> ArcCosineFourierFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            order=order,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        z = einx.dot("... din, din f -> ... f", x, W) / ls
        if self.order == 0:
            h = (z > 0.0).astype(x.dtype)
        else:
            h = jnp.maximum(z, 0.0) ** self.order
        return jnp.sqrt(2.0 / self.n_features) * h


class VariationalFourierFeatures(PyroxModule):
    r"""VSSGP — RFF with a learnable variational posterior over frequencies.

    Standard RFF (e.g. :class:`RBFFourierFeatures`) treats the spectral
    frequencies :math:`W` as a frozen prior draw; VSSGP (Gal & Turner,
    2015) treats :math:`W` as a latent with a learnable mean-field
    posterior, recovering spectral *uncertainty* on top of the
    feature-space uncertainty.

    Prior: :math:`p(W) = \mathcal{N}(0, I)` (RBF spectral density in
    lengthscale-1 units). The lengthscale is itself a sampled site
    (``LogNormal(log init_lengthscale, 1)``) so that frequencies are
    rescaled to the physical kernel.

    Under SVI, attach an :class:`~numpyro.infer.autoguide.AutoNormal` to
    learn the posterior on ``W``; under prior-only seeds, behaves
    identically to :class:`RBFFourierFeatures`.

    Attributes:
        in_features: Input dimension :math:`D`.
        n_features: Number of frequency pairs (output dim ``2 * n_features``).
        init_lengthscale: Prior location for the kernel lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> VariationalFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        # Same prior as RBFFourierFeatures — the *posterior* is what differs
        # under SVI: an attached AutoGuide learns q(W) instead of forcing W
        # to its prior draw.
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_forward(W, ls, self.n_features, x)


class HSGPFeatures(PyroxModule):
    r"""Hilbert-Space Gaussian Process feature layer (Riutort-Mayol et al., 2023).

    A *deterministic* Laplacian-eigenfunction basis on the bounded box
    :math:`[-L, L]^D` plus learnable per-basis amplitudes with a
    kernel-spectral-density prior:

    .. math::

        \hat{f}(x) = \sum_{j=1}^{M} \alpha_j\,\sqrt{S(\sqrt{\lambda_j})}\,\phi_j(x),
        \quad \alpha_j \sim \mathcal{N}(0, 1).

    This is the NN-side dual of :class:`pyrox.gp.FourierInducingFeatures`
    — same basis, different prior wiring. As ``M`` and ``L`` grow, the
    induced GP converges to the kernel passed in.

    Attributes:
        in_features: Input dimension :math:`D`.
        num_basis_per_dim: Per-axis number of 1D eigenfunctions; total
            basis count is ``prod(num_basis_per_dim)``.
        L: Per-axis box half-width.
        kernel: A stationary kernel from :mod:`pyrox.gp` whose spectral
            density supplies the per-basis prior variance. Currently
            :class:`pyrox.gp.RBF` and :class:`pyrox.gp.Matern` are
            supported by :func:`pyrox._basis.spectral_density`.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    num_basis_per_dim: tuple[int, ...] = eqx.field(static=True)
    L: tuple[float, ...] = eqx.field(static=True)
    kernel: Kernel
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        num_basis_per_dim: int | tuple[int, ...],
        L: float | tuple[float, ...],
        *,
        kernel: Kernel,
    ) -> HSGPFeatures:
        if isinstance(num_basis_per_dim, int):
            num_basis_per_dim = (num_basis_per_dim,) * in_features
        if isinstance(L, int | float):
            L = (float(L),) * in_features
        if len(num_basis_per_dim) != in_features:
            raise ValueError(
                f"num_basis_per_dim length ({len(num_basis_per_dim)}) "
                f"must match in_features ({in_features})."
            )
        if len(L) != in_features:
            raise ValueError(
                f"L length ({len(L)}) must match in_features ({in_features})."
            )
        if any(L_d <= 0 for L_d in L):
            raise ValueError(f"L must be all positive; got {L}.")
        if any(M_d < 1 for M_d in num_basis_per_dim):
            raise ValueError(
                f"num_basis_per_dim must be all >= 1; got {num_basis_per_dim}."
            )
        return cls(
            in_features=in_features,
            num_basis_per_dim=tuple(num_basis_per_dim),
            L=tuple(float(L_d) for L_d in L),
            kernel=kernel,
        )

    @property
    def num_basis(self) -> int:
        n = 1
        for m in self.num_basis_per_dim:
            n *= m
        return n

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, " N"]:
        Phi, lam = fourier_basis(x, self.num_basis_per_dim, self.L)  # (N, M), (M,)
        # Spectral density evaluated under the kernel's own context so any
        # priors on (variance, lengthscale) register exactly once.
        with _kernel_context(self.kernel):
            S = spectral_density(self.kernel, lam, D=self.in_features)
        sqrt_S = jnp.sqrt(S)
        alpha = self.pyrox_sample(
            "alpha",
            dist.Normal(0.0, 1.0).expand([self.num_basis]).to_event(1),
        )
        return einx.dot("n m, m -> n", Phi, sqrt_S * alpha)
