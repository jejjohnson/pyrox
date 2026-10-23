"""Pure-JAX feature helpers and Bayesian RFF feature layers.

This module hosts two related groups:

1. *Pure-JAX feature helpers* — `fourier_features`,
   `seasonal_frequencies`, `seasonal_features`,
   `interaction_features`, `standardize`,
   `unstandardize`. Stateless functions re-exported from
   `geonnax.basis` (the implementations moved there with the
   package split) for the deterministic coordinate-encoder layers.
2. *Bayesian random-feature layers* — `RBFFourierFeatures`,
   `MaternFourierFeatures`, `LaplaceFourierFeatures`,
   `RandomKitchenSinks`, `RBFCosineFeatures`,
   `MaternCosineFeatures`, `LaplaceCosineFeatures`,
   `ArcCosineFourierFeatures`,
   `VariationalFourierFeatures`, `HSGPFeatures` and
   `SphericalHarmonicFeatures`.
   These previously lived in `pyrox_nn._layers`. The deterministic
   RFF kernel is now consumed from `geonnax`
   (`geonnax.rff_forward` and `geonnax.rff_cosine_forward`),
   which is single-example ``(D,) -> (D,)``; the pyrox wrappers stay
   batched ``(*batch, D)`` and ``jax.vmap`` the geonnax core over the
   batch axis. The Bayesian site-registration (priors on ``W``, ``b``,
   ``lengthscale``) happens *once* per ``model()`` call and the sampled
   values are closed over the vmapped call — never sampled per data
   point.

Implementation uses `einx` (``einx.id`` for broadcasts/reshapes,
``einx.prod`` for axis reductions) for any non-trivial reshaping, per the
project convention.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import geonnax
import jax.numpy as jnp
import numpyro.distributions as dist

# Pure-JAX feature transforms moved to geonnax with the package split;
# re-export them (self-alias form marks them as intentional re-exports)
# so `from pyrox_nn._features import fourier_features` keeps working.
from geonnax.basis import (
    fourier_features as fourier_features,
    interaction_features as interaction_features,
    seasonal_features as seasonal_features,
    seasonal_frequencies as seasonal_frequencies,
    standardize as standardize,
    unstandardize as unstandardize,
)
from geonnax.geo import lonlat_to_cartesian3d
from jaxtyping import Array, Float
from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox_gp import funk_hecke_coefficients
from pyrox_gp._basis import (
    fourier_basis,
    harmonic_degrees,
    real_spherical_harmonics,
    spectral_density,
)
from pyrox_gp._context import _kernel_context
from pyrox_gp._protocols import Kernel

from pyrox_nn._batching import vmap_over_flat_batch


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

    The geonnax core is single-example ``(D_in,) -> (D_rff,)``;
    `vmap_over_flat_batch` preserves the documented support for
    both unbatched ``(D_in,)`` and batched ``(*batch, D_in)`` callers.
    """
    return vmap_over_flat_batch(
        lambda xi: geonnax.rff_forward(W, lengthscale, n_features, xi), x
    )


def _vmap_rff_cosine_forward(
    W: Float[Array, "D_in n_features"],
    b: Float[Array, " n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch n_features"]:
    """Vmap ``geonnax.rff_cosine_forward`` with the same flatten/restore
    pattern as `_vmap_rff_forward`."""
    return vmap_over_flat_batch(
        lambda xi: geonnax.rff_cosine_forward(W, b, lengthscale, n_features, xi), x
    )


class RBFFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with RBF spectral density.

    Both the spectral frequencies $W$ and the lengthscale
    $\ell$ are ``pyrox_sample`` sites — $W$ has a
    standard normal prior (the RBF spectral density) and $\ell$
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

    Spectral frequencies $W$ have a ``StudentT(df=2\nu)`` prior
    (the Matern spectral density). The smoothness $\nu$ controls
    the regularity: ``nu=0.5`` (Laplace), ``nu=1.5`` (Matern-3/2),
    ``nu=2.5`` (Matern-5/2).

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs.
        nu: Smoothness parameter $\nu$.
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

    Spectral frequencies $W$ have a ``Cauchy`` prior (Student-t
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
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _vmap_rff_forward(W, ls, self.n_features, x)


class RandomKitchenSinks(PyroxModule):
    r"""Random Kitchen Sinks: RFF + a learned linear head.

    Composes any RFF layer (`RBFFourierFeatures`,
    `MaternFourierFeatures`, `LaplaceFourierFeatures`)
    with a trainable linear projection:

    $$
    y = \phi(x)\, \beta + b
    $$

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

    $$
    \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)
    $$

    where $W \sim \mathcal{N}(0, I)$ and
    $b \sim \mathrm{Uniform}(0, 2\pi)$. This variant produces
    ``n_features``-dimensional output (half the dimension of the
    ``[cos, sin]`` variant in `RBFFourierFeatures`) and is
    commonly used in Random Kitchen Sinks implementations.

    All parameters ($W$, $b$, $\ell$) are
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

    Single-cosine analogue of `MaternFourierFeatures`:

    $$
    \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)
    $$

    where $W \sim \mathrm{StudentT}(2\nu)$ (the Matern spectral
    density) and $b \sim \mathrm{Uniform}(0, 2\pi)$. Output dim is
    ``n_features`` (vs ``2 * n_features`` for the ``[cos, sin]``
    variant). Approximates the same kernel as
    `MaternFourierFeatures` in expectation but with higher
    variance per draw — see Sutherland & Schneider (2015).

    All parameters ($W$, $b$, $\ell$) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        nu: Smoothness parameter $\nu$.
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

    Single-cosine analogue of `LaplaceFourierFeatures` (the
    Matern-1/2 kernel):

    $$
    \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)
    $$

    where $W \sim \mathrm{Cauchy}(0, 1)$ (Student-t with
    ``df = 1``) and $b \sim \mathrm{Uniform}(0, 2\pi)$. Output
    dim is ``n_features``.

    All parameters ($W$, $b$, $\ell$) are
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

    The arc-cosine kernel of order $p$ corresponds to an
    infinite-width single-layer ReLU network. The random feature map
    is:

    $$
    \phi(x) = \sqrt{2 / D}\,\max(0,\, x W / \ell)^p
    $$

    where $W \sim \mathcal{N}(0, I)$.

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

    Standard RFF (e.g. `RBFFourierFeatures`) treats the spectral
    frequencies $W$ as a frozen prior draw; VSSGP (Gal & Turner,
    2015) treats $W$ as a latent with a learnable mean-field
    posterior, recovering spectral *uncertainty* on top of the
    feature-space uncertainty.

    Prior: $p(W) = \mathcal{N}(0, I)$ (RBF spectral density in
    lengthscale-1 units). The lengthscale is itself a sampled site
    (``LogNormal(log init_lengthscale, 1)``) so that frequencies are
    rescaled to the physical kernel.

    Under SVI, attach an `AutoNormal` to
    learn the posterior on ``W``; under prior-only seeds, behaves
    identically to `RBFFourierFeatures`.

    Attributes:
        in_features: Input dimension $D$.
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
    $[-L, L]^D$ plus learnable per-basis amplitudes with a
    kernel-spectral-density prior:

    $$
    \hat{f}(x) = \sum_{j=1}^{M} \alpha_j\,\sqrt{S(\sqrt{\lambda_j})}\,\phi_j(x),
    \quad \alpha_j \sim \mathcal{N}(0, 1).
    $$

    This is the NN-side dual of `pyrox_gp.FourierInducingFeatures`
    — same basis, different prior wiring. As ``M`` and ``L`` grow, the
    induced GP converges to the kernel passed in.

    Attributes:
        in_features: Input dimension $D$.
        num_basis_per_dim: Per-axis number of 1D eigenfunctions; total
            basis count is ``prod(num_basis_per_dim)``.
        L: Per-axis box half-width.
        kernel: A stationary kernel from `pyrox_gp` whose spectral
            density supplies the per-basis prior variance. Currently
            `pyrox_gp.RBF` and `pyrox_gp.Matern` are
            supported by `pyrox_gp._basis.spectral_density`.
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


class SphericalHarmonicFeatures(PyroxModule):
    r"""Weight-space GP on the sphere $S^2$ from real spherical harmonics.

    The $S^2$ analogue of `HSGPFeatures`. For a zonal kernel
    $k(n, n') = \kappa(n \cdot n')$ with Funk-Hecke coefficients
    $a_l = 2\pi \int_{-1}^{1} \kappa(t) P_l(t)\,dt$, the addition theorem
    gives the Mercer expansion

    $$
    k(n, n') = \sum_{l=0}^{\infty} a_l \sum_{m=-l}^{l} Y_{lm}(n)\,Y_{lm}(n'),
    $$

    with $Y_{lm}$ the orthonormal real spherical harmonics. Truncating at
    degree $L$ = ``l_max`` gives the layer

    $$
    f(n) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \sqrt{a_l}\,\alpha_{lm}\,Y_{lm}(n),
    \qquad \alpha_{lm} \sim \mathcal{N}(0, 1),
    $$

    a GP whose covariance is the truncated kernel
    $k_L(n, n') = \sum_{l \le L} a_l \sum_m Y_{lm}(n) Y_{lm}(n')$, linear in
    the number of points. This is the NN-side dual of
    `pyrox_gp.SphericalHarmonicInducingFeatures` (VISH, Dutordoir et al.,
    2020): same harmonics, same `pyrox_gp.funk_hecke_coefficients`.

    The kernel is any isotropic `pyrox_gp` kernel, evaluated on the
    embedded unit vectors (chordal distance), as in VISH. A kernel with a
    per-dimension (ARD) lengthscale is not zonal and raises
    ``NotImplementedError``. Coefficients that quadrature round-off pushes
    below zero are clipped to zero (a kernel restricted from
    $\mathbb{R}^3$ to the sphere has $a_l \ge 0$).

    **Sites.** ``<pyrox_name>.alpha`` with prior
    $\mathcal{N}(0, I_{(L+1)^2})$; any priors on the kernel's
    hyperparameters register once per call under the kernel's own scope
    (evaluated inside `pyrox_gp._context._kernel_context`).

    Attributes:
        l_max: Maximum harmonic degree $L$ (inclusive); the layer has
            ``(l_max + 1) ** 2`` features.
        kernel: An isotropic `pyrox_gp` kernel (e.g. `pyrox_gp.Matern`,
            `pyrox_gp.RBF`) built without ``input_dim``.
        input_mode: ``"xyz"`` for unit vectors ``(N, 3)``, or ``"lonlat"``
            for ``(N, 2)`` longitude / latitude columns.
        input_unit: Unit of the ``"lonlat"`` columns, ``"degrees"`` or
            ``"radians"``; ``None`` for ``"xyz"``.
        num_quadrature: Gauss-Legendre nodes for the Funk-Hecke integral.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Examples:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> from pyrox_gp import Matern
        >>> layer = SphericalHarmonicFeatures.init(
        ...     l_max=10,
        ...     kernel=Matern(nu=1.5, init_lengthscale=0.5),
        ...     input_mode="lonlat",
        ...     input_unit="degrees",
        ... )
        >>> lonlat = jnp.array([[0.0, 0.0], [90.0, 45.0], [-120.0, -60.0]])
        >>> with handlers.seed(rng_seed=0):
        ...     f = layer(lonlat)
        >>> f.shape, layer.num_basis
        ((3,), 121)
    """

    kernel: Kernel
    l_max: int = eqx.field(static=True)
    input_mode: Literal["xyz", "lonlat"] = eqx.field(static=True, default="xyz")
    input_unit: Literal["degrees", "radians"] | None = eqx.field(
        static=True, default=None
    )
    num_quadrature: int = eqx.field(static=True, default=256)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        *,
        l_max: int,
        kernel: Kernel,
        input_mode: Literal["xyz", "lonlat"] = "xyz",
        input_unit: Literal["degrees", "radians"] | None = None,
        num_quadrature: int = 256,
        pyrox_name: str | None = None,
    ) -> SphericalHarmonicFeatures:
        """Validate the configuration and build the layer.

        Raises:
            ValueError: ``l_max < 0``, ``num_quadrature < 1``, an unknown
                ``input_mode``, or an ``input_unit`` that is missing for
                ``"lonlat"``, unknown, or given for ``"xyz"``.
        """
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {l_max}.")
        if num_quadrature < 1:
            raise ValueError(f"num_quadrature must be >= 1; got {num_quadrature}.")
        if input_mode not in ("xyz", "lonlat"):
            raise ValueError(
                f"input_mode must be 'xyz' or 'lonlat'; got {input_mode!r}."
            )
        if input_mode == "lonlat" and input_unit not in ("degrees", "radians"):
            raise ValueError(
                "input_mode='lonlat' needs an explicit input_unit, 'degrees' or "
                f"'radians'; got {input_unit!r}."
            )
        if input_mode == "xyz" and input_unit is not None:
            raise ValueError(
                f"input_unit applies to input_mode='lonlat' only; got {input_unit!r}."
            )
        return cls(
            kernel=kernel,
            l_max=l_max,
            input_mode=input_mode,
            input_unit=input_unit,
            num_quadrature=num_quadrature,
            pyrox_name=pyrox_name,
        )

    @property
    def num_basis(self) -> int:
        """Number of harmonics, ``(l_max + 1) ** 2``."""
        return (self.l_max + 1) ** 2

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, " N"]:
        """Evaluate one GP draw at ``x``.

        Args:
            x: Unit vectors ``(N, 3)`` (``input_mode="xyz"``) or lon / lat
                ``(N, 2)`` in ``input_unit`` (``input_mode="lonlat"``).

        Returns:
            The latent function values, shape ``(N,)``.
        """
        d_in = 3 if self.input_mode == "xyz" else 2
        if x.ndim != 2 or x.shape[-1] != d_in:
            raise ValueError(
                f"input_mode={self.input_mode!r} expects x of shape (N, {d_in}); "
                f"got {x.shape}."
            )
        if self.input_mode == "lonlat":
            if self.input_unit is None:  # built without init(): no default unit
                raise ValueError(
                    "input_mode='lonlat' needs an explicit input_unit, "
                    "'degrees' or 'radians'."
                )
            x = lonlat_to_cartesian3d(x, input_unit=self.input_unit)  # (N, 3)
        Y = real_spherical_harmonics(x, self.l_max)  # (N, M)
        # Funk-Hecke coefficients; the helper rejects ARD lengthscales. It
        # opens its own kernel context, so this outer one is a no-op today
        # (contexts are reentrant); it is kept, as in HSGPFeatures, so that
        # any kernel call added here shares the same hyperprior draw.
        with _kernel_context(self.kernel):
            a = funk_hecke_coefficients(
                self.kernel, self.l_max, num_quadrature=self.num_quadrature
            )  # (l_max + 1,)
        # a_l >= 0 for a kernel restricted from R^3; quadrature round-off can
        # push tail coefficients below 0 (often in float32). Clip them with a
        # double where: sqrt(maximum(a, 0)) has a 0 * inf = NaN gradient there.
        pos = a > 0
        sqrt_a_l = jnp.where(pos, jnp.sqrt(jnp.where(pos, a, 1.0)), 0.0)
        degrees = jnp.asarray(harmonic_degrees(self.l_max))  # (M,) degree l per column
        sqrt_a = sqrt_a_l[degrees]  # (M,)
        alpha = self.pyrox_sample(
            "alpha",
            dist.Normal(0.0, 1.0).expand([self.num_basis]).to_event(1),
        )
        # The quadrature runs in the default float dtype; keep the input's.
        return einx.dot("n m, m -> n", Y, (sqrt_a * alpha).astype(Y.dtype))
