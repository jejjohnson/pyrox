"""Concrete kernel classes — ``Parameterized`` wrappers over the math primitives.

Each class registers its hyperparameters with constraints through
:class:`pyrox._core.Parameterized`, so users can attach priors and
autoguides via :meth:`set_prior` / :meth:`autoguide` and flip between
prior/guide modes with :meth:`set_mode`. The numerical body delegates to
the pure closed-form functions in :mod:`pyrox.gp._src.kernels`.

Kernels with a static structural parameter (``Matern.nu``,
``Polynomial.degree``) take that parameter as a class field rather than
a registered JAX param — those numbers choose code paths, not
optimization targets.

Scalable matrix construction (mixed-precision accumulation, implicit
operators, batched matvec) lives in :mod:`gaussx`; these wrappers own
the NumPyro-aware surface only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox._core import Parameterized, pyrox_method
from pyrox.gp._protocols import Kernel
from pyrox.gp._src import kernels as _k


class _ParameterizedKernel(Parameterized, Kernel):
    """Shared base — mixes :class:`Parameterized` state with the :class:`Kernel`.

    Subclasses only need to implement :meth:`setup` (register params +
    priors) and :meth:`__call__` (evaluate the math primitive).
    """

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Stationary-kernel fast diagonal: constant variance on every point.

        Subclasses that are not strictly stationary (``Linear``,
        ``Polynomial``) override with a point-wise computation.
        """
        n = X.shape[0]
        return self.get_param("variance") * jnp.ones(n, dtype=X.dtype)


class RBF(_ParameterizedKernel):
    """Radial basis function (squared exponential) kernel."""

    pyrox_name: str = "RBF"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.asarray(self.init_lengthscale),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.rbf_kernel(
            X1, X2, self.get_param("variance"), self.get_param("lengthscale")
        )


class Matern(_ParameterizedKernel):
    """Matern kernel with ``nu in {0.5, 1.5, 2.5}``.

    ``nu`` is a static class attribute — it selects a code path in the
    underlying math primitive and is not a trainable parameter.
    """

    pyrox_name: str = "Matern"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    nu: float = 2.5

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.asarray(self.init_lengthscale),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.matern_kernel(
            X1,
            X2,
            self.get_param("variance"),
            self.get_param("lengthscale"),
            self.nu,
        )


class Periodic(_ParameterizedKernel):
    """Periodic (MacKay) kernel."""

    pyrox_name: str = "Periodic"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    init_period: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.asarray(self.init_lengthscale),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "period",
            jnp.asarray(self.init_period),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.periodic_kernel(
            X1,
            X2,
            self.get_param("variance"),
            self.get_param("lengthscale"),
            self.get_param("period"),
        )


class Linear(_ParameterizedKernel):
    """Linear kernel ``sigma^2 x^T x' + bias``."""

    pyrox_name: str = "Linear"
    init_variance: float = 1.0
    init_bias: float = 0.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param("bias", jnp.asarray(self.init_bias))

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.linear_kernel(
            X1, X2, self.get_param("variance"), self.get_param("bias")
        )

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        # Non-stationary: diagonal depends on |X[i]|^2.
        v = self.get_param("variance")
        b = self.get_param("bias")
        return v * jnp.sum(X * X, axis=-1) + b


class RationalQuadratic(_ParameterizedKernel):
    """Rational quadratic kernel."""

    pyrox_name: str = "RationalQuadratic"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    init_alpha: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.asarray(self.init_lengthscale),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "alpha",
            jnp.asarray(self.init_alpha),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.rational_quadratic_kernel(
            X1,
            X2,
            self.get_param("variance"),
            self.get_param("lengthscale"),
            self.get_param("alpha"),
        )


class Polynomial(_ParameterizedKernel):
    """Polynomial kernel ``sigma^2 (x^T x' + bias)^degree``.

    ``degree`` is a static class field (it selects an integer power, not
    an optimization target).
    """

    pyrox_name: str = "Polynomial"
    init_variance: float = 1.0
    init_bias: float = 0.0
    degree: int = 2

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param("bias", jnp.asarray(self.init_bias))

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.polynomial_kernel(
            X1,
            X2,
            self.get_param("variance"),
            self.get_param("bias"),
            self.degree,
        )

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        v = self.get_param("variance")
        b = self.get_param("bias")
        return v * (jnp.sum(X * X, axis=-1) + b) ** self.degree


class Cosine(_ParameterizedKernel):
    """Cosine kernel ``sigma^2 cos(2 pi ||x - x'|| / period)``."""

    pyrox_name: str = "Cosine"
    init_variance: float = 1.0
    init_period: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "period",
            jnp.asarray(self.init_period),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.cosine_kernel(
            X1, X2, self.get_param("variance"), self.get_param("period")
        )


class White(_ParameterizedKernel):
    """White-noise kernel ``sigma^2 delta(x, x')``."""

    pyrox_name: str = "White"
    init_variance: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.white_kernel(X1, X2, self.get_param("variance"))


class Constant(_ParameterizedKernel):
    """Constant kernel ``k(x, x') = sigma^2``."""

    pyrox_name: str = "Constant"
    init_variance: float = 1.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )

    @pyrox_method
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        return _k.constant_kernel(X1, X2, self.get_param("variance"))
