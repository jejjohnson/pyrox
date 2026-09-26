"""Concrete kernel classes — ``Parameterized`` wrappers over the math primitives.

Each class registers its hyperparameters with constraints through
`pyrox._core.Parameterized`, so users can attach priors and
autoguides via `set_prior` / `autoguide` and flip between
prior/guide modes with `set_mode`. The numerical body delegates to
the pure closed-form functions in `kernellib.functional`.

Kernels with a static structural parameter (``Matern.nu``,
``Polynomial.degree``) take that parameter as a class field rather than
a registered JAX param — those numbers choose code paths, not
optimization targets.

Scalable matrix construction (mixed-precision accumulation, implicit
operators, batched matvec) lives in `kernellib`; these wrappers own
the NumPyro-aware surface only.
``frozen()`` resolves the hyperparameters once and returns the plain
kernellib kernel of the same family, which is what kernellib's spectral
methods, feature maps and operators take.
"""

from __future__ import annotations

from typing import Any, ClassVar

import jax.numpy as jnp
import kernellib as kl
import numpyro.distributions as dist
from jaxtyping import Array, Float
from kernellib import functional as _k
from pyrox._core import Parameterized, pyrox_method

from pyrox_gp._protocols import Kernel


class _ParameterizedKernel(Parameterized, Kernel):
    """Shared base — mixes `Parameterized` state with the `Kernel`.

    Subclasses only need to implement `setup` (register params +
    priors) and `__call__` (evaluate the math primitive). Setting
    ``_frozen_cls`` / ``_frozen_params`` / ``_frozen_static`` enables
    `frozen`.
    """

    # The kernellib class `frozen` builds, the registered params it reads,
    # and the static fields it copies. Field names match kernellib's.
    _frozen_cls: ClassVar[type[kl.AbstractKernel] | None] = None
    _frozen_params: ClassVar[tuple[str, ...]] = ()
    _frozen_static: ClassVar[tuple[str, ...]] = ()

    @pyrox_method
    def frozen(self, **resolved: Any) -> kl.AbstractKernel:
        """The plain kernellib kernel at the current hyperparameter values.

        Each registered parameter is resolved once, inside this kernel's
        per-call context, so a prior contributes one sample site (under
        ``numpyro.handlers.trace``) and one draw (under ``seed``); call it
        inside an enclosing `_kernel_context` to share that draw with other
        evaluations. The result has no priors and no NumPyro state: it is the
        object kernellib's spectral methods, feature maps and operators take.

        Args:
            **resolved: Values to use instead of resolving a parameter, e.g.
                the ``variance`` / ``lengthscale`` a conditioned GP cached.
                Parameters given here are not read, so no site is registered
                for them.

        Returns:
            The kernellib kernel of the same family.

        Raises:
            NotImplementedError: If the class has no kernellib counterpart.
            ValueError: For a name in ``resolved`` the kernel does not have.
        """
        if self._frozen_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no kernellib counterpart to freeze to."
            )
        unknown = set(resolved) - set(self._frozen_params)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no parameters {sorted(unknown)}; "
                f"expected a subset of {self._frozen_params}."
            )
        params = {
            name: resolved[name] if name in resolved else self.get_param(name)
            for name in self._frozen_params
        }
        static = {name: getattr(self, name) for name in self._frozen_static}
        return self._frozen_cls(**params, **static)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Stationary-kernel fast diagonal: constant variance on every point.

        Subclasses that are not strictly stationary (``Linear``,
        ``Polynomial``) override with a point-wise computation.
        """
        n = X.shape[0]
        return self.get_param("variance") * jnp.ones(n, dtype=X.dtype)


class RBF(_ParameterizedKernel):
    """Radial basis function (squared exponential) kernel.

    ``input_dim``: set to the input dimension ``D`` to fit a separate
    lengthscale per input dimension (ARD). Leave as ``None`` for a single
    isotropic lengthscale.
    """

    _frozen_cls = kl.RBF
    _frozen_params = ("variance", "lengthscale")

    pyrox_name: str = "RBF"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    input_dim: int | None = None

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        lengthscale = (
            jnp.asarray(self.init_lengthscale)
            if self.input_dim is None
            else jnp.full((self.input_dim,), self.init_lengthscale)
        )
        self.register_param(
            "lengthscale",
            lengthscale,
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

    ``input_dim``: set to the input dimension ``D`` to fit a separate
    lengthscale per input dimension (ARD). Leave as ``None`` for a single
    isotropic lengthscale.
    """

    _frozen_cls = kl.Matern
    _frozen_params = ("variance", "lengthscale")
    _frozen_static = ("nu",)

    pyrox_name: str = "Matern"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    nu: float = 2.5
    input_dim: int | None = None

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        lengthscale = (
            jnp.asarray(self.init_lengthscale)
            if self.input_dim is None
            else jnp.full((self.input_dim,), self.init_lengthscale)
        )
        self.register_param(
            "lengthscale",
            lengthscale,
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

    _frozen_cls = kl.Periodic
    _frozen_params = ("variance", "lengthscale", "period")

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
    """Linear kernel ``sigma^2 x^T x' + bias``.

    ``bias`` is constrained nonnegative because ``k = sigma^2 X X^T + b 1 1^T``
    is only PSD for ``b >= 0`` (e.g. ``X = 0`` gives eigenvalue ``N*b``).
    """

    _frozen_cls = kl.Linear
    _frozen_params = ("variance", "bias")

    pyrox_name: str = "Linear"
    init_variance: float = 1.0
    init_bias: float = 0.0

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "bias",
            jnp.asarray(self.init_bias),
            constraint=dist.constraints.nonnegative,
        )

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
    """Rational quadratic kernel.

    ``input_dim``: set to the input dimension ``D`` to fit a separate
    lengthscale per input dimension (ARD). Leave as ``None`` for a single
    isotropic lengthscale.
    """

    _frozen_cls = kl.RationalQuadratic
    _frozen_params = ("variance", "lengthscale", "alpha")

    pyrox_name: str = "RationalQuadratic"
    init_variance: float = 1.0
    init_lengthscale: float = 1.0
    init_alpha: float = 1.0
    input_dim: int | None = None

    def setup(self) -> None:
        self.register_param(
            "variance",
            jnp.asarray(self.init_variance),
            constraint=dist.constraints.positive,
        )
        lengthscale = (
            jnp.asarray(self.init_lengthscale)
            if self.input_dim is None
            else jnp.full((self.input_dim,), self.init_lengthscale)
        )
        self.register_param(
            "lengthscale",
            lengthscale,
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
    an optimization target). ``bias`` is constrained nonnegative — the
    ``degree=1`` case reduces to `Linear` and has the same
    PSD-requires-``b>=0`` failure mode.
    """

    _frozen_cls = kl.Polynomial
    _frozen_params = ("variance", "bias")
    _frozen_static = ("degree",)

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
        self.register_param(
            "bias",
            jnp.asarray(self.init_bias),
            constraint=dist.constraints.nonnegative,
        )

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

    _frozen_cls = kl.Cosine
    _frozen_params = ("variance", "period")

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

    _frozen_cls = kl.White
    _frozen_params = ("variance",)

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

    _frozen_cls = kl.Constant
    _frozen_params = ("variance",)

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
