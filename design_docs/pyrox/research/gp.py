"""pyrox.gp reference implementation — GP as Distribution, Hierarchical-First, Guide-Aware
========================================================================================

Minimal, self-contained implementation of the GP building block protocols.
Executable specification of the design doc.

Covers:
- Covariance representations (DenseCov)
- Kernel protocol (KernelProtocol)
- Solver protocol (SolverProtocol)
- GPPrior (numpyro.Distribution over function values)
- ConditionedGP (posterior predictions)
- Guide protocol & implementations (WhitenedDelta, WhitenedMeanField, InducingPoint)
- ComposedGuide (GP guides + generic autoguides)
- Functional helpers (gp_sample, gp_factor)

Architecture:

    +------------------------------------------------------+
    |              User's NumPyro Model                     |
    |  (hyperpriors, multiple GPs, custom likelihoods)      |
    +----+---------------------+-------------------+--------+
         |                     |                   |
    +----v----+          +-----v-----+        +----v-----+
    | GPPrior |          | GPPrior   |        | any dist |
    | (dist)  |          | (dist)    |        | (NumPyro)|
    +----+----+          +-----+-----+        +----------+
         |                     |
    +----v---------------------v-----------------------+
    |              Kernel + Solver                      |
    |     (representation -> solve / log_det)           |
    +--------------------------------------------------+

    Guides (separate, composable):
    +--------------+  +--------------+  +--------------+
    |WhitenedGuide |  |InducingGuide |  |  FlowGuide   |
    |  (exact GP)  |  |  (sparse GP) |  |  (flexible)  |
    +--------------+  +--------------+  +--------------+

NOT production code -- for design exploration and testing only.

Formerly: pyrox-gp/research/base.py
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular


# ─── Forward references ─────────────────────────────────────────────
# In the real library these would be proper imports from pyrox._core,
# pyrox.gp.kernels, etc. Here we sketch the protocols without deps.

R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
R_contra = TypeVar("R_contra", contravariant=True)


# ═════════════════════════════════════════════════════════════════════
# §1  Covariance Representations
# ═════════════════════════════════════════════════════════════════════


class DenseCov(NamedTuple):
    matrix: jnp.ndarray  # (N, N)


class SolveResult(NamedTuple):
    alpha: jnp.ndarray  # (N,)  K_y^{-1} y
    aux: Any  # solver cache (L, preconditioner, etc.)


# ═════════════════════════════════════════════════════════════════════
# §2  Kernel Protocol
# ═════════════════════════════════════════════════════════════════════


@runtime_checkable
class KernelProtocol(Protocol[R_co]):
    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> R_co: ...
    def diagonal(self, X: jnp.ndarray) -> jnp.ndarray: ...


# ═════════════════════════════════════════════════════════════════════
# §3  Solver Protocol
# ═════════════════════════════════════════════════════════════════════


@runtime_checkable
class SolverProtocol(Protocol[R_contra]):
    def solve(self, rep: R_contra, y: jnp.ndarray, noise_var: float) -> SolveResult: ...
    def log_det(self, rep: R_contra, noise_var: float) -> jnp.ndarray: ...
    def predictive_moments(
        self, rep_train: R_contra, cross_cov: Any, test_diag: jnp.ndarray,
        solve_result: SolveResult, noise_var: float, full_cov: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]: ...


# ═════════════════════════════════════════════════════════════════════
# §4  GPPrior — A NumPyro Distribution over Function Values
# ═════════════════════════════════════════════════════════════════════


class GPPrior:
    """numpyro.distributions.Distribution subclass representing:

        f ~ N( m(X),  K(X,X) )

    where K is built by `kernel` and decomposed by `solver`.

    In a NumPyro model:
        f = numpyro.sample("f", GPPrior(kernel, X, solver))

    The user then does whatever they want with f.
    """

    def __init__(
        self,
        kernel: KernelProtocol,
        X: jnp.ndarray,
        solver: SolverProtocol,
        mean_fn: Optional[Callable] = None,
        jitter: float = 1e-6,
    ):
        self.kernel = kernel
        self.X = X
        self.solver = solver
        self.mean_fn = mean_fn
        self.jitter = jitter
        self.N = X.shape[0]

        # Pre-compute representation and decomposition
        self._rep = kernel(X)
        self._mean = jnp.zeros(self.N) if mean_fn is None else mean_fn(X)

    @property
    def event_shape(self):
        return (self.N,)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """Draw f ~ N(m, K) via: f = m + L eps, eps ~ N(0, I_N)"""
        K = self._rep.matrix
        L = jnp.linalg.cholesky(K + self.jitter * jnp.eye(self.N))
        eps = random.normal(key, shape=sample_shape + (self.N,))
        return self._mean + (eps @ L.T)

    def log_prob(self, f: jnp.ndarray) -> jnp.ndarray:
        """log N(f; m, K) using solver for stable computation."""
        residual = f - self._mean
        result = self.solver.solve(self._rep, residual, noise_var=self.jitter)
        ld = self.solver.log_det(self._rep, noise_var=self.jitter)
        return (
            -0.5 * jnp.dot(residual, result.alpha)
            - 0.5 * ld
            - 0.5 * self.N * jnp.log(2.0 * jnp.pi)
        )


# ═════════════════════════════════════════════════════════════════════
# §5  ConditionedGP — Posterior After Observing Data
# ═════════════════════════════════════════════════════════════════════


class ConditionedGP:
    """GP conditioned on observations through a Gaussian likelihood.

    Constructed via: posterior = GPPrior.condition(y, noise_var)

    Provides:
        predict(X_test) -> (mean, var)  closed-form moments
    """

    def __init__(self, prior: GPPrior, y: jnp.ndarray, noise_var: float):
        self.prior = prior
        self.noise_var = noise_var
        residual = y - prior._mean
        self._solve_result = prior.solver.solve(prior._rep, residual, noise_var)

    def predict(
        self, X_test: jnp.ndarray, full_cov: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior predictive moments at X_test."""
        p = self.prior
        cross = p.kernel(X_test, p.X)
        test_diag = p.kernel.diagonal(X_test)
        mu, var = p.solver.predictive_moments(
            p._rep, cross, test_diag, self._solve_result, self.noise_var, full_cov,
        )
        if p.mean_fn is not None:
            mu = mu + p.mean_fn(X_test)
        return mu, var


def _condition(self, y: jnp.ndarray, noise_var: float) -> ConditionedGP:
    """Condition on observations with Gaussian likelihood."""
    return ConditionedGP(self, y, noise_var)

GPPrior.condition = _condition


# ═════════════════════════════════════════════════════════════════════
# §6  Guide Protocol & Implementations
# ═════════════════════════════════════════════════════════════════════


@runtime_checkable
class GPGuideProtocol(Protocol):
    """Guide component for a single GP latent function site."""

    def __call__(self, site_name: str, *args: Any, **kwargs: Any) -> None: ...
    def get_posterior(self, params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]: ...


class WhitenedDeltaGuide:
    """MAP / point estimate in whitened space.

    q(f) = delta(f - L m_tilde)
    """

    def __init__(self, kernel: KernelProtocol, X: jnp.ndarray, jitter: float = 1e-6):
        self.kernel = kernel
        self.X = X
        self.N = X.shape[0]
        rep = kernel(X)
        K = rep.matrix if hasattr(rep, "matrix") else rep
        self._L = jnp.linalg.cholesky(K + jitter * jnp.eye(self.N))

    def __call__(self, site_name: str, *args, **kwargs):
        import numpyro
        import numpyro.distributions as dist
        v_loc = numpyro.param(f"{site_name}_v_loc", jnp.zeros(self.N))
        f = self._L @ v_loc
        numpyro.sample(site_name, dist.Delta(f, event_dim=1))

    def get_posterior(self, params):
        v_loc = params["v_loc"]
        return self._L @ v_loc, jnp.zeros(self.N)


class WhitenedMeanFieldGuide:
    """Diagonal Gaussian in whitened space.

    q(v) = N(m_tilde, diag(sigma^2))  in whitened space
    q(f) = N(L m_tilde, L diag(sigma^2) L^T)  in function space
    """

    def __init__(self, kernel: KernelProtocol, X: jnp.ndarray, jitter: float = 1e-6):
        self.kernel = kernel
        self.X = X
        self.N = X.shape[0]
        rep = kernel(X)
        K = rep.matrix if hasattr(rep, "matrix") else rep
        self._L = jnp.linalg.cholesky(K + jitter * jnp.eye(self.N))

    def __call__(self, site_name: str, *args, **kwargs):
        import numpyro
        import numpyro.distributions as dist
        v_loc = numpyro.param(f"{site_name}_v_loc", jnp.zeros(self.N))
        v_log_std = numpyro.param(f"{site_name}_v_log_std", -2.0 * jnp.ones(self.N))
        v_std = jax.nn.softplus(v_log_std) + 1e-6
        v = numpyro.sample(
            f"{site_name}_v",
            dist.Normal(v_loc, v_std).to_event(1),
            infer={"is_auxiliary": True},
        )
        f = self._L @ v
        numpyro.deterministic(site_name, f)

    def get_posterior(self, params):
        v_loc = params["v_loc"]
        v_log_std = params["v_log_std"]
        v_std = jax.nn.softplus(v_log_std) + 1e-6
        mean_f = self._L @ v_loc
        L_scaled = self._L * v_std
        var_f = jnp.sum(L_scaled**2, axis=-1)
        return mean_f, var_f


class InducingPointGuide:
    """Sparse variational guide using M << N inducing points.

    q(u) = N(m_u, S_u) on M inducing values.
    q(f) = integral p(f|u) q(u) du
    """

    def __init__(
        self,
        kernel: KernelProtocol,
        X: jnp.ndarray,
        num_inducing: int = 32,
        Z_init: Optional[jnp.ndarray] = None,
        whiten: bool = True,
        learn_inducing: bool = True,
        jitter: float = 1e-6,
    ):
        self.kernel = kernel
        self.X = X
        self.N = X.shape[0]
        self.M = num_inducing
        self.whiten = whiten
        self.learn_inducing = learn_inducing
        self.jitter = jitter
        if Z_init is not None:
            self.Z_init = Z_init
        else:
            idx = jnp.linspace(0, self.N - 1, self.M, dtype=int)
            self.Z_init = X[idx]

    def __call__(self, site_name: str, *args, **kwargs):
        import numpyro
        import numpyro.distributions as dist

        M = self.M
        Z = numpyro.param(f"{site_name}_Z", self.Z_init) if self.learn_inducing else self.Z_init
        u_loc = numpyro.param(f"{site_name}_u_loc", jnp.zeros(M))
        u_chol = numpyro.param(
            f"{site_name}_u_chol", 0.01 * jnp.eye(M),
            constraint=dist.constraints.lower_cholesky,
        )
        u = numpyro.sample(
            f"{site_name}_u",
            dist.MultivariateNormal(u_loc, scale_tril=u_chol),
            infer={"is_auxiliary": True},
        )

        if self.whiten:
            rep_uu = self.kernel(Z)
            K_uu = rep_uu.matrix if hasattr(rep_uu, "matrix") else rep_uu
            L_uu = jnp.linalg.cholesky(K_uu + self.jitter * jnp.eye(M))
            u_actual = L_uu @ u
        else:
            u_actual = u
            rep_uu = self.kernel(Z)
            K_uu = rep_uu.matrix if hasattr(rep_uu, "matrix") else rep_uu
            L_uu = jnp.linalg.cholesky(K_uu + self.jitter * jnp.eye(M))

        K_fu = self.kernel(self.X, Z)
        K_fu_mat = K_fu.matrix if hasattr(K_fu, "matrix") else K_fu
        alpha_u = solve_triangular(
            L_uu.T, solve_triangular(L_uu, u_actual, lower=True), lower=False,
        )
        f_mean = K_fu_mat @ alpha_u
        numpyro.deterministic(site_name, f_mean)


# ─── Guide Composition ──────────────────────────────────────────────


class ComposedGuide:
    """Compose GP-specific guides with generic NumPyro autoguides."""

    def __init__(
        self,
        gp_guides: Dict[str, GPGuideProtocol],
        model: Optional[Callable] = None,
        auto_guide_cls: Optional[type] = None,
        auto_guide_kwargs: Optional[Dict] = None,
    ):
        self.gp_guides = gp_guides
        self.model = model
        self.auto_guide_cls = auto_guide_cls
        self.auto_guide_kwargs = auto_guide_kwargs or {}
        self._auto_guide = None

    def _build_auto_guide(self, model):
        if self.auto_guide_cls is None or model is None:
            return None
        from numpyro.handlers import block
        gp_site_names = set(self.gp_guides.keys())
        aux_names = {f"{name}_v" for name in gp_site_names}
        aux_names |= {f"{name}_u" for name in gp_site_names}
        all_blocked = gp_site_names | aux_names
        blocked_model = block(model, hide_fn=lambda site: site["name"] in all_blocked)
        return self.auto_guide_cls(blocked_model, **self.auto_guide_kwargs)

    def __call__(self, *args, **kwargs):
        for site_name, gp_guide in self.gp_guides.items():
            gp_guide(site_name, *args, **kwargs)
        if self._auto_guide is None and self.model is not None:
            self._auto_guide = self._build_auto_guide(self.model)
        if self._auto_guide is not None:
            self._auto_guide(*args, **kwargs)


# ═════════════════════════════════════════════════════════════════════
# §7  Functional Helpers
# ═════════════════════════════════════════════════════════════════════


def gp_sample(
    name: str,
    kernel: KernelProtocol,
    X: jnp.ndarray,
    solver: SolverProtocol,
    mean_fn: Optional[Callable] = None,
    jitter: float = 1e-6,
    obs: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Sample a GP latent function inside a NumPyro model.

    Equivalent to: f = numpyro.sample(name, GPPrior(kernel, X, solver, mean_fn))
    """
    import numpyro
    prior = GPPrior(kernel, X, solver, mean_fn, jitter)
    return numpyro.sample(name, prior, obs=obs)


def gp_factor(
    name: str,
    kernel: KernelProtocol,
    X: jnp.ndarray,
    y: jnp.ndarray,
    noise_var: float,
    solver: SolverProtocol,
    mean_fn: Optional[Callable] = None,
) -> None:
    """Register GP log marginal likelihood as a NumPyro factor.

    Use for the collapsed form where f is integrated out analytically.
    Only valid for Gaussian likelihoods.
    """
    import numpyro

    N = X.shape[0]
    rep = kernel(X)
    y_centered = y if mean_fn is None else y - mean_fn(X)
    result = solver.solve(rep, y_centered, noise_var)
    ld = solver.log_det(rep, noise_var)
    lml = (
        -0.5 * jnp.dot(y_centered, result.alpha)
        - 0.5 * ld
        - 0.5 * N * jnp.log(2.0 * jnp.pi)
    )
    numpyro.factor(name, lml)
