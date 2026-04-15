"""
Sparse Variational Gaussian Processes (SVGP) with Extensive Equation & Shape Annotations
=======================================================================================

This module implements a Sparse Variational Gaussian Process (SVGP) regression model
together with a spectrum of variational guide families:

    delta, mean_field, low_rank, full_rank, orthogonal_decoupled, flow

This version FURTHER enhances the prior iteration by adding:
    - EVEN MORE granular docstrings and inline comments.
    - Explicit equation references (Eq. *) in every function.
    - Shape tracking for all intermediate tensors (with consistent notation).
    - Clarification of computational / memory complexity where relevant.

Core Mathematical Objects
-------------------------
Inducing Points / Variables:
    (1)  u = f(Z)                         where Z ∈ ℝ^{M×D}, u ∈ ℝ^{M}

GP Prior:
    (2)  p(u) = 𝓝(0, K_ZZ),   K_ZZ[i,j] = k(z_i, z_j)

Data Conditional (exact GP):
    (3)  p(f | u) = 𝓝( K_XZ K_ZZ^{-1} u,  K_XX - Q_XX ),
         Q_XX = K_XZ K_ZZ^{-1} K_ZX,
         where K_XZ ∈ ℝ^{B×M}, K_XX ∈ ℝ^{B×B}, f ∈ ℝ^{B}

Whitening:
    (4)  K_ZZ = L_Z L_Zᵀ,  u = L_Z v,  v ~ 𝓝(0, I_M)

Projection Matrix:
    (7)  A_X = K_XZ K_ZZ^{-1} = ( L_Z^{-T} ( L_Z^{-1} K_XZᵀ ) )ᵀ

Latent Mean (for a given u):
    (8)  μ_f = A_X u

Conditional Variance (diagonal approximation in training loop):
    (9)  Var_cond(x_i) = k(x_i, x_i) - (A_X K_ZX)_{ii}

Gaussian Variational Family Moments:
    (10)  E_q[f] = A_X m_u,  with  m_u = L_Z m̃
    (11)  diag Var_q[f] ≈ diag(K_XX - A_X K_ZX) + diag(A_X S_u A_Xᵀ)
    (12)  S_u = L_Z S̃ L_Zᵀ  (lifted covariance in function space)
    (13)  Low-rank S̃ = L_r L_rᵀ + diag(σ²)
    (16)  Full-rank S̃ = L̃ L̃ᵀ (Cholesky variational factor)

Orthogonally Decoupled Mean Extension:
    (14)  Extra mean = A_m (L_Zm a)  using separate inducing set Z_m

Flow-based Non-Gaussian Guide:
    (15)  v = T_K ∘ ... ∘ T_1 (z),  z ~ 𝓝(0, I_M),  T_k invertible

Evidence Lower Bound (ELBO):
    (5)  ELBO = Σ_{i=1}^N E_q[ log p(y_i | f_i) ] - KL[q(v) || p(v)]
    (6)  With mini-batching (batch size B): scale likelihood term by N / B

Shape Legend
------------
B   : mini-batch size during optimization
N   : total dataset size
M   : number of inducing points (covariance set)
M_m : number of mean inducing points (decoupled mean; optional)
D   : input feature dimensionality
r   : low-rank dimensionality in low_rank guide (r ≤ M)
S   : number of Monte Carlo samples (for flow predictions)
N*  : number of test points at prediction time

All contractions use einops.einsum for clarity; no implicit broadcasting hides semantics.

WARNING
-------
The full-rank (Cholesky) guide has O(M^2) parameters and certain predictive steps
scale as O(M^3) if full covariance outputs are requested. Use with caution for large M.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Literal

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as T
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from einops import rearrange, einsum


# ======================================================================================
# Kernel Abstractions
# ======================================================================================

class Kernel:
    """
    Abstract covariance kernel interface.

    Required
    --------
    __call__(X, X2=None) -> K

    Parameters
    ----------
    X : (N,D) float
        Input locations.
    X2 : (M,D) float | None
        If None, compute auto-covariance on X (so M=N).
    Returns
    -------
    K : (N,M) float
        Covariance matrix where K[i,j] = k(X[i], X2[j]).

    Notes
    -----
    Different kernel forms define different smoothness priors. Implementations
    should ensure numerical stability (e.g. no negative variances).
    """
    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> jnp.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RBF(Kernel):
    r"""
    Squared Exponential / RBF / Gaussian kernel.

    Definition (Eq. K1)
    -------------------
    k(x,x') = σ² exp( -½ Σ_d ( (x_d - x'_d)^2 / ℓ_d^2 ) )

    Parameters
    ----------
    variance : float
        σ² amplitude (must remain > 0 conceptually; handled implicitly).
    lengthscale : float | jnp.ndarray
        ℓ lengthscale(s); can be scalar or per-dimension vector broadcastable to D.

    Complexity
    ----------
    Time: O(N M D)
    Memory: O(N M)

    Returns
    -------
    K : (N,M)
    """
    variance: float = 1.0
    lengthscale: float | jnp.ndarray = 1.0

    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        X2 = X if X2 is None else X2
        ℓ = jnp.asarray(self.lengthscale)

        # Scale inputs by lengthscale per dimension.
        # Shapes:
        #   Xs  : (N,D)
        #   X2s : (M,D)
        Xs = X / ℓ
        X2s = X2 / ℓ

        # Compute squared distances via norm trick:
        # sqd[n,m] = ||Xs[n]||^2 - 2 Xs[n]·X2s[m] + ||X2s[m]||^2
        x_norm = einsum(Xs, "n d -> n 1", lambda t: jnp.sum(t**2, axis=-1))  # (N,1)
        x2_norm = einsum(X2s, "m d -> m", lambda t: jnp.sum(t**2, axis=-1))  # (M,)
        cross = einsum(Xs, X2s, "n d, m d -> n m")                           # (N,M)
        sqd = x_norm - 2 * cross + x2_norm                                   # (N,M)

        # Apply the radial basis function.
        return self.variance * jnp.exp(-0.5 * sqd)


@dataclass
class Matern32(Kernel):
    r"""
    Matérn ν = 3/2 kernel.

    Definition (Eq. K2)
    -------------------
    k(r) = σ² (1 + √3 r) exp(-√3 r)
      where r = sqrt( Σ_d ( (x_d - x'_d)^2 / ℓ_d^2 ) )

    Parameters
    ----------
    variance : float
        σ² amplitude.
    lengthscale : float | jnp.ndarray
        ℓ lengthscale(s).

    Complexity
    ----------
    Time:  O(N M D)
    Memory: O(N M)
    """
    variance: float = 1.0
    lengthscale: float | jnp.ndarray = 1.0

    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        X2 = X if X2 is None else X2
        ℓ = jnp.asarray(self.lengthscale)

        # Pairwise differences diff[n,m,d] = X[n,d] - X2[m,d]
        diff = rearrange(X, "n d -> n 1 d") - rearrange(X2, "m d -> 1 m d")  # (N,M,D)

        # Scaled Euclidean distance r[n,m]
        r = jnp.sqrt(jnp.sum((diff / ℓ) ** 2, axis=-1) + 1e-12)  # (N,M)
        c = jnp.sqrt(3.0)
        return self.variance * (1.0 + c * r) * jnp.exp(-c * r)


# ======================================================================================
# SVGP Core Model
# ======================================================================================

GuideType = Literal[
    "delta",
    "mean_field",
    "low_rank",
    "full_rank",                # full covariance variational Cholesky
    "orthogonal_decoupled",
    "flow",
]


@dataclass
class SVGP:
    """
    Sparse Variational GP Regression Model (training graph without variational assumptions).

    Purpose
    -------
    Encodes p(y, v_s) under whitening (Eq. 4) for use with arbitrary guides q(v_s).
    This object does NOT embed variational parameters; those are external.

    Parameters
    ----------
    kernel : Kernel
        Covariance function k(·,·).
    inducing_inputs : (M,D)
        Inducing point locations Z used for approximation (u = f(Z)).
    noise_prior : Distribution | None
        Prior on observation noise σ_n. Default Exponential(1.0) if not fixed.
    mean_function : callable | None
        Optional m(x); if provided, added to latent mean.
    jitter : float
        Stabilizer > 0 added to diagonal of K_ZZ for Cholesky.
    fixed_noise : float | None
        If provided, noise std stays constant at this value.
    guide_type : GuideType
        Stored for convenience (guides are decoupled).
    low_rank : int
        Rank for low-rank variational family (cap at M).
    mean_inducing_inputs : (M_m,D) | None
        Optional mean basis for orthogonally decoupled strategy.
    num_flow_layers : int
        Number of affine + tanh blocks for flow guide.

    Important Shapes
    ----------------
    Z_s : (M,D)
    v_s : (M,) whitened latent
    u_s : (M,) unwhitened (Eq. 4)
    """

    kernel: Kernel
    inducing_inputs: jnp.ndarray
    noise_prior: Optional[dist.Distribution] = None
    mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    jitter: float = 1e-6
    fixed_noise: Optional[float] = None
    guide_type: GuideType = "mean_field"
    low_rank: int = 8
    mean_inducing_inputs: Optional[jnp.ndarray] = None
    num_flow_layers: int = 3

    def model(self, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int] = None) -> None:
        """
        Probabilistic model graph.

        Inputs
        ------
        X : (B,D)
        y : (B,)
        full_N : int | None
            Total dataset size for mini-batch scaling (Eq. 6). If None, no scaling.

        Steps & Equations
        -----------------
        1. K_ZZ = k(Z,Z) + jitter I               (for stability)
        2. Cholesky: K_ZZ = L_Z L_Zᵀ
        3. Sample whitened v_s ~ 𝓝(0,I)           (Eq. 4 prior)
        4. If noise not fixed, sample noise.
        5. Unwhiten u_s = L_Z v_s                 (Eq. 4)
        6. K_XZ = k(X,Z)
        7. A_X = K_XZ K_ZZ^{-1}                   (Eq. 7) via triangular solves
        8. mean_f = A_X u_s                       (Eq. 8)
        9. If orthogonal decoupling: provide placeholder a_decoupled_model (zeros)
           (Guide will substitute actual parameters).
        10. Add optional mean_function(X)
        11. Cond variance diag: Var_cond (Eq. 9)
        12. Likelihood: y ~ Normal( mean_f, sqrt(Var_cond + noise²) ), scaled (Eq. 6)

        Outputs
        -------
        Creates stochastic nodes 'v_s' and 'y'; deterministic 'a_decoupled_model' if required.

        Complexity
        ----------
        Cholesky: O(M^3)
        Projection per batch: O(B M^2) (due to triangular solves & matmuls)
        """
        Zs = self.inducing_inputs  # (M,D)
        M = Zs.shape[0]

        # (1) + (2): Prior covariance & Cholesky
        K_ZZ = self.kernel(Zs) + self.jitter * jnp.eye(M)   # (M,M)
        L_Z = jnp.linalg.cholesky(K_ZZ)                     # (M,M)

        # (3) Whitened prior sample
        v_s = numpyro.sample("v_s", dist.MultivariateNormal(jnp.zeros(M), jnp.eye(M)))  # (M,)

        # (4) Noise parameter / sample
        if self.fixed_noise is not None:
            noise = self.fixed_noise
        else:
            noise = numpyro.sample(
                "noise",
                self.noise_prior if self.noise_prior is not None else dist.Exponential(1.0),
            )

        # (5) Unwhiten: u_s = L_Z v_s
        u_s = einsum(L_Z, v_s, "m k, k -> m")  # (M,)

        # (6) Cross-cov
        K_XZ = self.kernel(X, Zs)  # (B,M)

        # (7) Projection matrix A_X = K_XZ K_ZZ^{-1}
        tmp = solve_triangular(L_Z, K_XZ.T, lower=True)       # (M,B)
        A_X = solve_triangular(L_Z.T, tmp, lower=False).T     # (B,M)

        # (8) Mean latent function
        mean_f = einsum(A_X, u_s, "b m, m -> b")  # (B,)

        # (9) Placeholder for decoupled mean (guide sets actual 'a')
        if self.guide_type == "orthogonal_decoupled":
            Zm = self.mean_inducing_inputs if self.mean_inducing_inputs is not None else Zs
            _ = numpyro.deterministic("a_decoupled_model", jnp.zeros(Zm.shape[0]))

        # (10) Add deterministic mean function
        if self.mean_function is not None:
            mean_f = mean_f + self.mean_function(X)

        # (11) Conditional variance (diagonal)
        # Var_cond(x_i) = k(x_i,x_i) - (A_X K_ZX)_{ii}
        K_XX_diag = jnp.diag(self.kernel(X))          # (B,)
        AKZX = einsum(A_X, K_XZ, "b m, b m -> b")     # (B,)
        var_f = jnp.clip(K_XX_diag - AKZX, a_min=0.0) # (B,)

        # (12) Likelihood with scaling
        scale = 1.0 if full_N is None else full_N / X.shape[0]
        numpyro.sample(
            "y",
            dist.Normal(mean_f, jnp.sqrt(var_f + noise**2)),
            obs=y,
            infer={"scale": scale},
        )

    def make_svi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        guide_fn: Callable,
        learning_rate: float = 1e-2,
        seed: int = 0,
        full_N: Optional[int] = None,
    ):
        """
        Construct the SVI optimizer components (model + external guide).

        Parameters
        ----------
        X : (B,D) initial batch (or full dataset)
        y : (B,)
        guide_fn : callable
            Function signature guide_fn(svgp, X, y, full_N) -> None
        learning_rate : float
            Adam step size.
        seed : int
            PRNG seed.
        full_N : int | None
            Total dataset size (for mini-batch scaling; Eq. 6).

        Returns
        -------
        svi  : SVI object
        state: SVIState (initialized parameters & optimizer state)

        Notes
        -----
        This does not run training; call svi.update(...) inside a loop.
        """
        model = lambda Xb, yb: self.model(Xb, yb, full_N=full_N)
        guide = lambda Xb, yb: guide_fn(self, Xb, yb, full_N)
        svi = SVI(model, guide, Adam(learning_rate), loss=Trace_ELBO())
        rng = random.PRNGKey(seed)
        state = svi.init(rng, X, y)
        return svi, state


# ======================================================================================
# Helper Functions (Projection, Covariances, Variance Contributions)
# ======================================================================================

def compute_cholesky(kernel: Kernel, Z: jnp.ndarray, jitter: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build K_ZZ + jitter I and its Cholesky (Eq. 2 + stabilization).

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    Z : (M,D)
        Inducing locations.
    jitter : float
        Positive scalar added to diagonal to ensure PD.

    Returns
    -------
    K_ZZ : (M,M)
    L_Z  : (M,M) lower-triangular (Cholesky factor)

    Complexity
    ----------
    Time: O(M^2 D + M^3)  (covariance + Cholesky)
    """
    K_ZZ = kernel(Z) + jitter * jnp.eye(Z.shape[0])  # (M,M)
    L_Z = jnp.linalg.cholesky(K_ZZ)                  # (M,M)
    return K_ZZ, L_Z


def project_A(K_XZ: jnp.ndarray, L_Z: jnp.ndarray) -> jnp.ndarray:
    """
    Compute projection A_X = K_XZ K_ZZ^{-1} (Eq. 7) using Cholesky solves.

    Parameters
    ----------
    K_XZ : (B,M)
        Cross-covariance between batch inputs X and inducing Z.
    L_Z : (M,M)
        Cholesky of K_ZZ.

    Returns
    -------
    A_X : (B,M)

    Steps
    -----
    1. tmp = L_Z^{-1} K_XZᵀ                -> (M,B)
    2. A_Xᵀ = L_Z^{-T} tmp                 -> (M,B)
    3. A_X = (A_Xᵀ)ᵀ                       -> (B,M)

    Numerical Stability
    -------------------
    Triangular solves more stable than explicit inversion.

    Complexity
    ----------
    O(B M^2)
    """
    tmp = solve_triangular(L_Z, K_XZ.T, lower=True)           # (M,B)
    A_X = solve_triangular(L_Z.T, tmp, lower=False).T         # (B,M)
    return A_X


def conditional_base_variance(kernel: Kernel, X: jnp.ndarray, K_XZ: jnp.ndarray, A_X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute base diagonal conditional variance (Eq. 9) ignoring variational uncertainty.

        Var_cond(x_i) = k(x_i,x_i) - (A_X K_ZX)_{ii}

    Parameters
    ----------
    kernel : Kernel
    X : (B,D)
    K_XZ : (B,M)
    A_X : (B,M)

    Returns
    -------
    var_base : (B,)

    Notes
    -----
    Negative values can occur numerically; clamp at 0.
    """
    K_XX_diag = jnp.diag(kernel(X))            # (B,)
    AKZX = einsum(A_X, K_XZ, "b m, b m -> b")  # (B,)
    return jnp.clip(K_XX_diag - AKZX, 0.0)


def unwhiten_mean(m_tilde: jnp.ndarray, L_Z: jnp.ndarray) -> jnp.ndarray:
    """
    Unwhiten variational mean (Eq. 10 portion):

        m_u = L_Z m̃

    Parameters
    ----------
    m_tilde : (M,)
    L_Z : (M,M)

    Returns
    -------
    m_u : (M,)
    """
    return einsum(L_Z, m_tilde, "m k, k -> m")


def full_S_u_mean_field(diag_std: jnp.ndarray, L_Z: jnp.ndarray) -> jnp.ndarray:
    """
    Construct S_u for mean-field variational covariance:

        S̃ = diag(σ²)
        (12) S_u = L_Z S̃ L_Zᵀ = (L_Z diag(σ)) (L_Z diag(σ))ᵀ.

    Parameters
    ----------
    diag_std : (M,)
        σ (standard deviations)
    L_Z : (M,M)

    Returns
    -------
    S_u : (M,M)
    """
    LZ_scaled = L_Z * diag_std
    return einsum(LZ_scaled, LZ_scaled, "m k, n k -> m n")


def full_S_u_low_rank(Lr: jnp.ndarray, diag_std: jnp.ndarray, L_Z: jnp.ndarray) -> jnp.ndarray:
    """
    Construct S_u for low-rank + diagonal covariance:

        (13) S̃ = L_r L_rᵀ + diag(σ²)
        (12) S_u = L_Z S̃ L_Zᵀ = (L_Z L_r)(L_Z L_r)ᵀ + (L_Z diag(σ))(L_Z diag(σ))ᵀ

    Parameters
    ----------
    Lr : (M,r)
    diag_std : (M,)
    L_Z : (M,M)

    Returns
    -------
    S_u : (M,M)
    """
    LZLr = einsum(L_Z, Lr, "m k, k r -> m r")       # (M,r)
    LZdiag = L_Z * diag_std                         # (M,M)
    low_rank_part = einsum(LZLr, LZLr, "m r, n r -> m n")
    diag_part = einsum(LZdiag, LZdiag, "m k, n k -> m n")
    return low_rank_part + diag_part


def full_S_u_full_rank(L_tilde: jnp.ndarray, L_Z: jnp.ndarray) -> jnp.ndarray:
    """
    Construct S_u for full-rank Cholesky variational covariance (Eq. 16):

        S̃ = L̃ L̃ᵀ  (L̃ lower-triangular)
        (12) S_u = L_Z S̃ L_Zᵀ = (L_Z L̃)(L_Z L̃)ᵀ

    Parameters
    ----------
    L_tilde : (M,M)  lower-triangular
    L_Z : (M,M)

    Returns
    -------
    S_u : (M,M)
    """
    LZLt = einsum(L_Z, L_tilde, "m k, k n -> m n")  # (M,M)
    return einsum(LZLt, LZLt, "m k, n k -> m n")    # (M,M)


def diag_variance_contrib_mean_field(A_X: jnp.ndarray, L_Z: jnp.ndarray, diag_std: jnp.ndarray) -> jnp.ndarray:
    """
    Compute diagonal contribution of A_X S_u A_Xᵀ for mean-field:

        S_u = (L_Z diag(σ))(L_Z diag(σ))ᵀ
        Let W = A_X L_Z  ⇒  A_X S_u A_Xᵀ = (W diag(σ))(W diag(σ))ᵀ
        diag(...) = Σ_k ( W[i,k] σ[k] )²

    Parameters
    ----------
    A_X : (B,M)
    L_Z : (M,M)
    diag_std : (M,)

    Returns
    -------
    diag_contrib : (B,)
    """
    W = einsum(A_X, L_Z, "b m, m k -> b k")      # (B,M)
    W_scaled = W * diag_std                      # (B,M)
    return einsum(W_scaled, W_scaled, "b k, b k -> b")


def diag_variance_contrib_low_rank(A_X: jnp.ndarray, L_Z: jnp.ndarray, Lr: jnp.ndarray, diag_std: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal variance for low-rank + diagonal:

        S_u = (L_Z L_r)(L_Z L_r)ᵀ + (L_Z diag(σ))(L_Z diag(σ))ᵀ
        Let ALZ = A_X L_Z.
        diag(A_X S_u A_Xᵀ) =
            || ALZ L_r ||²_row + || (ALZ * σ) ||²_row

    Parameters
    ----------
    A_X : (B,M)
    L_Z : (M,M)
    Lr : (M,r)
    diag_std : (M,)

    Returns
    -------
    diag_contrib : (B,)
    """
    ALZ = einsum(A_X, L_Z, "b m, m k -> b k")            # (B,M)
    ALZLr = einsum(ALZ, Lr, "b k, k r -> b r")           # (B,r)
    low_rank_diag = einsum(ALZLr, ALZLr, "b r, b r -> b")
    diag_part = einsum(ALZ, diag_std * ALZ, "b k, b k -> b")
    return low_rank_diag + diag_part


def diag_variance_contrib_full_rank(A_X: jnp.ndarray, L_Z: jnp.ndarray, L_tilde: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal variance for full-rank Cholesky guide:

        S_u = (L_Z L̃)(L_Z L̃)ᵀ
        Let B = A_X L_Z L̃ ⇒ A_X S_u A_Xᵀ = B Bᵀ
        diag = row-wise squared norms of B.

    Parameters
    ----------
    A_X : (B,M)
    L_Z : (M,M)
    L_tilde : (M,M) lower-triangular

    Returns
    -------
    diag_contrib : (B,)
    """
    ALZ = einsum(A_X, L_Z, "b m, m k -> b k")     # (B,M)
    B = einsum(ALZ, L_tilde, "b k, k m -> b m")   # (B,M)
    return einsum(B, B, "b m, b m -> b")


# ======================================================================================
# Variational Guide Functions
# ======================================================================================

def guide_delta(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Delta (deterministic) guide:

        q(v) = δ(v - m̃)

    Parameters
    ----------
    delta_mean : (M,) variational mean

    Comments
    --------
    No covariance; no uncertainty encoded at inducing layer. Fast but crude.
    """
    M = svgp.inducing_inputs.shape[0]
    m = numpyro.param("delta_mean", jnp.zeros(M))
    numpyro.sample("v_s", dist.Delta(m, event_dim=1))
    if svgp.fixed_noise is None:
        noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
        numpyro.deterministic("noise", noise)


def guide_mean_field(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Mean-field guide:

        q(v) = 𝓝(m̃, diag(σ²))

    Parameters
    ----------
    mf_mean : (M,)
    mf_raw_std : (M,)   → σ = softplus(raw)+ε

    Notes
    -----
    Captures independent uncertainty per inducing dimension; no correlations.
    """
    M = svgp.inducing_inputs.shape[0]
    m = numpyro.param("mf_mean", jnp.zeros(M))
    raw = numpyro.param("mf_raw_std", -2.0 * jnp.ones(M))
    std = jax.nn.softplus(raw) + 1e-6
    numpyro.sample("v_s", dist.Normal(m, std).to_event(1))
    if svgp.fixed_noise is None:
        noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
        numpyro.deterministic("noise", noise)


def guide_low_rank(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Low-rank + diagonal guide (Eq. 13):

        q(v) = 𝓝(m̃, L_r L_rᵀ + diag(σ²))
        Sampling: v = m̃ + L_r ε_r + diag_std ε_d

    Parameters
    ----------
    lr_mean : (M,)
    lr_Lr : (M,r)
    lr_raw_diag_std : (M,) → diag_std

    Remarks
    -------
    Balances expressivity and parameter cost: O(M r + M).
    """
    M = svgp.inducing_inputs.shape[0]
    r = min(svgp.low_rank, M)
    m = numpyro.param("lr_mean", jnp.zeros(M))
    Lr = numpyro.param("lr_Lr", 0.01 * jnp.ones((M, r)))
    raw = numpyro.param("lr_raw_diag_std", -2.0 * jnp.ones(M))
    diag_std = jax.nn.softplus(raw) + 1e-6
    eps_r = numpyro.sample("eps_r", dist.Normal(0, 1).expand([r]).to_event(1))  # (r,)
    eps_d = numpyro.sample("eps_d", dist.Normal(0, 1).expand([M]).to_event(1))  # (M,)
    low_rank_part = einsum(Lr, eps_r, "m r, r -> m")                            # (M,)
    v = m + low_rank_part + diag_std * eps_d                                    # (M,)
    numpyro.deterministic("v_s", v)
    numpyro.sample("v_s", dist.Delta(v, event_dim=1))
    if svgp.fixed_noise is None:
        noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
        numpyro.deterministic("noise", noise)


def guide_full_rank(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Full-rank Cholesky guide (Eq. 16):

        q(v) = 𝓝(m̃, S̃),  S̃ = L̃ L̃ᵀ
        L̃ lower-triangular with M(M+1)/2 parameters.

    Parameters
    ----------
    fr_mean : (M,)
    fr_chol : (M,M) lower-triangular scale_tril of S̃

    Trade-offs
    ----------
    + Maximum Gaussian flexibility
    - O(M^2) storage, O(M^3) Cholesky-based manipulations in predictions
    """
    M = svgp.inducing_inputs.shape[0]
    m = numpyro.param("fr_mean", jnp.zeros(M))
    L_tilde = numpyro.param(
        "fr_chol",
        0.01 * jnp.eye(M),
        constraint=dist.constraints.lower_cholesky,
    )
    numpyro.sample("v_s", dist.MultivariateNormal(m, scale_tril=L_tilde))
    if svgp.fixed_noise is not None:
        return
    noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
    numpyro.deterministic("noise", noise)


def guide_orthogonal_decoupled(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Orthogonally Decoupled guide (Eq. 14 extension):

        q(v_s) mean-field for covariance structure on Z_s.
        Separate whitened mean parameters a over Z_m.

    Parameters
    ----------
    od_mf_mean : (M_s,)
    od_mf_raw_std : (M_s,) → std
    od_a_mean : (M_m,) whitened mean coordinates

    Benefit
    -------
    Rich mean representation without full covariance explosion.
    """
    Zm = svgp.mean_inducing_inputs if svgp.mean_inducing_inputs is not None else svgp.inducing_inputs
    M_s = svgp.inducing_inputs.shape[0]
    M_m = Zm.shape[0]
    m_s = numpyro.param("od_mf_mean", jnp.zeros(M_s))
    raw_s = numpyro.param("od_mf_raw_std", -2.0 * jnp.ones(M_s))
    std_s = jax.nn.softplus(raw_s) + 1e-6
    numpyro.sample("v_s", dist.Normal(m_s, std_s).to_event(1))
    a = numpyro.param("od_a_mean", jnp.zeros(M_m))
    numpyro.deterministic("a_decoupled_model", a)
    if svgp.fixed_noise is None:
        noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
        numpyro.deterministic("noise", noise)


def guide_flow(svgp: SVGP, X: jnp.ndarray, y: jnp.ndarray, full_N: Optional[int]):
    """
    Normalizing Flow guide (Eq. 15):

        Base: z ~ 𝓝(0,I)
        Flow: v = T_K ∘ ... ∘ T_1 (z)
        Layers: AffineTransform → Tanh → AffineTransform (per block)

    Parameters (per layer k)
    ------------------------
    flow_loc_pre_k, flow_log_scale_pre_k
    flow_loc_post_k, flow_log_scale_post_k

    Expressivity
    ------------
    Non-Gaussian q(v); KL handled automatically by Trace_ELBO.

    Notes
    -----
    For large M, multiple tanh transforms may saturate. Consider alternative
    flows (e.g. spline) for production use.
    """
    M = svgp.inducing_inputs.shape[0]
    transforms = []
    for k in range(svgp.num_flow_layers):
        loc_pre = numpyro.param(f"flow_loc_pre_{k}", jnp.zeros(M))
        log_scale_pre = numpyro.param(f"flow_log_scale_pre_{k}", -1.0 * jnp.ones(M))
        scale_pre = jax.nn.softplus(log_scale_pre) + 1e-6
        transforms.append(T.AffineTransform(loc=loc_pre, scale=scale_pre))
        transforms.append(T.TanhTransform())
        loc_post = numpyro.param(f"flow_loc_post_{k}", jnp.zeros(M))
        log_scale_post = numpyro.param(f"flow_log_scale_post_{k}", -1.0 * jnp.ones(M))
        scale_post = jax.nn.softplus(log_scale_post) + 1e-6
        transforms.append(T.AffineTransform(loc=loc_post, scale=scale_post))
    flow = T.ComposeTransform(transforms)
    base = dist.Normal(jnp.zeros(M), jnp.ones(M)).to_event(1)
    qv = dist.TransformedDistribution(base, flow)
    numpyro.sample("v_s", qv)
    if svgp.fixed_noise is None:
        noise = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
        numpyro.deterministic("noise", noise)


# Registry for dispatch
GUIDE_REGISTRY: Dict[GuideType, Callable] = {
    "delta": guide_delta,
    "mean_field": guide_mean_field,
    "low_rank": guide_low_rank,
    "full_rank": guide_full_rank,
    "orthogonal_decoupled": guide_orthogonal_decoupled,
    "flow": guide_flow,
}


# ======================================================================================
# Prediction Helpers
# ======================================================================================

def _reconstruct_flow_transform(params: Dict, M: int, num_layers: int) -> T.ComposeTransform:
    """
    Recreate the flow transform from stored parameters.

    Parameters
    ----------
    params : dict
        Parameter dictionary (params["params"] from SVI).
    M : int
        Latent dimensionality (inducing count).
    num_layers : int
        Number of (Affine→Tanh→Affine) blocks.

    Returns
    -------
    flow : ComposeTransform
    """
    transforms = []
    for k in range(num_layers):
        loc_pre = params[f"flow_loc_pre_{k}"]
        log_scale_pre = params[f"flow_log_scale_pre_{k}"]
        scale_pre = jax.nn.softplus(log_scale_pre) + 1e-6
        transforms.append(T.AffineTransform(loc=loc_pre, scale=scale_pre))
        transforms.append(T.TanhTransform())
        loc_post = params[f"flow_loc_post_{k}"]
        log_scale_post = params[f"flow_log_scale_post_{k}"]
        scale_post = jax.nn.softplus(log_scale_post) + 1e-6
        transforms.append(T.AffineTransform(loc=loc_post, scale=scale_post))
    return T.ComposeTransform(transforms)


def predict_gaussian_family(
    params: Dict,
    svgp: SVGP,
    Xnew: jnp.ndarray,
    guide_type: GuideType,
    full_cov: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Analytic prediction for *Gaussian* guides:

        delta, mean_field, low_rank, full_rank, orthogonal_decoupled

    Steps (Prediction-Specific)
    ---------------------------
    1. Form K_ZZ, L_Z (Eq. 2).
    2. K_XZ, compute A = K_XZ K_ZZ^{-1} (Eq. 7).
    3. Extract variational parameters (m̃, covariance form).
    4. Unwhiten mean: m_u = L_Z m̃ (Eq. 10).
    5. Mean: μ = A m_u (+ decoupled mean + user mean) (Eqs. 10, 14).
    6. Diag variance: base_cond + diag(A S_u Aᵀ) (Eqs. 9, 11, 12).
    7. Full covariance (optional): K_XX - A K_ZX + A S_u Aᵀ.

    Parameters
    ----------
    params : dict
        SVI parameters container {"params": {...}}.
    svgp : SVGP
        Model configuration.
    Xnew : (N*,D)
        Test inputs.
    guide_type : GuideType
        Must be one of Gaussian families listed above.
    full_cov : bool
        If True, return full covariance matrix (not supported for flow).

    Returns
    -------
    mean : (N*,)
    var_or_cov :
        (N*,) if full_cov is False
        (N*,N*) if full_cov is True

    Complexity
    ----------
    - Base projection: O(N* M^2)
    - Full covariance (non-delta): adds O(M^2 N* + N*^2 M) depending on multiplication order.

    Notes
    -----
    For large N* or M, requesting full covariance is expensive.
    """
    Zs = svgp.inducing_inputs
    kernel = svgp.kernel
    P = params["params"]
    M = Zs.shape[0]

    # (1) + (2)
    K_ZZ, L_Z = compute_cholesky(kernel, Zs, svgp.jitter)  # (M,M)
    K_XZ = kernel(Xnew, Zs)                                # (N*,M)
    A = project_A(K_XZ, L_Z)                               # (N*,M)

    # (3) Extract variational parameters & compute diag contributions
    if guide_type == "delta":
        m_tilde = P["delta_mean"]
        m_u = unwhiten_mean(m_tilde, L_Z)
        S_diag_contrib = 0.0
        S_u = jnp.zeros((M, M))
    elif guide_type == "mean_field":
        m_tilde = P["mf_mean"]
        diag_std = jax.nn.softplus(P["mf_raw_std"]) + 1e-6
        m_u = unwhiten_mean(m_tilde, L_Z)
        S_diag_contrib = diag_variance_contrib_mean_field(A, L_Z, diag_std)
        S_u = full_S_u_mean_field(diag_std, L_Z) if full_cov else None
    elif guide_type == "low_rank":
        m_tilde = P["lr_mean"]
        Lr = P["lr_Lr"]
        diag_std = jax.nn.softplus(P["lr_raw_diag_std"]) + 1e-6
        m_u = unwhiten_mean(m_tilde, L_Z)
        S_diag_contrib = diag_variance_contrib_low_rank(A, L_Z, Lr, diag_std)
        S_u = full_S_u_low_rank(Lr, diag_std, L_Z) if full_cov else None
    elif guide_type == "full_rank":
        m_tilde = P["fr_mean"]
        L_tilde = P["fr_chol"]                       # (M,M)
        m_u = unwhiten_mean(m_tilde, L_Z)
        S_diag_contrib = diag_variance_contrib_full_rank(A, L_Z, L_tilde)
        S_u = full_S_u_full_rank(L_tilde, L_Z) if full_cov else None
    elif guide_type == "orthogonal_decoupled":
        m_tilde = P["od_mf_mean"]
        diag_std = jax.nn.softplus(P["od_mf_raw_std"]) + 1e-6
        m_u = unwhiten_mean(m_tilde, L_Z)
        S_diag_contrib = diag_variance_contrib_mean_field(A, L_Z, diag_std)
        S_u = full_S_u_mean_field(diag_std, L_Z) if full_cov else None
    else:
        raise ValueError(f"Guide {guide_type} not handled in gaussian predictor.")

    # (4) Decoupled mean part (Eq. 14)
    if guide_type == "orthogonal_decoupled":
        Zm = svgp.mean_inducing_inputs if svgp.mean_inducing_inputs is not None else Zs
        K_ZmZm, L_Zm = compute_cholesky(kernel, Zm, svgp.jitter)
        K_XZm = kernel(Xnew, Zm)                     # (N*,M_m)
        A_m = project_A(K_XZm, L_Zm)                 # (N*,M_m)
        a = P["od_a_mean"]                           # (M_m,)
        LZma = einsum(L_Zm, a, "m k, k -> m")        # (M_m,)
        decoupled_mean = einsum(A_m, LZma, "n m, m -> n")
    else:
        decoupled_mean = 0.0

    # (5) Add user mean if available
    user_mean = svgp.mean_function(Xnew) if svgp.mean_function is not None else 0.0
    mean = einsum(A, m_u, "n m, m -> n") + decoupled_mean + user_mean  # (N*,)

    # (6) Diagonal variance
    base_cond = conditional_base_variance(kernel, Xnew, K_XZ, A)       # (N*,)
    var = jnp.clip(base_cond + S_diag_contrib, 1e-12)

    if not full_cov:
        return mean, var

    # (7) Full covariance assembly
    K_XX = kernel(Xnew)                      # (N*,N*)
    AKZ = einsum(A, K_XZ, "n m, p m -> n p") # (N*,N*)
    base_full = K_XX - AKZ                   # (N*,N*)
    if guide_type == "delta":
        cov = base_full
    else:
        cov = base_full + einsum(A, S_u, A, "n m, m k, p k -> n p")
    return mean, cov


def predict_flow_family(
    params: Dict,
    svgp: SVGP,
    Xnew: jnp.ndarray,
    n_samples: int = 16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Monte Carlo prediction for flow-based (non-Gaussian) guide.

    Method
    ------
    Sample v^(j) ~ q(v) (Eq. 15), map to u^(j) = L_Z v^(j) (Eq. 4),
    compute mean_j = A u^(j), then aggregate:

        mean ≈ (1/S) Σ_j mean_j
        var_diag ≈ base_cond + (1/S) Σ_j (mean_j - mean)^2

    Parameters
    ----------
    params : dict
    svgp : SVGP
    Xnew : (N*,D)
    n_samples : int (S)

    Returns
    -------
    mean : (N*,)
    var  : (N*,)  (diagonal only)

    Notes
    -----
    Full predictive covariance would require O(S N*^2) accumulation — omitted for practicality.
    """
    Zs = svgp.inducing_inputs
    kernel = svgp.kernel
    P = params["params"]
    M = Zs.shape[0]

    K_ZZ, L_Z = compute_cholesky(kernel, Zs, svgp.jitter)  # (M,M)
    K_XZ = kernel(Xnew, Zs)                                # (N*,M)
    A = project_A(K_XZ, L_Z)                               # (N*,M)

    flow = _reconstruct_flow_transform(P, M, svgp.num_flow_layers)
    base = dist.Normal(jnp.zeros(M), jnp.ones(M)).to_event(1)
    qv = dist.TransformedDistribution(base, flow)

    def sample_mean(key):
        v = qv.sample(key)                                 # (M,)
        u = einsum(L_Z, v, "m k, k -> m")                  # (M,)
        return einsum(A, u, "n m, m -> n")                 # (N*,)

    keys = random.split(random.PRNGKey(0), n_samples)      # (S,)
    mean_samples = jax.vmap(sample_mean)(keys)             # (S,N*)
    user_mean = svgp.mean_function(Xnew) if svgp.mean_function is not None else 0.0
    mean_samples = mean_samples + user_mean                # broadcast add

    mean = jnp.mean(mean_samples, axis=0)                  # (N*,)

    base_cond = conditional_base_variance(kernel, Xnew, K_XZ, A)   # (N*,)
    mc_var = jnp.mean((mean_samples - mean[None, :])**2, axis=0)   # (N*,)
    var = jnp.clip(base_cond + mc_var, 1e-12)
    return mean, var


def predict_dispatch(
    params: Dict,
    svgp: SVGP,
    Xnew: jnp.ndarray,
    guide_type: GuideType,
    full_cov: bool = False,
    n_samples: int = 16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified prediction dispatcher handling all guide families.

    Parameters
    ----------
    params : dict
        Output of SVI.get_params() wrapped as {"params": ...}.
    svgp : SVGP
        Configured model object.
    Xnew : (N*,D)
        Test inputs.
    guide_type : GuideType
        Which variational family to use.
    full_cov : bool
        If True (and Gaussian family), returns (N*,N*) covariance.
    n_samples : int
        MC samples for flow guide only.

    Returns
    -------
    mean : (N*,)
    variance_or_cov : (N*,) or (N*,N*)

    Constraints
    -----------
    full_cov not supported for flow (non-Gaussian MC diag only).
    """
    if guide_type in {
        "delta",
        "mean_field",
        "low_rank",
        "full_rank",
        "orthogonal_decoupled",
    }:
        return predict_gaussian_family(params, svgp, Xnew, guide_type, full_cov=full_cov)
    elif guide_type == "flow":
        if full_cov:
            raise ValueError("Full covariance not supported for flow guide (Monte Carlo diag only).")
        return predict_flow_family(params, svgp, Xnew, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown guide_type {guide_type}")


# ======================================================================================
# Training Utility
# ======================================================================================

def train_svgp(
    svgp: SVGP,
    X: jnp.ndarray,
    y: jnp.ndarray,
    num_steps: int = 2000,
    learning_rate: float = 1e-2,
    batch_size: Optional[int] = None,
    seed: int = 0,
    progress: bool = True,
) -> Tuple[Dict, list]:
    """
    Optimize SVGP variational parameters via SVI.

    Parameters
    ----------
    svgp : SVGP
        Model (with chosen guide_type).
    X : (N,D)
        Full training inputs.
    y : (N,)
        Training targets.
    num_steps : int
        Number of SVI updates.
    learning_rate : float
        Adam optimizer LR.
    batch_size : int | None
        If None, full-batch optimization; else mini-batching with scaling (Eq. 6).
    seed : int
        RNG seed for reproducibility.
    progress : bool
        Print periodic loss if True.

    Returns
    -------
    params : dict
        Learned parameter dictionary (pass into predict_dispatch).
    losses : list[float]
        Sequence of losses (negative ELBO) per step.

    Workflow
    --------
    For t in 0..num_steps-1:
      1. Sample batch (if needed)
      2. state, loss = svi.update(state, X_batch, y_batch)
      3. Record loss
    """
    if svgp.guide_type not in GUIDE_REGISTRY:
        raise ValueError(f"Guide {svgp.guide_type} not registered")

    guide_fn = GUIDE_REGISTRY[svgp.guide_type]
    N = X.shape[0]
    if batch_size is None:
        batch_size = N
    full_N = N if batch_size < N else None

    # Initialize with first batch.
    init_X = X[:batch_size]
    init_y = y[:batch_size]
    svi, state = svgp.make_svi(init_X, init_y, guide_fn, learning_rate=learning_rate, seed=seed, full_N=full_N)

    rng = random.PRNGKey(seed)
    losses = []
    for t in range(num_steps):
        if batch_size < N:
            rng, sk = random.split(rng)
            idx = random.choice(sk, N, shape=(batch_size,), replace=False)  # (B,)
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y

        state, loss = svi.update(state, Xb, yb)
        losses.append(loss)

        if progress and (t % max(1, num_steps // 10) == 0 or t == num_steps - 1):
            print(f"[{t:05d}] loss={loss:.6f}")

    params = svi.get_params(state)
    return params, losses


# ======================================================================================
# Example / Smoke Test
# ======================================================================================

def _example():
    """
    Smoke test across all guides to verify interfaces.

    Procedure
    ---------
    1. Generate synthetic regression data: y = 1.5 sin(x) + ε.
    2. Train separate SVGP models with each guide type for a modest number of steps.
    3. Predict mean/variance on dense test grid (diag variance only).
    4. Print final loss and output shapes for sanity.

    Note
    ----
    This is not a convergence or performance benchmark. Full covariance
    predictions are not requested here for speed.
    """
    key = random.PRNGKey(0)
    N = 300
    X = random.uniform(key, (N, 1), minval=-4.0, maxval=4.0)
    f_true = jnp.sin(X[:, 0]) * 1.5
    y = f_true + 0.25 * random.normal(key, (N,))

    M = 40
    Z = jnp.linspace(-4, 4, M).reshape(M, 1)
    Zm = jnp.linspace(-4, 4, M // 2).reshape(M // 2, 1)

    guide_list = [
        "delta",
        "mean_field",
        "low_rank",
        "full_rank",
        "orthogonal_decoupled",
        "flow",
    ]

    for g in guide_list:
        print(f"\n=== Training guide: {g} ===")
        svgp = SVGP(
            kernel=RBF(variance=1.0, lengthscale=1.0),
            inducing_inputs=Z,
            fixed_noise=0.25,
            guide_type=g,
            low_rank=10,
            mean_inducing_inputs=Zm if g == "orthogonal_decoupled" else None,
            num_flow_layers=2,
        )
        params, losses = train_svgp(
            svgp,
            X,
            y,
            num_steps=300,
            learning_rate=3e-3,
            batch_size=64,
            progress=False,
        )
        Xtest = jnp.linspace(-4, 4, 200).reshape(-1, 1)
        mean, var = predict_dispatch(
            {"params": params},
            svgp,
            Xtest,
            guide_type=g,
            full_cov=False,
            n_samples=16,
        )
        print(f"Guide={g:22s}  mean.shape={mean.shape}  var.shape={var.shape}  final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    _example()