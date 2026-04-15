"""nal (Full) Gaussian Process (VGP) with Richly Annotated Multi‑Guide Inference
====================================================================================

This module implements a *full* Variational Gaussian Process (VGP) following the
same style and depth of annotation as the (SVGP) sparse module: extensive
equation tracking, array shape tracing, and modular helper utilities.

Unlike SVGP (which introduces M << N inducing variables), here we place a
variational distribution directly over the latent function values at ALL
training inputs. This gives an exact prior but (typically) O(N^3) training
complexity due to the Cholesky of the full kernel matrix.

Notation Summary
----------------
Training inputs:         X = { x_i }_{i=1..N},      X ∈ ℝ^{N×D}
Latent function vector:  f = [ f(x_1), ..., f(x_N) ]ᵀ ∈ ℝ^{N}
Observations (regression):
    (Eq. 1)   y_i | f_i  ~  𝓝(f_i, σ_n²)

Zero-mean GP prior (can add a separate mean function m(x)):
    (Eq. 2)   f ~ 𝓝(0, K),   K = K(X,X) ∈ ℝ^{N×N}

Whitening
---------
Add jitter ε > 0 for stability:
    (Eq. 3)   K̃ = K + ε I
Compute Cholesky:
    (Eq. 4)   K̃ = L Lᵀ,  L lower-triangular (N×N)
Define whitened latent variable v:
    (Eq. 5)   f = L v,   v ~ 𝓝(0, I_N)

Variational Families (over v)
-----------------------------
We approximate p(v|y) with q(v). Supported guide families:

(Guide 1) Delta / MAP
    (Eq. 6)     q(v) = δ(v - m̃)

(Guide 2) Mean-field diagonal Gaussian
    (Eq. 7)     q(v) = 𝓝(m̃, diag(σ²))

(Guide 3) Low-rank + diagonal
    (Eq. 8)     q(v) = 𝓝(m̃, L_r L_rᵀ + diag(σ²)),  L_r ∈ ℝ^{N×r}

(Guide 4) Full-rank Cholesky
    (Eq. 9)     q(v) = 𝓝(m̃, S̃),  S̃ = L̃ L̃ᵀ,  L̃ lower-triangular (N×N)

(Guide 5) Normalizing Flow (non-Gaussian)
    (Eq.10)     v = T_K∘...∘T_1 (z),   z ~ 𝓝(0, I_N),
                each T_k = Affine ∘ Tanh ∘ Affine (here for illustration)

Observation Likelihood (Gaussian)
---------------------------------
    (Eq.11)   y | f  ~  𝓝(f, σ_n² I)

Whitened variable makes the prior p(v) = 𝓝(0, I), simplifying KL terms.

Moments under q(v)
------------------
For any Gaussian guide (Eq. 6–9):
    (Eq.12)   μ_v = m̃
    (Eq.13)   Σ_v = S̃   (structure depends on family)
Map to function space:
    (Eq.14)   μ_f = L μ_v
    (Eq.15)   Σ_f = L Σ_v Lᵀ
We use Σ_f diag for the expected log likelihood.

Likelihood Expectation
----------------------
From (Eq.11):
    log p(y|f) = -½ Σ_i [ (y_i - f_i)² / σ_n² + log(2πσ_n²) ]
Taking expectation under q(f):
    E_q[(y_i - f_i)²] = (y_i - μ_fi)² + Var_q[f_i]
Thus:
    (Eq.16)   E_q[ log p(y|f) ] =
        -½ Σ_i [ ( (y_i - μ_fi)² + Var_q[f_i] ) / σ_n² + log(2πσ_n²) ]

ELBO
----
    (Eq.17)   ELBO = E_q[ log p(y|f) ] - KL[ q(v) || p(v) ]

KL Terms
--------
Prior p(v) = 𝓝(0, I).
For Gaussian q(v)=𝓝(m̃,S̃):
    (Eq.18)   KL = ½( tr(S̃) + m̃ᵀ m̃ - N - log det S̃ )

Specializations:
- Mean-field: S̃ = diag(σ²) → tr(S̃)=Σσ_i², log det S̃ = Σ log σ_i²
- Low-rank + diag:
    (Eq.19) log det S̃ = log det(D) + log det( I_r + L_rᵀ D^{-1} L_r )
           tr(S̃) = tr(D) + tr(L_rᵀ L_r)
  where D = diag(σ²).
- Full-rank: S̃ = L̃L̃ᵀ → log det S̃ = 2 Σ log diag(L̃)
- Delta: limit S̃→0 ⇒ KL → ½ (m̃ᵀ m̃ - N) ignoring divergent constant
- Flow: computed implicitly by NumPyro's Trace_ELBO via reparameterization.

Prediction
----------
At training inputs (this module only; extend for new X if desired):
    (Eq.14–15) for Gaussian families, or MC for flow.
Return either:
  - Mean only
  - Mean + diag variance
  - Mean + full covariance (Gaussian guides)

Shape Legend
------------
N : number of training points
D : feature dimension
r : low-rank rank (≤ N)
S : number of MC samples for flow predictions
All arrays are annotated inline in functions.

Implementation Overview
-----------------------
Main class: VGP
  - model()  builds prior + likelihood with whitened v
  - guide()  provides q(v)
  - predict() obtains posterior mean/variance at training points
  - train()   SVI loop
Helper functions provide shape‑explicit building blocks referencing equations.

Design Choices
--------------
- We reuse whitened variable v for numerical stability.
- Flow guide uses affine+tanh composition; extend with richer transforms if needed.
- Mini-batching is supported, but each batch re-Cholesky's its submatrix; for true
  stochastic approximate VGP you would employ alternative approximations.

All docstrings and inline comments explicitly cite equations (Eq.*) and shapes.
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
from einops import einsum  # (alias of jnp.einsum with same semantics)


# =============================================================================
# Kernel Base & Example Implementation
# =============================================================================

class Kernel:
    """
    Abstract positive definite kernel interface.

    Method
    ------
    __call__(X, X2=None) -> (N,M) covariance matrix

    Parameters
    ----------
    X  : (N,D)
    X2 : (M,D) | None  (defaults to X)

    Returns
    -------
    K  : (N,M)
    """
    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> jnp.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RBF(Kernel):
    """
    Squared Exponential / RBF kernel.

    (Eq. K1) k(x,x') = σ² exp( -½ Σ_d ( (x_d - x'_d)^2 / ℓ_d^2 ) )

    Parameters
    ----------
    variance : float
        σ² amplitude > 0 (not constrained here, rely on user).
    lengthscale : float | jnp.ndarray
        ℓ lengthscale(s), can be scalar or shape (D,).

    Shapes
    ------
    X : (N,D)
    X2: (M,D)
    K : (N,M)
    """
    variance: float = 1.0
    lengthscale: float | jnp.ndarray = 1.0

    def __call__(self, X: jnp.ndarray, X2: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        X2 = X if X2 is None else X2
        ℓ = jnp.asarray(self.lengthscale)
        Xs  = X  / ℓ  # (N,D)
        X2s = X2 / ℓ  # (M,D)
        x_norm  = jnp.sum(Xs**2,  axis=-1, keepdims=True)      # (N,1)
        x2_norm = jnp.sum(X2s**2, axis=-1, keepdims=True).T    # (1,M)
        cross   = Xs @ X2s.T                                   # (N,M)
        sqd = x_norm - 2 * cross + x2_norm                     # (N,M)
        return self.variance * jnp.exp(-0.5 * sqd)


# =============================================================================
# Guide Type Enumeration
# =============================================================================

GuideType = Literal["delta", "mean_field", "low_rank", "full_rank", "flow"]


# =============================================================================
# Helper Functions (All Fully Documented with Equations / Shapes)
# =============================================================================

def compute_kernel_cholesky(kernel: Kernel, X: jnp.ndarray, jitter: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute stabilized Gram matrix and its Cholesky.

    Steps
    -----
    (Eq. 2)  K = k(X,X)
    (Eq. 3)  K̃ = K + ε I
    (Eq. 4)  Cholesky: K̃ = L Lᵀ

    Parameters
    ----------
    kernel : Kernel
    X : (N,D)
    jitter : float
        ε > 0 stability term

    Returns
    -------
    K : (N,N)
    L : (N,N)  lower-triangular

    Complexity
    ----------
    O(N^2 D + N^3)

    Shapes
    ------
    K, L both (N,N)
    """
    K = kernel(X)
    K_tilde = K + jitter * jnp.eye(X.shape[0])
    L = jnp.linalg.cholesky(K_tilde)
    return K, L


def whiten_forward(L: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Map whitened latent v to function values f.

    Equation
    --------
    (Eq. 5 & 14)  f = L v

    Parameters
    ----------
    L : (N,N)
    v : (N,)

    Returns
    -------
    f : (N,)
    """
    return einsum(L, v, "n k, k -> n")


def diag_cov_mean_field(L: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal of function-space covariance for mean-field guide.

    Given S̃ = diag(σ²), Σ_f = L S̃ Lᵀ = (L diag(σ)) (L diag(σ))ᵀ.

    The diagonal is row-wise squared norms:

        Var_q[f_i] = Σ_k ( L[i,k] * σ_k )².

    Parameters
    ----------
    L : (N,N)
    std : (N,)  σ (positive)

    Returns
    -------
    var_f_diag : (N,)

    Equations
    ---------
    (Eq.15) with S̃ diagonal
    """
    L_scaled = L * std  # (N,N)
    return jnp.sum(L_scaled**2, axis=-1)  # (N,)


def diag_cov_low_rank(L: jnp.ndarray, Lr: jnp.ndarray, diag_std: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal of Σ_f for low-rank + diagonal S̃ = L_r L_rᵀ + diag(σ²).

        Σ_f = L S̃ Lᵀ
             = (L L_r)(L L_r)ᵀ  +  (L diag(σ))(L diag(σ))ᵀ

    Diagonal:
        Var_q[f_i] = || (L L_r)[i,:] ||² + || (L[i,:] * σ) ||²

    Parameters
    ----------
    L : (N,N)
    Lr : (N,r)
    diag_std : (N,)  σ

    Returns
    -------
    var_f_diag : (N,)

    Equations
    ---------
    Combine (Eq. 8) inside (Eq.15).
    """
    LLr   = L @ Lr         # (N,r)
    Ldiag = L * diag_std   # (N,N)
    return jnp.sum(LLr**2, axis=-1) + jnp.sum(Ldiag**2, axis=-1)


def full_cov_low_rank(L: jnp.ndarray, Lr: jnp.ndarray, diag_std: jnp.ndarray) -> jnp.ndarray:
    """
    Full Σ_f for low-rank + diagonal:

        Σ_f = (L L_r)(L L_r)ᵀ + (L diag(σ))(L diag(σ))ᵀ.

    Parameters
    ----------
    L : (N,N)
    Lr : (N,r)
    diag_std : (N,)

    Returns
    -------
    Σ_f : (N,N)
    """
    LLr   = L @ Lr
    Ldiag = L * diag_std
    return (LLr @ LLr.T) + (Ldiag @ Ldiag.T)


def diag_cov_full_rank(L: jnp.ndarray, L_tilde: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonal of full-rank Σ_f where S̃ = L̃ L̃ᵀ:

        Σ_f = L S̃ Lᵀ = (L L̃)(L L̃)ᵀ

    Var_q[f_i] = || (L L̃)[i,:] ||².

    Parameters
    ----------
    L : (N,N)
    L_tilde : (N,N)

    Returns
    -------
    var_f_diag : (N,)
    """
    LLt = L @ L_tilde  # (N,N)
    return jnp.sum(LLt**2, axis=-1)


def full_cov_full_rank(L: jnp.ndarray, L_tilde: jnp.ndarray) -> jnp.ndarray:
    """
    Full Σ_f for full-rank Cholesky variational covariance:

        Σ_f = (L L̃)(L L̃)ᵀ

    Parameters
    ----------
    L : (N,N)
    L_tilde : (N,N)

    Returns
    -------
    Σ_f : (N,N)
    """
    LLt = L @ L_tilde
    return LLt @ LLt.T


# =============================================================================
# VGP Class
# =============================================================================

@dataclass
class VGP:
    """
    Full Variational Gaussian Process with Multi-Guide Variational Families.

    Parameters
    ----------
    kernel : Kernel
        Prior covariance function k(·,·)
    noise_prior : Distribution | None
        Prior over observation noise std σ_n (used if fixed_noise is None).
    mean_function : callable | None
        Deterministic mean m(x): ℝ^{N×D} → ℝ^{N}
    jitter : float
        Jitter ε added to K diagonal (Eq.3).
    fixed_noise : float | None
        If not None, treat noise std as known constant.
    guide_type : GuideType
        Variational family: "delta", "mean_field", "low_rank", "full_rank", "flow".
    low_rank : int
        Rank r for low-rank family (Eq. 8) (clipped to ≤ N).
    num_flow_layers : int
        Number of (Affine→Tanh→Affine) blocks in flow transform (Eq.10).
    mc_flow_samples : int
        MC samples for training/prediction with flow guide.

    Cached (after first model call) — these are convenience caches; *not*
    strictly required for correctness:
        _train_X : (N,D) training inputs
        _K       : (N,N)
        _L       : (N,N) Cholesky of stabilized kernel
        _N       : int

    Notes
    -----
    The whitened prior ensures stable inference for large variations in current
    kernel hyperparameters. Flow guide can capture multi-modal or skewed posteriors
    but at higher computational cost (MC variance).
    """
    kernel: Kernel
    noise_prior: Optional[dist.Distribution] = None
    mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    jitter: float = 1e-6
    fixed_noise: Optional[float] = None
    guide_type: GuideType = "mean_field"
    low_rank: int = 8
    num_flow_layers: int = 2
    mc_flow_samples: int = 16

    # Caches (populated on model call)
    _train_X: Optional[jnp.ndarray] = None
    _K: Optional[jnp.ndarray] = None
    _L: Optional[jnp.ndarray] = None
    _N: Optional[int] = None

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    def model(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        NumPyro model defining prior over whitened v and likelihood for y.

        Steps (matching intro equations)
        ---------------------------------
        (1) Compute Gram matrix:        K = k(X,X)                 (Eq.2)
        (2) Stabilize:                   K̃ = K + ε I               (Eq.3)
        (3) Cholesky:                    K̃ = L Lᵀ                 (Eq.4)
        (4) Sample whitened latent:     v ~ 𝓝(0, I_N)             (Eq.5 prior form)
        (5) Transform to f:             f = L v                   (Eq.5,14)
        (6) Add deterministic mean:     if m(x) provided
        (7) Noise parameter:
              - fixed_noise ? constant : sample from prior
        (8) Likelihood: y | f ~ 𝓝(f, σ_n² I)                     (Eq.11)

        Shapes
        ------
        X : (N,D)
        K, L : (N,N)
        v, f, y : (N,)

        Notes
        -----
        Stores (X, K, L, N) for subsequent prediction.
        """
        N = X.shape[0]

        # (1)–(3)
        K, L = compute_kernel_cholesky(self.kernel, X, self.jitter)

        # Cache (mutating dataclass instance – acceptable for runtime convenience)
        self._train_X = X
        self._K = K
        self._L = L
        self._N = N

        # (4) whitened latent
        v = numpyro.sample("v", dist.MultivariateNormal(jnp.zeros(N), jnp.eye(N)))

        # (5) map to function values
        f = whiten_forward(L, v)

        # (6) add mean function if present
        if self.mean_function is not None:
            f = f + self.mean_function(X)

        # (7) noise handling
        if self.fixed_noise is not None:
            noise = self.fixed_noise
        else:
            noise = numpyro.sample(
                "noise",
                self.noise_prior if self.noise_prior is not None else dist.Exponential(1.0),
            )

        # (8) Likelihood
        numpyro.sample("y", dist.Normal(f, noise), obs=y)

    # -------------------------------------------------------------------------
    # GUIDE
    # -------------------------------------------------------------------------
    def guide(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Variational guide q(v) over whitened latent v.

        Families Implemented
        --------------------
        "delta"      (Eq. 6)  q(v)=δ(v-m̃)
        "mean_field" (Eq. 7)  q(v)=𝓝(m̃, diag(σ²))
        "low_rank"   (Eq. 8)  q(v)=𝓝(m̃, L_r L_rᵀ + diag(σ²))
        "full_rank"  (Eq. 9)  q(v)=𝓝(m̃, L̃L̃ᵀ)
        "flow"       (Eq.10)  q(v)=T(z) with z ~ 𝓝(0,I)

        Noise Parameter
        ---------------
        For non-fixed noise we include a point-estimate parameter noise_q
        (variational delta on σ_n).

        Shapes
        ------
        N : number of data points
        m̃ : (N,)
        std / diag_std : (N,)
        L_r : (N,r)
        L̃  : (N,N) lower-triangular
        v   : (N,)
        """
        N = X.shape[0]
        gt = self.guide_type

        # Convenience closure for noise param (point estimate)
        def noise_param():
            if self.fixed_noise is None:
                noise_q = numpyro.param("noise_q", jnp.array(0.1), constraint=dist.constraints.positive)
                numpyro.deterministic("noise", noise_q)

        if gt == "delta":
            m = numpyro.param("delta_mean", jnp.zeros(N))
            numpyro.sample("v", dist.Delta(m, event_dim=1))
            noise_param()

        elif gt == "mean_field":
            m = numpyro.param("mf_mean", jnp.zeros(N))
            raw = numpyro.param("mf_raw_std", -2.0 * jnp.ones(N))
            std = jax.nn.softplus(raw) + 1e-6
            numpyro.sample("v", dist.Normal(m, std).to_event(1))
            noise_param()

        elif gt == "low_rank":
            r = min(self.low_rank, N)
            m = numpyro.param("lr_mean", jnp.zeros(N))
            Lr = numpyro.param("lr_Lr", 0.01 * jnp.ones((N, r)))
            raw = numpyro.param("lr_raw_diag_std", -2.0 * jnp.ones(N))
            diag_std = jax.nn.softplus(raw) + 1e-6
            # Reparameterized sample:
            eps_r = numpyro.sample("eps_r", dist.Normal(0, 1).expand([r]).to_event(1))  # (r,)
            eps_d = numpyro.sample("eps_d", dist.Normal(0, 1).expand([N]).to_event(1))  # (N,)
            low_rank_part = einsum(Lr, eps_r, "n r, r -> n")
            v = m + low_rank_part + diag_std * eps_d
            numpyro.deterministic("v_det", v)
            numpyro.sample("v", dist.Delta(v, event_dim=1))
            noise_param()

        elif gt == "full_rank":
            m = numpyro.param("fr_mean", jnp.zeros(N))
            L_tilde = numpyro.param(
                "fr_chol", 0.01 * jnp.eye(N), constraint=dist.constraints.lower_cholesky
            )
            numpyro.sample("v", dist.MultivariateNormal(m, scale_tril=L_tilde))
            noise_param()

        elif gt == "flow":
            transforms = []
            for k in range(self.num_flow_layers):
                loc_pre = numpyro.param(f"flow_loc_pre_{k}", jnp.zeros(N))
                log_scale_pre = numpyro.param(f"flow_log_scale_pre_{k}", -1.0 * jnp.ones(N))
                scale_pre = jax.nn.softplus(log_scale_pre) + 1e-6
                transforms.append(T.AffineTransform(loc=loc_pre, scale=scale_pre))
                transforms.append(T.TanhTransform())
                loc_post = numpyro.param(f"flow_loc_post_{k}", jnp.zeros(N))
                log_scale_post = numpyro.param(f"flow_log_scale_post_{k}", -1.0 * jnp.ones(N))
                scale_post = jax.nn.softplus(log_scale_post) + 1e-6
                transforms.append(T.AffineTransform(loc=loc_post, scale=scale_post))
            flow = T.ComposeTransform(transforms)
            base = dist.Normal(jnp.zeros(N), jnp.ones(N)).to_event(1)
            qv = dist.TransformedDistribution(base, flow)
            numpyro.sample("v", qv)
            noise_param()

        else:
            raise ValueError(f"Unknown guide_type: {gt}")

    # -------------------------------------------------------------------------
    # BUILD SVI
    # -------------------------------------------------------------------------
    def make_svi(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> Tuple[SVI, "SVIState"]:
        """
        Construct SVI object with Trace_ELBO.

        Parameters
        ----------
        X : (N,D)
        y : (N,)
        learning_rate : float
        seed : int

        Returns
        -------
        svi : SVI
        state : SVIState

        Notes
        -----
        - model() + guide() closure ensures up-to-date hyperparameters.
        - For flow guide, ELBO uses MC KL.
        """
        model = lambda X_, y_: self.model(X_, y_)
        guide = lambda X_, y_: self.guide(X_, y_)
        svi = SVI(model, guide, Adam(learning_rate), loss=Trace_ELBO())
        rng = random.PRNGKey(seed)
        state = svi.init(rng, X, y)
        return svi, state

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    def predict(
        self,
        params: Dict,
        X: Optional[jnp.ndarray] = None,
        full_cov: bool = False,
        return_var: bool = True,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Posterior predictive over latent f at training inputs (no new X variant).

        Inputs
        ------
        params : dict
            SVI parameter dictionary (svi.get_params(...) output).
        X : (N,D) | None
            Use if different from originally cached X, else uses self._train_X.
        full_cov : bool
            If True and guide is Gaussian, return full Σ_f (Eq.15).
        return_var : bool
            If False, returns mean only.

        Returns
        -------
        mean_f : (N,)
        var_or_cov :
            None                     if return_var=False
            (N,)                     if return_var=True, full_cov=False
            (N,N)                    if return_var=True, full_cov=True (Gaussian guides)
            For flow, MC diag variance (and optionally MC full covariance if full_cov=True)

        Procedure (Gaussian Guides)
        ---------------------------
        1. Get L (Eq.4) — from cache or recompute.
        2. Extract q(v) parameters → m̃ (and covariance structure).
        3. Compute μ_f = L m̃ (Eq.14).
        4. If variance requested, build diag or full Σ_f = L S̃ Lᵀ (Eq.15).

        Flow Guide
        ----------
        MC sample v^(s) ~ q(v), map f^(s)=L v^(s), average & compute empirical variance.

        Complexity
        ----------
        - Full covariance full_rank: O(N^3) (due to multiplications).
        - Diagonal statistics: O(N^2) typical due to L times structures.
        """
        assert (X is not None) or (self._train_X is not None), "Need training X or supply X."
        X_use = self._train_X if X is None else X
        N = X_use.shape[0]

        # Recompute L if not cached or if new X provided
        if (self._L is None) or (X is not None and (self._train_X is None or X_use is not self._train_X)):
            _, L = compute_kernel_cholesky(self.kernel, X_use, self.jitter)
        else:
            L = self._L

        P = params["params"]
        gt = self.guide_type

        # ---------------- Delta Guide (Eq.6) ----------------
        if gt == "delta":
            m = P["delta_mean"]                 # (N,)
            mean = whiten_forward(L, m)         # (N,)
            if not return_var:
                return mean, None
            if full_cov:
                return mean, jnp.zeros((N, N))
            return mean, jnp.zeros(N)

        # ---------------- Mean-field (Eq.7) ----------------
        if gt == "mean_field":
            m = P["mf_mean"]                    # (N,)
            std = jax.nn.softplus(P["mf_raw_std"]) + 1e-6  # (N,)
            mean = whiten_forward(L, m)
            if not return_var:
                return mean, None
            if full_cov:
                L_scaled = L * std
                cov = L_scaled @ L_scaled.T
                return mean, cov
            var = diag_cov_mean_field(L, std)
            return mean, var

        # ---------------- Low-rank + diag (Eq.8) ----------------
        if gt == "low_rank":
            m = P["lr_mean"]                    # (N,)
            Lr = P["lr_Lr"]                     # (N,r)
            diag_std = jax.nn.softplus(P["lr_raw_diag_std"]) + 1e-6  # (N,)
            mean = whiten_forward(L, m)
            if not return_var:
                return mean, None
            if full_cov:
                cov = full_cov_low_rank(L, Lr, diag_std)
                return mean, cov
            var = diag_cov_low_rank(L, Lr, diag_std)
            return mean, var

        # ---------------- Full-rank (Eq.9) ----------------
        if gt == "full_rank":
            m = P["fr_mean"]                    # (N,)
            L_tilde = P["fr_chol"]              # (N,N)
            mean = whiten_forward(L, m)
            if not return_var:
                return mean, None
            if full_cov:
                cov = full_cov_full_rank(L, L_tilde)
                return mean, cov
            var = diag_cov_full_rank(L, L_tilde)
            return mean, var

        # ---------------- Flow (Eq.10) ----------------
        if gt == "flow":
            transforms = []
            for k in range(self.num_flow_layers):
                loc_pre = P[f"flow_loc_pre_{k}"]
                log_scale_pre = P[f"flow_log_scale_pre_{k}"]
                scale_pre = jax.nn.softplus(log_scale_pre) + 1e-6
                transforms.append(T.AffineTransform(loc=loc_pre, scale=scale_pre))
                transforms.append(T.TanhTransform())
                loc_post = P[f"flow_loc_post_{k}"]
                log_scale_post = P[f"flow_log_scale_post_{k}"]
                scale_post = jax.nn.softplus(log_scale_post) + 1e-6
                transforms.append(T.AffineTransform(loc=loc_post, scale=scale_post))
            flow = T.ComposeTransform(transforms)
            base = dist.Normal(jnp.zeros(N), jnp.ones(N)).to_event(1)
            qv = dist.TransformedDistribution(base, flow)

            rng = random.PRNGKey(0)  # For reproducibility; in production pass a key
            keys = random.split(rng, self.mc_flow_samples)

            def sample_f(k):
                v = qv.sample(k)             # (N,)
                return whiten_forward(L, v)  # (N,)

            Fs = jax.vmap(sample_f)(keys)    # (S,N)
            mean = jnp.mean(Fs, axis=0)
            if not return_var:
                return mean, None
            if full_cov:
                centered = Fs - mean[None, :]
                cov = (centered.T @ centered) / self.mc_flow_samples  # (N,N)
                return mean, cov
            var = jnp.mean((Fs - mean[None, :]) ** 2, axis=0)  # (N,)
            return mean, var

        raise ValueError(f"Unknown guide_type: {gt}")

    # -------------------------------------------------------------------------
    # TRAINING CONVENIENCE
    # -------------------------------------------------------------------------
    def train(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        num_steps: int = 2000,
        learning_rate: float = 1e-2,
        batch_size: Optional[int] = None,
        seed: int = 0,
        progress: bool = True,
    ) -> Tuple[Dict, list]:
        """
        Optimize ELBO via SVI.

        Parameters
        ----------
        X : (N,D)
        y : (N,)
        num_steps : int
        learning_rate : float
        batch_size : int | None
            If provided, subsamples are used. Note:
            - Full VGP re-Cholesky's per batch (sub-K) → inconsistent with full prior
              unless advanced control variates / corrections are used.
            - Provided for API parity with SVGP; for correctness prefer full batch.
        seed : int
        progress : bool

        Returns
        -------
        params : dict (svi.get_params)
        losses : list[float]

        Complexity
        ----------
        Full batch each step: O(N^3).
        Mini-batch variant: O(B^3) per step but changes objective (approx).

        Practical Advice
        ----------------
        For large N, prefer sparse or structured GP approximations (e.g., SVGP).
        """
        N = X.shape[0]
        if batch_size is None:
            batch_size = N

        svi, state = self.make_svi(
            X if batch_size == N else X[:batch_size],
            y if batch_size == N else y[:batch_size],
            learning_rate=learning_rate,
            seed=seed,
        )
        rng = random.PRNGKey(seed)
        losses = []
        for t in range(num_steps):
            if batch_size < N:
                rng, sk = random.split(rng)
                idx = random.choice(sk, N, shape=(batch_size,), replace=False)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            state, loss = svi.update(state, Xb, yb)
            losses.append(loss)

            if progress and (t % max(1, num_steps // 10) == 0 or t == num_steps - 1):
                print(f"[{t:05d}] loss={loss:.6f}")

        params = svi.get_params(state)
        return params, losses


# =============================================================================
# Example (Smoke Test)
# =============================================================================

def _example():
    """
    Quick usage demonstration (not a benchmark):

    1. Generate synthetic 1D data (sinusoidal latent + Gaussian noise).
    2. Fit VGP with each guide type.
    3. Predict posterior mean & diag variance.
    4. Demonstrate full covariance retrieval for mean_field guide.

    Shapes
    ------
    X : (N,1)
    f_true, y, mean : (N,)
    var / diag : (N,)
    cov : (N,N)
    """
    key = random.PRNGKey(42)
    N = 80
    X = jnp.linspace(-4.0, 4.0, N).reshape(-1, 1)
    kernel = RBF(variance=1.3, lengthscale=1.0)

    # True latent
    K_true = kernel(X)
    L_true = jnp.linalg.cholesky(K_true + 1e-6 * jnp.eye(N))
    f_true = L_true @ random.normal(key, (N,))
    noise = 0.15
    y = f_true + noise * random.normal(key, (N,))

    guide_list = ["delta", "mean_field", "low_rank", "full_rank", "flow"]

    for gt in guide_list:
        print(f"\n=== Guide: {gt} ===")
        vgp = VGP(
            kernel=kernel,
            fixed_noise=noise,
            guide_type=gt,
            low_rank=10,
            num_flow_layers=2,
            mc_flow_samples=32,
        )
        params, losses = vgp.train(
            X,
            y,
            num_steps=400 if gt != "full_rank" else 250,
            learning_rate=5e-3,
            batch_size=None,
            progress=False,
        )
        mean, var = vgp.predict({"params": params}, X, full_cov=False, return_var=True)
        print(f"mean.shape={mean.shape} var.shape={None if var is None else var.shape} final_loss={losses[-1]:.4f}")

    # Full covariance example for mean_field
    vgp = VGP(kernel=kernel, fixed_noise=noise, guide_type="mean_field")
    params, _ = vgp.train(X, y, num_steps=200, learning_rate=5e-3, progress=False)
    _, cov = vgp.predict({"params": params}, X, full_cov=True, return_var=True)
    print("Full covariance (mean_field) shape:", cov.shape)


if __name__ == "__main__":
    _example()