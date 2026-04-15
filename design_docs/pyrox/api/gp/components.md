---
status: draft
version: 0.1.0
---

# Layer 1 — Components

Protocols and implementations that compose Layer 0 primitives. The four protocols form an orthogonal stack — any valid (Kernel x Solver x InferenceStrategy x Integrator) combination works.

For full subsystem-level detail, see [gp_moments.md](gp_moments.md), [gp_state_space.md](gp_state_space.md), and [gp_integration.md](gp_integration.md).

---

## Protocol Summary

| Protocol | Role | Key Method | Detail |
|---|---|---|---|
| `Kernel` | What covariance structure | `__call__(X1, X2) -> Array` | [gp_moments.md §Kernel](gp_moments.md) |
| `Solver` | How to do linear algebra | `solve(K, y)`, `logdet(K)` | [gp_moments.md §Solver](gp_moments.md) |
| `InferenceStrategy` | What expectations define sites | `compute_sites(f, y, lik) -> (nat1, nat2)` | [gp_state_space.md §InferenceStrategy](gp_state_space.md) |
| `Integrator` | How to compute $\mathbb{E}_q[g(f)]$ | `integrate(fn, mean, var) -> Array` | [gp_integration.md §Integrator](gp_integration.md) |
| `Guide` | Variational posterior structure | `sample(key) -> f`, `log_prob(f) -> Scalar` | [gp_moments.md §Guide](gp_moments.md) |
| `InducingFeatures` | Inter-domain inducing variables | `K_uu(kernel)`, `k_u(X, kernel)` | [features/gp_inducing_features.md](../../features/gp/inducing_features.md) |
| `MultiOutputKernel` | Multi-output kernel (latent + mixing) | `Kgg(X1, X2)`, `mixing_matrix()` | [features/gp_multi_output.md](../../features/gp/multi_output.md) |

---

## Kernel Protocol

```python
class Kernel(eqx.Module):
    @abc.abstractmethod
    def __call__(self, X1, X2) -> Array: ...
    def to_representation(self) -> CovarianceRepresentation: ...
```

**Implementations:** `RBF`, `Matern`, `Periodic`, `Linear`, `DeepKernel`, `SumKernel`, `ProductKernel`.

**Covariance representations** (output of `to_representation()`):

| Representation | Structure | Used by |
|---|---|---|
| `DenseCov` | Full $N \times N$ matrix | `CholeskySolver` |
| `LowRankPlusDiag` | $WW^T + \operatorname{diag}(d)$ | `WoodburySolver` |
| `KroneckerCov` | $K_1 \otimes K_2$ | `KroneckerSolver` |
| `StateSpaceRep` | LTI SDE $(A, Q, H, P_\infty)$ | `KalmanSolver` |

---

## Solver Protocol

```python
class Solver(eqx.Module):
    @abc.abstractmethod
    def solve(self, representation, y) -> Array: ...
    @abc.abstractmethod
    def logdet(self, representation) -> Scalar: ...
    def log_marginal(self, representation, y) -> Scalar: ...
```

**Implementations:**

| Solver | Representation | Complexity | Detail |
|---|---|---|---|
| `CholeskySolver` | Dense | $O(N^3)$ | [gp_moments.md](gp_moments.md) |
| `CGSolver` | Dense/matrix-free | $O(N^2 k)$ | via gaussx |
| `BBMMSolver` | Dense/matrix-free | $O(N^2 k)$ amortized | via gaussx |
| `WoodburySolver` | LowRankPlusDiag | $O(NM^2)$ | via gaussx |
| `KroneckerSolver` | Kronecker | $O(N_1^3 + N_2^3)$ | via gaussx |
| `KalmanSolver` | StateSpace | $O(NS^3)$ | [gp_state_space.md](gp_state_space.md) |

---

## InferenceStrategy Protocol

```python
class InferenceStrategy(eqx.Module):
    @abc.abstractmethod
    def compute_sites(self, f_mean, f_var, y, likelihood) -> tuple[Array, Array]: ...
```

The Bayes-Newton unification: VI, EP, Laplace, and PL are all instances of Newton's method on different objectives, differing only in how they compute the natural-parameter sites $(\lambda_1, \lambda_2)$.

**Implementations:**

| Strategy | Method | Convergence | PSD guaranteed? | Detail |
|---|---|---|---|---|
| `NewtonLaplace` | $\nabla^2 \log p$ | Quadratic | No | [gp_state_space.md](gp_state_space.md) |
| `VariationalInference` | Natural gradient on ELBO | Linear | Yes | [gp_state_space.md](gp_state_space.md) |
| `ExpectationPropagation` | Moment matching | — | No | [gp_state_space.md](gp_state_space.md) |
| `PosteriorLinearisation` | Statistical linearisation | Linear | Yes | [gp_state_space.md](gp_state_space.md) |
| `GaussNewton` | $J^T W J$ Hessian | Quadratic | Yes | [gp_state_space.md](gp_state_space.md) |

---

## Integrator Protocol

```python
class Integrator(eqx.Module):
    @abc.abstractmethod
    def integrate(self, fn, mean, var) -> Array: ...
    def points_and_weights(self, mean, var) -> tuple[Array, Array]: ...
```

Computes $\mathbb{E}_{q(f)}[g(f)]$ where $q(f) = \mathcal{N}(\mu, \sigma^2)$.

**Implementations:**

| Integrator | Points | Accuracy | Scaling in $P$ | Detail |
|---|---|---|---|---|
| `GaussHermite(K)` | $K^P$ | Order $2K-1$ | Exponential | [gp_integration.md](gp_integration.md) |
| `SigmaPoints` | $2P+1$ | 3rd order | Linear | [gp_integration.md](gp_integration.md) |
| `Cubature` | $2P$ | 3rd order | Linear | [gp_integration.md](gp_integration.md) |
| `Taylor(order)` | 1 | 1st or 2nd | $O(P)$ or $O(P^2)$ | [gp_integration.md](gp_integration.md) |
| `MonteCarlo(S)` | $S$ | $O(1/\sqrt{S})$ | Independent | [gp_integration.md](gp_integration.md) |

---

## Guide Protocol

```python
class Guide(eqx.Module):
    @abc.abstractmethod
    def sample(self, key) -> Array: ...
    @abc.abstractmethod
    def log_prob(self, f) -> Scalar: ...
```

**Whitening principle:** $f = L\epsilon + \mu$ where $\epsilon \sim \mathcal{N}(0, I)$ and $L = \operatorname{chol}(\Sigma)$. The guide optimizes $\epsilon$-space, which has unit-scale geometry regardless of the prior covariance.

**Guide implementations** (from the VGP/SVGP reference implementations):

| Guide | Parameters | Covariance $\tilde{S}$ | Memory | Use case |
|---|---|---|---|---|
| `DeltaGuide` (MAP) | $\tilde{m}$ only | $\tilde{S} = 0$ | $O(M)$ | Point estimate, no uncertainty |
| `MeanFieldGuide` | $\tilde{m}, \sigma$ | $\text{diag}(\sigma^2)$ | $O(M)$ | Independent marginal uncertainty |
| `LowRankGuide` | $\tilde{m}, L_r, \sigma$ | $L_r L_r^T + \text{diag}(\sigma^2)$ | $O(Mr)$ | Correlated uncertainty, rank-$r$ |
| `FullRankGuide` | $\tilde{m}, \tilde{L}$ | $\tilde{L}\tilde{L}^T$ | $O(M^2)$ | Full posterior covariance |
| `OrthogonalDecoupledGuide` | Separate mean/cov inducing | Decoupled $A_m, A_c$ | $O(M_m + M)$ | Independent mean/covariance control |
| `FlowGuide` | Normalizing flow params | Non-Gaussian | $O(MK)$ | Flexible posterior (non-Gaussian) |
| `WhitenedGuide` | Wraps any of the above | Whitened $v$-space | Same | Improved optimization landscape |
| `InducingPointGuide` | $M \ll N$ inducing variables | Sparse variational | $O(NM^2)$ | Large $N$ via inducing points |
| `KalmanGuide` | Temporal structure | Banded/state-space | $O(NS^3)$ | Temporal GPs (Markovian) |

See [../examples/vgp_numpyro.py](../examples/vgp_numpyro.py) and [../examples/svgp_numpyro.py](../examples/svgp_numpyro.py) for complete implementations of all guide families with detailed equation annotations.

---

## InducingFeatures Protocol

```python
class InducingFeatures(eqx.Module):
    @abc.abstractmethod
    def K_uu(self, kernel) -> Array: ...
    @abc.abstractmethod
    def k_u(self, X, kernel) -> Array: ...
    def is_diagonal(self) -> bool: ...
```

Inter-domain inducing features replace spatial inducing points $Z$ with RKHS inner products $u_m = \langle f, \phi_m \rangle_{\mathcal{H}}$. The SVGP framework (Guide, ELBO, predictions) is unchanged; only $K_{uu}$ and $k_u(x)$ are computed differently. For orthogonal eigenfunction bases, $K_{uu}$ is diagonal, reducing all $O(M^3)$ operations to $O(M)$.

**Implementations:**

| InducingFeatures | Domain | Basis | $K_{uu}$ | Detail |
|---|---|---|---|---|
| `SphericalHarmonicFeatures` | $S^{d-1}$ | Spherical harmonics $Y_\ell^m$ | Diagonal | [features/gp_inducing_features.md](../../features/gp/inducing_features.md) |
| `FourierFeatures` | $[-L, L]^D$ | Laplacian eigenfunctions | Diagonal | [features/gp_inducing_features.md](../../features/gp/inducing_features.md) |
| `LaplacianFeatures` | Compact manifold | Precomputed eigenpairs | Diagonal | [features/gp_inducing_features.md](../../features/gp/inducing_features.md) |

---

## PathwiseSampler

Efficient posterior function samples via Matheron's rule. Not a protocol — a concrete model-layer component that composes `ConditionedGP` with `gaussx.matheron_update`.

| Sampler | Use case | Prior basis | Update basis | Cost per point |
|---|---|---|---|---|
| `PathwiseSampler` | Exact GP | RFF ($D$ features) | Training points ($N$) | $O(D + N)$ |
| `DecoupledPathwiseSampler` | Sparse GP | RFF ($D$ features) | Inducing points ($M$) | $O(D + M)$ |

See [features/gp_pathwise_sampling.md](../../features/gp/pathwise_sampling.md) and [models.md](models.md) for full API.
