---
status: draft
version: 0.1.0
---

# GP Layer 1 -- Component Examples

Protocols in action: Kernel, Solver, InferenceStrategy, Integrator, Guide. All components live in `pyrox.gp`.

---

## Kernel + Solver

```python
from pyrox.gp.kernels import RBF, Matern
from pyrox.gp.solvers import CholeskySolver, WoodburySolver

# Kernel evaluates to a covariance representation
kernel = RBF(variance=1.0, lengthscale=0.5)
rep = kernel(X_train, X_train)  # DenseCov

# Solver operates on the representation
solver = CholeskySolver()
alpha = solver.solve(rep, y)
lml = solver.log_marginal(rep, y)
```

---

## Swapping Solvers

```python
from pyrox.gp.solvers import CholeskySolver, CGSolver, KalmanSolver

# Same kernel, different solver -- the layer stack is orthogonal
lml_chol = CholeskySolver().log_marginal(K, y)      # O(N^3) exact
lml_cg   = CGSolver(rtol=1e-5).log_marginal(K, y)   # O(N^2 k) iterative
```

---

## InferenceStrategy + Integrator

```python
from pyrox.gp.strategies import VariationalInference, PosteriorLinearisation
from pyrox.gp.integrators import GaussHermite, SigmaPoints

# VI with Gauss-Hermite quadrature
vi_gh = VariationalInference(integrator=GaussHermite(K=20))

# Posterior linearisation with sigma points (= unscented Kalman smoother)
pl_sp = PosteriorLinearisation(integrator=SigmaPoints())
```

---

## Guide Construction

```python
from pyrox.gp.guides import WhitenedGuide, InducingPointGuide

# Whitened guide: f = L epsilon + mu, optimize in epsilon-space
guide = WhitenedGuide(prior=gp_prior)

# Sparse: only M inducing points are variational parameters
guide = InducingPointGuide(Z=inducing_locations, prior=gp_prior)
```

---

*For primitive functions underneath these protocols, see [gp_primitives.md](gp_primitives.md). For full GP workflows, see [gp_models.md](gp_models.md).*
