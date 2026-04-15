---
status: draft
version: 0.1.0
---

# GP Layer 0 -- Primitive Examples

Pure JAX kernel evaluation, covariance construction, and GP math. All primitives live in `pyrox.gp._src`.

---

## Kernel Evaluation

```python
from pyrox.gp._src.kernels import rbf_kernel, matern_kernel

# Pure function -- no protocol, no class
K = rbf_kernel(X_train, X_train, variance=1.0, lengthscale=0.5)  # (N, N)

# Differentiable w.r.t. hyperparameters
grad_fn = jax.grad(lambda l: rbf_kernel(X, X, 1.0, l).sum())
dK_dl = grad_fn(0.5)
```

---

## Log Marginal Likelihood

```python
from pyrox.gp._src.covariance import log_marginal_likelihood

# The two atoms: solve + logdet
lml = log_marginal_likelihood(y, K, noise_var=0.1)
# = -0.5 * (y^T (K + sigma^2 I)^{-1} y + log|K + sigma^2 I| + N log(2 pi))
```

---

## Kalman Predict/Update

```python
from pyrox.gp._src.kalman import kalman_predict, kalman_update

# One step of Kalman filtering -- pure functions
m_pred, P_pred = kalman_predict(m, P, A, Q)
m_filt, P_filt = kalman_update(m_pred, P_pred, H, R, y_t)
```

---

## State-Space Conversion

```python
from pyrox.gp._src.sde import matern_to_sde

# Matern-3/2 -> LTI SDE with state dim S=2
A, Q, H, P_inf = matern_to_sde(variance=1.0, lengthscale=1.0, nu=1.5)
# A: (2,2), Q: (2,2), H: (1,2), P_inf: (2,2)
```

---

*For how these primitives are wrapped by protocols, see [gp_components.md](gp_components.md). For full GP workflows, see [gp_models.md](gp_models.md).*
