---
status: draft
version: 0.1.0
---

# pyrox x Spectral Normalization: Gap Analysis

**Subject:** Spectral normalization layers for SNGP-style uncertainty,
sourced from Liu et al. (2020, 2022) and TensorFlow/Edward2.

**Date:** 2026-04-02

---

## 1  Scope

Spectral Normalized Gaussian Process (SNGP) uses spectral normalization
to preserve distance awareness in neural network feature extractors,
then places a GP (or RFF approximation) on the last layer. The spectral
normalization component belongs in `pyrox.nn`; the GP head composes with
`pyrox.nn`'s existing `RandomFourierFeatures` layer or `pyrox.gp`'s GP
primitives. Since both live in the same `pyrox` package, cross-module
composition is a straightforward internal import.

---

## 2  What pyrox.nn Already Provides

| Component | Status | Relevance |
|---|---|---|
| `RandomFourierFeatures` | Designed | RFF layer for the GP head |
| `DenseVariational` | Designed | Alternative Bayesian last layer |
| `MCDropout` | Designed | Alternative uncertainty method |

The missing piece is the spectral normalization wrapper that constrains
hidden layers to be bi-Lipschitz (distance-preserving).

---

## 3  Gap Catalog

### Gap 1: SpectralNormalization Wrapper

**Source:** Liu et al. (2020) *Simple and Principled Uncertainty
Estimation with Deterministic Deep Learning*, Edward2 `SpectralNormalization`

**Priority:** High

A layer wrapper that constrains the spectral norm of a linear layer's
weight matrix to be at most `coeff`. Uses power iteration to estimate
the top singular value.

```python
class SpectralNormalization(eqx.Module):
    """Spectral normalization wrapper for any linear layer.

    Constrains ||W||_2 <= coeff via power iteration. The normalized
    weight is W_hat = W * coeff / sigma(W) where sigma(W) is the
    estimated spectral norm.

    This is NOT a PyroxModule -- it's a deterministic wrapper. It
    constrains the feature extractor's Lipschitz constant, making
    the features suitable for distance-aware uncertainty methods
    (SNGP, DUE).
    """
    layer: eqx.Module          # wrapped layer (e.g., eqx.nn.Linear)
    coeff: float               # spectral norm upper bound (default: 0.95)
    n_power_iterations: int    # power iteration steps (default: 1)
    u: Float[Array, " H_out"]  # left singular vector estimate (state)
    v: Float[Array, " H_in"]   # right singular vector estimate (state)

    def __call__(self, x: Array) -> Array:
        """Forward pass with spectrally normalized weights."""
        ...

    def normalize_weight(self) -> Array:
        """Return W_hat = W * coeff / sigma_hat(W)."""
        ...
```

**Key design considerations:**
- Power iteration state (`u`, `v`) must be updated during training but
  frozen at inference. In Equinox's immutable model, this means returning
  an updated module from the forward pass (functional state update).
- Alternative: precompute the normalization before each training step
  rather than inside the forward pass. This avoids the state-update issue.
- `coeff < 1.0` (typically 0.95) ensures the mapping is contractive.
  `coeff = 1.0` preserves distances exactly.

**Estimated lines:** ~60

---

### Gap 2: SNGP Composition Pattern

**Source:** Liu et al. (2020, 2022), Edward2 SNGP tutorial

**Priority:** Medium (example, not a new class)

SNGP is not a single layer but a composition:
1. Spectrally normalized feature extractor (hidden layers)
2. RFF approximation of an RBF kernel (last hidden -> feature space)
3. Bayesian linear regression head (feature space -> output)

This is a **user-level pattern** in pyrox, not a class to implement.

```python
def sngp_model(x, y=None):
    # 1. Spectrally normalized feature extractor
    h = SpectralNormalization(eqx.nn.Linear(D_in, 256), coeff=0.95)(x)
    h = jax.nn.relu(h)
    h = SpectralNormalization(eqx.nn.Linear(256, 256), coeff=0.95)(h)
    h = jax.nn.relu(h)

    # 2. RFF layer (approximate GP kernel)
    rff = RandomFourierFeatures(n_features=128, init_lengthscale=1.0, key=key)
    phi = rff(h)  # (N, 256)

    # 3. Bayesian last layer
    w = numpyro.sample("w", dist.Normal(jnp.zeros(256), 1.0))
    logits = phi @ w

    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
```

**Estimated lines:** ~40 (example only)

---

### Gap 3: DUE (Deterministic Uncertainty Estimation)

**Source:** van Amersfoort et al. (2021) *On Feature Collapse and Deep
Kernel Learning for Single Forward Pass Uncertainty*

**Priority:** Low

DUE extends SNGP by replacing the RFF+BLR head with a proper inducing-point
GP (SVGP). The feature extractor is still spectrally normalized.

```python
def due_model(x, y=None):
    # Spectrally normalized feature extractor
    h = sn_feature_extractor(x)

    # Deep kernel GP via pyrox.gp (same package, internal import)
    kernel = DeepKernel(feature_extractor=None, base_kernel=RBF())
    prior = GPPrior(kernel=kernel, solver=WoodburySolver(), X=h)
    f = gp_sample("f", prior, guide=SparseGuide(Z=Z_features))

    numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)
```

This pattern composes `pyrox.nn` and `pyrox.gp` components. Since both
are submodules of the merged `pyrox` package, there is no cross-package
dependency -- the DUE example simply imports from both submodules:
- **pyrox.nn** owns: `SpectralNormalization`, `RandomFourierFeatures`
- **pyrox.gp** owns: `GPPrior`, `gp_sample`, `SparseGuide`, `DeepKernel`

**Estimated lines:** ~30 (example only)

---

## 4  Integration Plan

| Component | File | Type | Priority | Dependencies |
|---|---|---|---|---|
| `SpectralNormalization` | `nn/spectral.py` | Layer | High | `eqx.Module` |
| SNGP example | `examples/sngp.md` | Example | Medium | `SpectralNormalization`, `RandomFourierFeatures` |
| DUE example | `examples/due.md` | Example | Low | `pyrox.nn` + `pyrox.gp` (internal) |

---

## 5  References

1. Liu, J. Z., et al. (2020). *Simple and Principled Uncertainty
   Estimation with Deterministic Deep Learning.* NeurIPS.

2. Liu, J. Z., et al. (2022). *A Simple Approach to Improve
   Single-Model Deep Uncertainty via Distance-Awareness.* JMLR.

3. van Amersfoort, J., et al. (2021). *On Feature Collapse and Deep
   Kernel Learning for Single Forward Pass Uncertainty.* NeurIPS.

4. Miyato, T., et al. (2018). *Spectral Normalization for Generative
   Adversarial Networks.* ICLR.
