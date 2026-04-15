---
status: draft
version: 0.1.0
---

# pyrox.nn x Edward2 Layer Gaps

**Source:** [google/edward2](https://github.com/google/edward2)

Layers from Edward2 not currently in pyrox.nn, organized by priority tier.

---

## Tier 1 — High Value

### Gap 1: DenseRank1

Rank-1 BNN (Dusenberry et al., 2020). Shared kernel with per-ensemble-member rank-1 multiplicative perturbations.

**Math:**

$$y_i = \phi\!\bigl((r_i \circ W s_i)\,x + b_i\bigr), \quad r_i \sim p(r),\; s_i \sim p(s), \quad i = 1,\ldots,M$$

where $r_i \in \mathbb{R}^{d_\text{out}}$, $s_i \in \mathbb{R}^{d_\text{in}}$, $W$ is shared, and $\circ$ is elementwise.

**Complexity:** $O(d_\text{in} \cdot d_\text{out})$ per member (same as standard dense). $M$ members share $W$; overhead is $M(d_\text{in} + d_\text{out})$ for the rank-1 vectors.

```python
class DenseRank1(PyroxModule):
    output_dim: int
    ensemble_size: int
    alpha_init: Float[Array, "M d_out"]
    gamma_init: Float[Array, "M d_in"]

    def __call__(self, x: Float[Array, "N d_in"]) -> Float[Array, "M N d_out"]: ...
```

**Ref:** [edward2/tensorflow/layers/dense.py#DenseRank1](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/dense.py)

---

### Gap 2: RandomFeatureGaussianProcess + LaplaceRandomFeatureCovariance

Full SNGP output layer (Liu et al., 2020). RFF feature map + online precision accumulation during training + Laplace covariance at test time.

**Math (train):** Accumulate precision matrix from RFF features $\phi(x)$:

$$\hat{\Sigma}^{-1} \leftarrow (1 - m)\,\hat{\Sigma}^{-1} + m\,\frac{1}{B}\sum_{b=1}^{B} \phi(x_b)\,\phi(x_b)^\top$$

**Math (test):** Predictive variance via Laplace approximation:

$$\sigma^2(x_*) = \phi(x_*)^\top \hat{\Sigma}\,\phi(x_*), \qquad \hat{\Sigma} = \bigl(\hat{\Sigma}^{-1}\bigr)^{-1}$$

**Complexity:** Train: $O(B D^2)$ per batch to update precision ($D$ = RFF dim). Test: $O(D^2)$ per point (after one-time $O(D^3)$ Cholesky inversion).

```python
class LaplaceRandomFeatureCovariance(eqx.Module):
    precision: Float[Array, "D D"]
    momentum: float

    def update(self, features: Float[Array, "B D"]) -> "LaplaceRandomFeatureCovariance": ...
    def covariance(self) -> Float[Array, "D D"]: ...

class RandomFeatureGaussianProcess(eqx.Module):
    rff: RandomFourierFeatures
    output_linear: eqx.nn.Linear
    covariance: LaplaceRandomFeatureCovariance

    def __call__(
        self, x: Float[Array, "N d_in"], *, return_cov: bool = False,
    ) -> Float[Array, "N d_out"] | tuple[Float[Array, "N d_out"], Float[Array, "N"]]: ...
```

**Ref:** [edward2/tensorflow/layers/random_feature.py](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py)

---

### Gap 3: MCSoftmaxDenseFA / MCSigmoidDenseFA

Heteroscedastic output layer (Collier et al., 2021). Input-dependent noise via low-rank + diagonal covariance with MC integration.

**Math:**

$$\text{logit}(x) = W_\mu x + \epsilon, \quad \epsilon \sim \mathcal{N}\!\bigl(0,\; W_r W_r^\top + \text{diag}(\sigma^2)\bigr)$$

$$p(y = k \mid x) \approx \frac{1}{S}\sum_{s=1}^{S} \text{softmax}_k\!\bigl(\text{logit}(x) + \epsilon_s\bigr)$$

where $W_r \in \mathbb{R}^{C \times r}$ (low-rank) and $\sigma \in \mathbb{R}^C$ (diagonal).

**Complexity:** $O(NCr + NCrS)$ where $N$ = batch, $C$ = classes, $r$ = rank, $S$ = MC samples.

```python
class MCSoftmaxDenseFA(PyroxModule):
    num_classes: int
    rank: int
    num_mc_samples: int
    loc_layer: eqx.nn.Linear
    scale_layer: eqx.nn.Linear     # low-rank factor (d_in -> num_classes * rank)
    diag_layer: eqx.nn.Linear      # diagonal factor (d_in -> num_classes)

    def __call__(
        self, x: Float[Array, "N d_in"], *, key: PRNGKeyArray,
    ) -> Float[Array, "N C"]: ...

class MCSigmoidDenseFA(PyroxModule):
    """Same structure, sigmoid instead of softmax. For multi-label / binary."""
    ...
```

**Ref:** [edward2/tensorflow/layers/heteroscedastic.py](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/heteroscedastic.py)

---

### Gap 4: DenseVariationalDropout

Variational dropout with learned per-weight dropout rates (Kingma et al., 2015; Molchanov et al., 2017). Enables automatic sparsity.

**Math:** Multiplicative noise on weights:

$$y = (W \circ z)\,x, \quad z_{ij} \sim \mathcal{N}(1, \alpha_{ij}), \quad \alpha_{ij} = \frac{p_{ij}}{1 - p_{ij}}$$

where $\alpha_{ij}$ is the learnable dropout rate in log-space. High $\alpha$ → weight is prunable.

**KL regularizer** (additive, computed analytically):

$$\mathrm{KL} \approx -\frac{1}{2}\sum_{ij}\log\alpha_{ij} + C$$

**Complexity:** $O(d_\text{in} \cdot d_\text{out})$ forward; same parameter count as standard dense (doubled: $W$ + $\log\alpha$).

```python
class DenseVariationalDropout(PyroxModule):
    output_dim: int
    log_alpha_init: float         # initial log-dropout-rate (e.g., -5.0)
    threshold: float              # pruning threshold on alpha (e.g., 3.0)

    def __call__(self, x: Float[Array, "N d_in"]) -> Float[Array, "N d_out"]: ...
    def sparsity(self) -> float: ...  # fraction of weights above threshold
```

**Ref:** [edward2/tensorflow/layers/dense.py#DenseVariationalDropout](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/dense.py)

---

## Tier 2 — Valuable for Specific Use Cases

### Gap 5: DenseHierarchical

Horseshoe prior via hierarchical multiplicative noise (Louizos et al., 2017). Local + global shrinkage for automatic relevance determination.

**Math:**

$$W_{ij} = \theta_{ij} \cdot z_j^{(\text{local})} \cdot z^{(\text{global})}, \quad z_j^{(\text{local})} \sim \mathcal{N}(1, \sigma_{j}^2), \quad z^{(\text{global})} \sim \mathcal{N}(1, \tau^2)$$

The local noise $z_j$ prunes individual units; the global noise $z$ controls overall sparsity. $\sigma_j$ and $\tau$ are learnable.

**Complexity:** $O(d_\text{in} \cdot d_\text{out})$ forward. Extra parameters: $d_\text{in}$ (local) + 1 (global) log-variances.

```python
class DenseHierarchical(PyroxModule):
    output_dim: int
    prior_std: float
    local_log_var_init: float
    global_log_var_init: float

    def __call__(self, x: Float[Array, "N d_in"]) -> Float[Array, "N d_out"]: ...
```

**Ref:** [edward2/tensorflow/layers/dense.py#DenseHierarchical](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/dense.py)

---

### Gap 6: DenseDVI

Deterministic Variational Inference (Wu et al., 2018). Propagates means and covariances analytically through the network — no MC sampling.

**Math:** Given input moments $(\mu_x, \Sigma_x)$ and weight posterior $q(W) = \mathcal{N}(M, S)$:

$$\mu_y = M\,\mu_x, \qquad \Sigma_y = M\,\Sigma_x\,M^\top + \text{diag}(S \circ (\mu_x^2 + \text{diag}(\Sigma_x)))$$

Activation: moment-matched through the nonlinearity via Gauss-Hermite quadrature.

**Complexity:** $O(d_\text{in}^2 \cdot d_\text{out})$ per layer (propagating second-order statistics). Zero-variance gradients.

```python
class DenseDVI(eqx.Module):
    output_dim: int
    weight_mean: Float[Array, "d_out d_in"]
    weight_log_var: Float[Array, "d_out d_in"]

    def __call__(
        self, mean: Float[Array, "N d_in"], var: Float[Array, "N d_in"],
    ) -> tuple[Float[Array, "N d_out"], Float[Array, "N d_out"]]: ...
```

**Ref:** [edward2/tensorflow/layers/dense.py#DenseDVI](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/dense.py)

---

### Gap 7: NCPNormalOutput

Output-side Noise Contrastive Prior (Hafner et al., 2018). Completes the NCP pattern — pyrox.nn only has the input-side `NCPContinuousPerturb`.

**Math:** Given doubled batch $[\hat{y}_\text{clean}; \hat{y}_\text{noisy}]$ from `NCPContinuousPerturb`:

$$\mathcal{L}_\text{NCP} = \mathrm{KL}\!\bigl[\mathcal{N}(\hat{y}_\text{noisy},\,\hat{\sigma}_\text{noisy}^2) \;\|\; \mathcal{N}(\mu_\text{prior},\,\sigma_\text{prior}^2)\bigr]$$

Regularizes the network to have prior-like uncertainty on perturbed inputs.

**Complexity:** $O(N \cdot d_\text{out})$ — just a KL between two Gaussians.

```python
class NCPNormalOutput(eqx.Module):
    prior_mean: float
    prior_std: float

    def __call__(
        self, y_clean: Float[Array, "N d"], y_noisy: Float[Array, "N d"],
    ) -> tuple[Float[Array, "N d"], Scalar]: ...
    # returns (y_clean, kl_loss)
```

**Ref:** [edward2/tensorflow/layers/noise.py#NCPNormalOutput](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/noise.py)

---

### Gap 8: MultiHeadDotProductAttentionBE

BatchEnsemble multi-head attention. Applies per-member rank-1 perturbations to Q, K, V, O projections.

**Math:** Standard multi-head attention but each linear projection uses BatchEnsemble:

$$Q_i = (r_i^Q \circ W_Q \, s_i^Q)\,x, \quad K_i = (r_i^K \circ W_K \, s_i^K)\,x, \quad V_i = (r_i^V \circ W_V \, s_i^V)\,x$$

Attention weights $\text{softmax}(Q_i K_i^\top / \sqrt{d_k})$ are deterministic.

**Complexity:** Same as standard attention: $O(T^2 d)$. Overhead: $4M(d_\text{in} + d_\text{head})$ for rank-1 vectors across Q/K/V/O.

```python
class MultiHeadAttentionBE(eqx.Module):
    embed_dim: int
    num_heads: int
    ensemble_size: int
    # per-member rank-1 vectors for Q, K, V, O projections

    def __call__(
        self,
        query: Float[Array, "T D"],
        key: Float[Array, "S D"],
        value: Float[Array, "S D"],
    ) -> Float[Array, "M T D"]: ...
```

**Ref:** [edward2/jax/nn/dense_heteroscedastic.py](https://github.com/google/edward2/blob/main/edward2/jax/nn/dense_heteroscedastic.py) (JAX impl)

---

### Gap 9: EnsembleNormalization

Per-ensemble-member normalization layers. Required for BatchEnsemble / Rank1 in architectures with normalization.

**Math (LayerNorm):** Standard layer norm but with per-member scale/bias:

$$\hat{x}_i = \gamma_i \cdot \frac{x - \mu}{\sigma} + \beta_i, \quad i = 1,\ldots,M$$

**Complexity:** $O(N \cdot d)$ per member. Overhead: $M \cdot 2d$ parameters (scale + bias per member).

```python
class LayerNormEnsemble(eqx.Module):
    feature_dim: int
    ensemble_size: int
    scales: Float[Array, "M d"]
    biases: Float[Array, "M d"]

    def __call__(self, x: Float[Array, "M N d"]) -> Float[Array, "M N d"]: ...
```

**Ref:** [edward2/jax/nn/normalization.py](https://github.com/google/edward2/blob/main/edward2/jax/nn/normalization.py) (JAX impl)

---

## References

1. Dusenberry, M. W., et al. (2020). *Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors.* ICML.
2. Liu, J. Z., et al. (2020). *Simple and Principled Uncertainty Estimation with Deterministic Deep Learning.* NeurIPS.
3. Collier, M., et al. (2021). *Correlated Input-Dependent Label Noise in Large-Scale Image Classification.* CVPR.
4. Kingma, D. P., Salimans, T., & Welling, M. (2015). *Variational Dropout and the Local Reparameterization Trick.* NeurIPS.
5. Molchanov, D., Ashukha, A., & Vetrov, D. (2017). *Variational Dropout Sparsifies Deep Neural Networks.* ICML.
6. Louizos, C., Ullrich, K., & Welling, M. (2017). *Bayesian Compression for Deep Learning.* NeurIPS.
7. Wu, A., et al. (2018). *Deterministic Variational Inference for Robust Bayesian Neural Networks.* ICLR.
8. Hafner, D., et al. (2018). *Noise Contrastive Priors for Functional Uncertainty.* UAI.
