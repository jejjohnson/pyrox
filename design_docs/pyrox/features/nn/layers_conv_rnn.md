---
status: draft
version: 0.1.0
---

# pyrox x Additional Layer Families: Gap Analysis

**Subject:** Convolutional, recurrent, and attention-based probabilistic layers
sourced from TensorFlow Probability (Edward2), Bayesian Layers (Google),
and Flipout (Wen et al. 2018).

**Date:** 2026-04-02

---

## 1  Scope

pyrox.nn's Layer 1 currently provides dense, dropout, RFF, and NCP layer
families. This doc catalogs convolutional, recurrent, and attention
probabilistic layers that follow the same Edward2-style pattern: drop-in
Bayesian replacements for standard Equinox layers.

All layers are `eqx.Module` subclasses that register NumPyro sample sites
via `PyroxModule.pyrox_sample()`. Inference is delegated entirely to NumPyro.

---

## 2  What pyrox.nn Already Provides

| Family | Layers | Status |
|---|---|---|
| Dense | `DenseVariational`, `DenseReparameterization`, `DenseFlipout`, `DenseBatchEnsemble` | Designed |
| Random features | `RandomFourierFeatures`, `RandomKitchenSinks` | Designed |
| Dropout | `MCDropout` | Designed |
| Noise contrastive | `NCPContinuousPerturb`, `DenseNCP` | Designed |
| GP | `RBFKernel`, `LinearKernel`, `MaternKernel`, mean functions | Designed |

---

## 3  Gap Catalog

### Gap 1: Convolutional Layers

**Source:** TFP Edward2 (`Conv2DReparameterization`, `Conv2DFlipout`),
Bayesian Layers (Atanov et al.)

**Priority:** High

Convolutional counterparts of the dense Bayesian layers. Same prior/posterior
pattern, but over conv filter weights instead of dense weight matrices.

```python
class Conv2DReparameterization(PyroxModule):
    """Bayesian 2D convolution with reparameterized weight posteriors.

    Registers sample sites for kernel weights and (optionally) bias.
    Uses local reparameterization: sample activations, not weights.
    """
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int]
    padding: str  # "SAME" | "VALID"
    prior_std: float

    def __call__(self, x: Float[Array, "H W C"]) -> Float[Array, "H2 W2 C_out"]:
        ...


class Conv2DFlipout(PyroxModule):
    """Bayesian 2D convolution with Flipout estimator.

    Lower-variance gradient estimates than reparameterization
    by decorrelating weight perturbations across the batch.
    """
    # Same interface as Conv2DReparameterization
    ...
```

**Key design considerations:**
- Mirror Equinox's `eqx.nn.Conv2d` interface for drop-in replacement
- Support `groups` parameter for depthwise-separable variants
- Flipout requires batch dimension -- document this constraint

**Estimated lines:** ~120 (both layers)

---

### Gap 2: Recurrent Layers

**Source:** TFP Edward2 (`LSTMCellReparameterization`), Bayesian RNN
literature (Gal & Ghahramani 2016, Fortunato et al. 2017)

**Priority:** Medium

Bayesian LSTM/GRU cells with priors on recurrent and input-to-hidden weights.

```python
class LSTMCellVariational(PyroxModule):
    """Bayesian LSTM cell with variational weight posteriors.

    Places priors on all four gate weight matrices (input, forget,
    cell, output) and their biases. Compatible with jax.lax.scan
    for sequence processing.
    """
    input_size: int
    hidden_size: int
    prior_std: float

    def __call__(
        self,
        x: Float[Array, " D"],
        carry: tuple[Float[Array, " H"], Float[Array, " H"]],
    ) -> tuple[tuple[Float[Array, " H"], Float[Array, " H"]], Float[Array, " H"]]:
        ...


class GRUCellVariational(PyroxModule):
    """Bayesian GRU cell with variational weight posteriors."""
    input_size: int
    hidden_size: int
    prior_std: float

    def __call__(
        self,
        x: Float[Array, " D"],
        carry: Float[Array, " H"],
    ) -> tuple[Float[Array, " H"], Float[Array, " H"]]:
        ...
```

**Key design considerations:**
- Must work with `jax.lax.scan` for efficient sequence processing
- Weight sharing across time steps -- sample weights once per sequence, not per step
- Variational dropout on recurrent connections (Gal & Ghahramani 2016)

**Estimated lines:** ~150 (LSTM + GRU)

---

### Gap 3: Attention Layers

**Source:** Uncertainty-aware attention (Fan et al. 2020), Bayesian
Transformer literature

**Priority:** Low

Bayesian multi-head attention with uncertainty on query/key/value projections.

```python
class MultiHeadAttentionVariational(PyroxModule):
    """Multi-head attention with Bayesian Q/K/V projections.

    Each projection (query, key, value, output) uses variational
    weight posteriors. Attention weights themselves are deterministic
    (softmax of Q @ K^T / sqrt(d)).
    """
    embed_dim: int
    num_heads: int
    prior_std: float

    def __call__(
        self,
        query: Float[Array, "T D"],
        key: Float[Array, "S D"],
        value: Float[Array, "S D"],
    ) -> Float[Array, "T D"]:
        ...
```

**Key design considerations:**
- Only the linear projections are Bayesian, not the attention mechanism itself
- Can compose with existing `DenseFlipout` for the projections
- Consider whether full Bayesian attention (uncertainty on attention weights) is worth the complexity -- likely not for v1

**Estimated lines:** ~80

---

## 4  Integration Plan

| Layer | File | Priority | Dependencies |
|---|---|---|---|
| `Conv2DReparameterization` | `nn/conv.py` | High | `PyroxModule`, Equinox conv primitives |
| `Conv2DFlipout` | `nn/conv.py` | High | `PyroxModule`, Equinox conv primitives |
| `LSTMCellVariational` | `nn/recurrent.py` | Medium | `PyroxModule`, `jax.lax.scan` |
| `GRUCellVariational` | `nn/recurrent.py` | Medium | `PyroxModule` |
| `MultiHeadAttentionVariational` | `nn/attention.py` | Low | `PyroxModule`, `DenseFlipout` |

---

## 5  References

1. Wen, Y., Vicol, P., Ba, J., Tran, D., & Grosse, R. (2018).
   *Flipout: Efficient Pseudo-Independent Weight Perturbations on
   Mini-Batches.* ICLR.

2. Gal, Y. & Ghahramani, Z. (2016). *A Theoretically Grounded
   Application of Dropout in Recurrent Neural Networks.* NeurIPS.

3. Fortunato, M., Blundell, C., & Vinyals, O. (2017). *Bayesian
   Recurrent Neural Networks.* arXiv:1704.02798.

4. Fan, X., et al. (2020). *Bayesian Attention Modules.* arXiv:2010.10604.

5. Tran, D., et al. (2019). *Bayesian Layers: A Module for Neural
   Network Uncertainty.* NeurIPS.
