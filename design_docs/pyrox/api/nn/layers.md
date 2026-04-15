---
status: draft
version: 0.1.0
---

# Components — Probabilistic Layers (`pyrox.nn`)

## Dense (`nn.dense`)

| Layer | Key Parameters | Description |
|---|---|---|
| `DenseVariational` | `output_dim`, `prior_std`, `use_bias`, `learn_prior` | Dense layer with Normal prior on weights, variational posterior via `pyrox_sample`. Optionally learnable prior std. |
| `DenseReparameterization` | `output_dim`, `prior_std` | Dense layer with reparameterized weight posteriors |
| `DenseFlipout` | `output_dim`, `prior_std` | Dense layer with Flipout estimator for lower-variance gradients |
| `DenseBatchEnsemble` | `output_dim`, `ensemble_size` | Batch ensemble dense layer (rank-1 perturbations) |

## Random Features (`nn.random_feature`)

| Layer | Key Parameters | Description |
|---|---|---|
| `RandomFourierFeatures` | `n_features`, `learn_lengthscale`, `init_lengthscale` | RBF kernel approximation via random projections. Output dim = `2 * n_features` (cos + sin). Optionally learnable lengthscale. |
| `RandomKitchenSinks` | `n_features`, `lengthscale` | Random Kitchen Sinks projection layer |

## Dropout (`nn.dropout`)

| Layer | Key Parameters | Description |
|---|---|---|
| `MCDropout` | `keep_prob` | Monte Carlo dropout — always stochastic (training and inference). Uses inverted dropout scaling. |

## Noise (`nn.noise`)

| Layer | Key Parameters | Description |
|---|---|---|
| `NCPContinuousPerturb` | `input_noise` | Doubles the batch by concatenating original inputs with noise-perturbed copies. Used with `DenseNCP`. (Hafner et al., 2018) |
| `DenseNCP` | `output_dim`, `prior_std`, `latent_mean`, `latent_std` | Variational dense layer with NCP KL penalty on perturbed predictions. Registers `ncp_kl` as a deterministic site. Should be the last layer. |

## GP Kernel Layers — Migration Note

The parameterized GP kernel layers (`RBFKernel`, `LinearKernel`, `MaternKernel`) and GP mean function layers that were previously in `pyrox_nn.layers.gaussian_process` now live in `pyrox.gp.kernels`. These are `Parameterized` subclasses with learnable parameters, priors, and autoguides. See [components.md](gp/components.md) for the full Kernel protocol and implementations.

```python
# Old (pyrox-nn)
from pyrox_nn.layers.gaussian_process import RBFKernel, LinearKernel, MaternKernel

# New (pyrox — merged package)
from pyrox.gp.kernels import RBF, Matern, Periodic, Linear
```
