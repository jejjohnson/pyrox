# NN API

The `pyrox.nn` subpackage ships uncertainty-aware neural network layers in three families:

1. **Dense / Bayesian-linear layers** (`pyrox.nn._layers`) — twelve layers covering reparameterization, Flipout, NCP, MC-Dropout, and several random-feature variants.
2. **Bayesian Neural Field stack** (`pyrox.nn._bnf`) — five layers that together implement the BNF architecture (Saad et al., Nat. Comms. 2024).
3. **Pure-JAX feature helpers** (`pyrox.nn._features`) — pandas-free building blocks the BNF layers wrap.

## Dense / Bayesian-linear layers

::: pyrox.nn.DenseReparameterization

::: pyrox.nn.DenseFlipout

::: pyrox.nn.DenseVariational

::: pyrox.nn.MCDropout

::: pyrox.nn.DenseNCP

::: pyrox.nn.NCPContinuousPerturb

::: pyrox.nn.RBFFourierFeatures

::: pyrox.nn.RBFCosineFeatures

::: pyrox.nn.MaternFourierFeatures

::: pyrox.nn.LaplaceFourierFeatures

::: pyrox.nn.ArcCosineFourierFeatures

::: pyrox.nn.RandomKitchenSinks

## Wave-4 spectral layers (#41)

::: pyrox.nn.VariationalFourierFeatures

::: pyrox.nn.OrthogonalRandomFeatures

::: pyrox.nn.HSGPFeatures

## Bayesian Neural Field stack

::: pyrox.nn.Standardization

::: pyrox.nn.FourierFeatures

::: pyrox.nn.SeasonalFeatures

::: pyrox.nn.InteractionFeatures

::: pyrox.nn.BayesianNeuralField

## Pure-JAX feature helpers

::: pyrox.nn.fourier_features

::: pyrox.nn.seasonal_features

::: pyrox.nn.seasonal_frequencies

::: pyrox.nn.interaction_features

::: pyrox.nn.standardize

::: pyrox.nn.unstandardize
