# Estimator API

The `pyrox_nn.api` subpackage exposes sklearn-style immutable estimator facades. Each estimator wraps a `pyrox_nn` model + the inference runners in `pyrox.inference` behind a one-call `fit`/`predict` ergonomic.

## Base contracts

::: pyrox_nn.api.EstimatorBase

::: pyrox_nn.api.FittedEstimator

## Bayesian Neural Field family

::: pyrox_nn.api.BNFEstimator

::: pyrox_nn.api.BNFEstimatorMLE

::: pyrox_nn.api.BNFEstimatorVI

::: pyrox_nn.api.FittedBNF

## Neural fields on arrays and xarray

`NeuralFieldEstimator` fits any Bayesian neural field, a callable that maps
(encoded) coordinates to a field while registering its weights as NumPyro
sites (`BayesianSIREN`, `SphericalHarmonicFeatures`, `BayesianNeuralField`,
…), with `pyrox.inference.ensemble_map` or `ensemble_vi`. It takes plain
arrays with optional per-row noise and a validity mask, so padded batches
fit exactly like their valid rows, or an xarray Dataset (`fit_xarray`,
`pip install pyrox-nn[xarray]`) whose missing cells are masked; datetimes
become days since the first timestamp (or `time_origin=`), and
`FittedNeuralField.predict_xarray` maps a Dataset's grid the same way.
Under MAP, `pyrox_param` sites and constrained latents (e.g. a LogNormal
prior) are optimised in unconstrained space against the constrained-space
density (no Jacobian), and
fields that draw `numpyro.prng_key()` are rejected (use `inference="vi"`). The BNF
estimators above are this estimator on a `BayesianNeuralField`.

```python
import geonnax
from pyrox_nn import BayesianSIREN
from pyrox_nn.api import NeuralFieldEstimator

est = NeuralFieldEstimator(
    model_fn=lambda: BayesianSIREN.init(441, 128, 1, depth=4, first_omega=1.0),
    encoder=geonnax.SphericalHarmonicEncoder(l_max=20, input_mode="lonlat"),
    noise="learned", ensemble_size=8, num_epochs=2000,
)
fitted = est.fit(X, y, sigma=sig, valid=mask, seed=0)  # X (N, 2) lon/lat in radians
mean, (q05, q95) = fitted.predict(X_new, quantiles=(0.05, 0.95))
```

::: pyrox_nn.api.NeuralFieldEstimator

::: pyrox_nn.api.FittedNeuralField
