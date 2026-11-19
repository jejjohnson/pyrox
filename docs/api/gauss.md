# Gauss API

`pyrox_gauss` holds the Gaussian NumPyro distributions whose covariance or precision is a [gaussx](https://github.com/jejjohnson/gaussx) operator. They moved here from gaussx with their names, constructors and numerics unchanged ([#453](https://github.com/jejjohnson/pyrox/issues/453)), so moving to them is an import change. `pyrox_gauss` imports gaussx and NumPyro and never imports `pyrox`; the model packages build these distributions from the hyperparameters they sample.

## Multivariate normals

::: pyrox_gauss.AbstractMultivariateNormal

::: pyrox_gauss.MultivariateNormal

::: pyrox_gauss.MultivariateNormalPrecision

## State-space models

`LGSSM` is a density over the observations with the states marginalised by the Kalman filter; `MarkovGaussian` is the density over the states themselves.

::: pyrox_gauss.LGSSM

::: pyrox_gauss.MaskedLGSSM

::: pyrox_gauss.LGSSMFactory

::: pyrox_gauss.MarkovGaussian

## GMRFs

The GMRF fields (`gaussx.GaussianMRF`, `IntrinsicGMRF`, `ConstrainedGMRF`, `BYM2GMRF`) stay in gaussx, because gaussx's Laplace and VB code and argminax use them as linear-algebra objects. `GMRFDistribution` hands one to `numpyro.sample`.

::: pyrox_gauss.GMRFDistribution
