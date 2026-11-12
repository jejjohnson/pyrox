# Likelihoods API

The `pyrox.likelihoods` module holds the protocols that every observation model and variational guide in the pyrox packages subclasses, so a package built on the core can take a pyrox likelihood without depending on pyrox-gp. It carries no numerics: expectations and per-site derivatives of a likelihood are computed by gaussx in the packages that need them. `pyrox_gp` re-exports the protocols, so `pyrox_gp.Likelihood is pyrox.likelihoods.Likelihood`.

## Protocols

::: pyrox.likelihoods.Likelihood

::: pyrox.likelihoods.Guide

## Observation models

`AbstractObservation` is the base of an observation model that declares its own hyperparameters and builds a gaussx likelihood holding the data; `latent_dim` (1 by default) is the number of latents per site. `pyrox_lgm` re-exports it, so `pyrox_lgm.AbstractObservation is pyrox.likelihoods.AbstractObservation`.

pyrox-gp's likelihoods are observation models too: each one's `build(y, theta, data)` returns the gaussx likelihood below, with the inputs in `data["X"]`, and `log_prob(f, y, X)` evaluates it. `pyrox_gp.GaussianLikelihood` subclasses `pyrox.likelihoods.GaussianLikelihood`, so the closed-form paths that dispatch on the core class take it.

| pyrox name | gaussx likelihood it builds |
|---|---|
| `pyrox_gp.GaussianLikelihood` | `gaussx.GaussianLikelihood` |
| `pyrox_gp.BernoulliLikelihood` | `gaussx.BernoulliLikelihood` |
| `pyrox_gp.PoissonLikelihood` | `gaussx.PoissonLikelihood` |
| `pyrox_gp.StudentTLikelihood` | `gaussx.StudentTLikelihood` |
| `pyrox_gp.SoftmaxLikelihood` | `gaussx.SoftmaxLikelihood` |
| `pyrox_gp.HeteroscedasticGaussianLikelihood` | `gaussx.HeteroscedasticGaussianLikelihood` |

::: pyrox.likelihoods.AbstractObservation

## Gaussian observations

::: pyrox.likelihoods.GaussianLikelihood

## Helpers

::: pyrox.likelihoods.likelihood_accepts_inputs
