# Likelihoods API

The `pyrox.likelihoods` module holds the protocols that every observation model and variational guide in the pyrox packages subclasses, so a package built on the core can take a pyrox likelihood without depending on pyrox-gp. It carries no numerics: expectations and per-site derivatives of a likelihood are computed by gaussx in the packages that need them. `pyrox_gp` re-exports each name, so `pyrox_gp.Likelihood is pyrox.likelihoods.Likelihood`.

## Protocols

::: pyrox.likelihoods.Likelihood

::: pyrox.likelihoods.Guide

## Observation models

`AbstractObservation` is the base of an observation model that declares its own hyperparameters and builds a gaussx likelihood holding the data; `latent_dim` (1 by default) is the number of latents per site. `pyrox_lgm` re-exports it, so `pyrox_lgm.AbstractObservation is pyrox.likelihoods.AbstractObservation`.

::: pyrox.likelihoods.AbstractObservation

## Gaussian observations

::: pyrox.likelihoods.GaussianLikelihood

## Helpers

::: pyrox.likelihoods.likelihood_accepts_inputs
