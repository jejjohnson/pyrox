"""State-space models as NumPyro models.

Temporal GPs and Gaussian state-space models on top of pyrox and gaussx, in
the package whose inference recurses in time. ``pyrox_ssm`` never imports
``pyrox_gp``, ``pyrox_lgm`` or ``pyrox_nn``.

* Linear-Gaussian state-space models: `LinearGaussianSSMPrior`,
  `ConditionedLinearGaussianSSM`, `lgssm_factor`; the general model given
  by its transition, observation and noise matrices.
* Markov GPs: `MarkovGPPrior`, `ConditionedMarkovGP`, `markov_gp_factor`,
  `markov_gp_sample`; the Kalman filter and RTS smoother over an SDE kernel.
  `markov_gp_from_innovations` and `markov_gp_innovations_sample` build the
  latent path from whitened innovations in linear time.
* Sparse Markov GPs: `SparseMarkovGPPrior`, `ConditionedSparseMarkovGP`,
  `sparse_markov_elbo`, `sparse_markov_factor`.
* Non-Gaussian likelihoods on a Markov GP: `LaplaceMarkovInference`,
  `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov`,
  `ExpectationPropagationMarkov`, all returning `ConditionedNonGaussMarkovGP`,
  and `markov_laplace_factor`, the approximate evidence as a NumPyro factor.
* Nonlinear state-space models: `nonlinear_ssm_factor`, the
  moment-matched log likelihood of `gaussx.nonlinear_kalman_filter` (EKF,
  UKF, CKF, Gauss-Hermite or Monte Carlo) as a NumPyro factor.
* Ragged sites: `pad_sites` pads per-site series to one shape and returns
  the observation mask every Markov entry point takes;
  `markov_gp_sites_factor` scores the padded sites as one factor, and
  `noncentred_log_normal` draws per-site hyperparameters non-centred.
* Shared plus per-site GPs: `SharedPlusSiteMarkovGPPrior` builds a shared
  GP and one GP per site as one linear-Gaussian SSM on the merged time grid.
  `ConditionedSharedPlusSiteMarkovGP` is its conditioned form and
  `shared_plus_site_factor` its NumPyro factor.
* The gaussx SDE kernels (`MaternSDE`, `SumSDE`, …) and the `SDEKernel`
  protocol, re-exported for convenience.

The normalizing Kalman filter lives in `pyrox_ssm.flows`, behind the
``flows`` extra, and the ensemble Kalman log likelihood of a filterax run,
`ensemble_factor`, in `pyrox_ssm.ensemble`, behind the ``ensemble`` extra.

Three names changed to one order, ``Conditioned<Model>`` for conditioned
results and ``<Model>Prior`` for priors: ``SparseConditionedMarkovGP`` is now
`ConditionedSparseMarkovGP`, ``NonGaussConditionedMarkovGP`` is
`ConditionedNonGaussMarkovGP` and ``SharedPlusSiteMarkovGP`` is
`SharedPlusSiteMarkovGPPrior`. The old names still import, with a
`DeprecationWarning`, and are the same class objects.
"""

import warnings

# State-space (SDE) kernels live in gaussx; re-export them so a Markov GP
# needs one import.
from gaussx import (
    ConstantSDE,
    CosineSDE,
    IntegratedWienerSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SDEKernel,
    SDEParams,
    SumSDE,
)

from pyrox_ssm._hierarchical import (
    ConditionedSharedPlusSiteMarkovGP,
    SharedPlusSiteMarkovGPPrior,
    shared_plus_site_factor,
)
from pyrox_ssm._inference_nongauss_markov import (
    ConditionedNonGaussMarkovGP,
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    PosteriorLinearizationMarkov,
    markov_implicit_adjoint,
    markov_laplace_factor,
)
from pyrox_ssm._lgssm import (
    ConditionedLinearGaussianSSM,
    LinearGaussianSSMPrior,
    lgssm_factor,
)
from pyrox_ssm._markov import (
    ConditionedMarkovGP,
    MarkovGPPrior,
    markov_gp_factor,
    markov_gp_from_innovations,
    markov_gp_innovations_sample,
    markov_gp_sample,
)
from pyrox_ssm._nonlinear import nonlinear_ssm_factor
from pyrox_ssm._sites import (
    markov_gp_sites_factor,
    noncentred_log_normal,
    pad_sites,
)
from pyrox_ssm._sparse_markov import (
    ConditionedSparseMarkovGP,
    SparseMarkovGPPrior,
    sparse_markov_elbo,
    sparse_markov_factor,
)


__version__ = "0.2.1"  # x-release-please-version

# Old names, kept as deprecated aliases of the same class objects.
_RENAMED = {
    "SparseConditionedMarkovGP": "ConditionedSparseMarkovGP",
    "NonGaussConditionedMarkovGP": "ConditionedNonGaussMarkovGP",
    "SharedPlusSiteMarkovGP": "SharedPlusSiteMarkovGPPrior",
}


def __getattr__(name: str):
    """Serve a renamed class under its old name, with a `DeprecationWarning`."""
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(
            f"pyrox_ssm.{name} is deprecated; use pyrox_ssm.{new}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module 'pyrox_ssm' has no attribute {name!r}")


__all__ = [
    "ConditionedLinearGaussianSSM",
    "ConditionedMarkovGP",
    "ConditionedNonGaussMarkovGP",
    "ConditionedSharedPlusSiteMarkovGP",
    "ConditionedSparseMarkovGP",
    "ConstantSDE",
    "CosineSDE",
    "ExpectationPropagationMarkov",
    "GaussNewtonMarkovInference",
    "IntegratedWienerSDE",
    "LaplaceMarkovInference",
    "LinearGaussianSSMPrior",
    "MarkovGPPrior",
    "MaternSDE",
    "PeriodicSDE",
    "PosteriorLinearizationMarkov",
    "ProductSDE",
    "QuasiPeriodicSDE",
    "SDEKernel",
    "SDEParams",
    "SharedPlusSiteMarkovGPPrior",
    "SparseMarkovGPPrior",
    "SumSDE",
    "__version__",
    "lgssm_factor",
    "markov_gp_factor",
    "markov_gp_from_innovations",
    "markov_gp_innovations_sample",
    "markov_gp_sample",
    "markov_gp_sites_factor",
    "markov_implicit_adjoint",
    "markov_laplace_factor",
    "noncentred_log_normal",
    "nonlinear_ssm_factor",
    "pad_sites",
    "shared_plus_site_factor",
    "sparse_markov_elbo",
    "sparse_markov_factor",
]
"""pyrox_ssm recurses in time and must not depend on its sibling packages."""
