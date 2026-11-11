"""State-space models as NumPyro models.

Temporal GPs and Gaussian state-space models on top of pyrox and gaussx, in
the package whose inference recurses in time. ``pyrox_ssm`` never imports
``pyrox_gp``, ``pyrox_lgm`` or ``pyrox_nn``.

* Linear-Gaussian state-space models: `LinearGaussianSSMPrior`,
  `ConditionedLinearGaussianSSM`, `lgssm_factor`; the general model given
  by its transition, observation and noise matrices.
* Markov GPs: `MarkovGPPrior`, `ConditionedMarkovGP`, `markov_gp_factor`,
  `markov_gp_sample`; the Kalman filter and RTS smoother over an SDE kernel.
* Sparse Markov GPs: `SparseMarkovGPPrior`, `SparseConditionedMarkovGP`,
  `sparse_markov_elbo`, `sparse_markov_factor`.
* Non-Gaussian likelihoods on a Markov GP: `LaplaceMarkovInference`,
  `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov`,
  `ExpectationPropagationMarkov`, all returning `NonGaussConditionedMarkovGP`,
  and `markov_laplace_factor`, the approximate evidence as a NumPyro factor.
* Ragged sites: `pad_sites` pads per-site series to one shape and returns
  the observation mask every Markov entry point takes.
* The gaussx SDE kernels (`MaternSDE`, `SumSDE`, …) and the `SDEKernel`
  protocol, re-exported for convenience.

The normalizing Kalman filter lives in `pyrox_ssm.flows`, behind the
``flows`` extra.
"""

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

from pyrox_ssm._inference_nongauss_markov import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    NonGaussConditionedMarkovGP,
    PosteriorLinearizationMarkov,
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
    markov_gp_sample,
)
from pyrox_ssm._sites import pad_sites
from pyrox_ssm._sparse_markov import (
    SparseConditionedMarkovGP,
    SparseMarkovGPPrior,
    sparse_markov_elbo,
    sparse_markov_factor,
)


__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "ConditionedLinearGaussianSSM",
    "ConditionedMarkovGP",
    "ConstantSDE",
    "CosineSDE",
    "ExpectationPropagationMarkov",
    "GaussNewtonMarkovInference",
    "IntegratedWienerSDE",
    "LaplaceMarkovInference",
    "LinearGaussianSSMPrior",
    "MarkovGPPrior",
    "MaternSDE",
    "NonGaussConditionedMarkovGP",
    "PeriodicSDE",
    "PosteriorLinearizationMarkov",
    "ProductSDE",
    "QuasiPeriodicSDE",
    "SDEKernel",
    "SDEParams",
    "SparseConditionedMarkovGP",
    "SparseMarkovGPPrior",
    "SumSDE",
    "__version__",
    "lgssm_factor",
    "markov_gp_factor",
    "markov_gp_sample",
    "markov_laplace_factor",
    "pad_sites",
    "sparse_markov_elbo",
    "sparse_markov_factor",
]
"""pyrox_ssm recurses in time and must not depend on its sibling packages."""
