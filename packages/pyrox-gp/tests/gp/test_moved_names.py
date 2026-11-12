"""Names that left pyrox-gp: the Markov track raises, the protocols are the core's.

The Markov-GP track and the SDE kernels moved to pyrox-ssm with no
deprecation window, so each old ``pyrox_gp`` name raises an `ImportError`
naming its new home. The `Likelihood` / `Guide` protocols moved to the pyrox
core and pyrox-gp re-exports the same objects.
"""

from __future__ import annotations

import importlib

import pyrox.likelihoods
import pyrox_gp
import pytest


MOVED_TO_SSM = [
    "ConditionedMarkovGP",
    "ConstantSDE",
    "CosineSDE",
    "ExpectationPropagationMarkov",
    "GaussNewtonMarkovInference",
    "IntegratedWienerSDE",
    "LaplaceMarkovInference",
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
    "markov_gp_factor",
    "markov_gp_sample",
    "sparse_markov_elbo",
    "sparse_markov_factor",
]
MOVED_TO_SSM_FLOWS = ["NormalizingKalmanPrior", "normalizing_kalman_factor"]


@pytest.mark.parametrize("name", MOVED_TO_SSM + MOVED_TO_SSM_FLOWS)
def test_importing_a_moved_name_raises_an_import_error_naming_pyrox_ssm(name) -> None:
    home = "pyrox_ssm.flows" if name in MOVED_TO_SSM_FLOWS else "pyrox_ssm"
    with pytest.raises(ImportError, match=rf"`from {home} import {name}`"):
        exec(f"from pyrox_gp import {name}", {})
    with pytest.raises(ImportError, match="moved to pyrox-ssm"):
        getattr(pyrox_gp, name)
    assert name not in pyrox_gp.__all__


@pytest.mark.parametrize("name", MOVED_TO_SSM)
def test_the_new_home_exports_the_name(name) -> None:
    assert hasattr(importlib.import_module("pyrox_ssm"), name)


@pytest.mark.parametrize("name", MOVED_TO_SSM_FLOWS)
def test_the_flows_home_exports_the_name(name) -> None:
    pytest.importorskip("flowjax")
    assert hasattr(importlib.import_module("pyrox_ssm.flows"), name)


def test_an_unknown_name_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NotAName'"):
        pyrox_gp.NotAName  # noqa: B018


@pytest.mark.parametrize("module", ["_markov", "_sparse_markov", "_markov_flow"])
def test_the_moved_modules_are_gone(module) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"pyrox_gp.{module}")


@pytest.mark.parametrize("name", ["Likelihood", "Guide"])
def test_pyrox_gp_reexports_the_core_object(name) -> None:
    assert getattr(pyrox_gp, name) is getattr(pyrox.likelihoods, name)


def test_pyrox_gp_gaussian_likelihood_subclasses_the_core_one() -> None:
    # pyrox-gp's builds a gaussx likelihood (gh-375), which the core cannot
    # import; subclassing keeps every dispatch on the core class.
    assert issubclass(pyrox_gp.GaussianLikelihood, pyrox.likelihoods.GaussianLikelihood)
