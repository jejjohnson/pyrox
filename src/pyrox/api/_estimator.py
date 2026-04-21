"""Minimal sklearn-style immutable Estimator facade.

This is the slice of `pyrox.api` needed to build the BNF estimator
family. The full sklearn-style validation surface (multi-output
support, pipeline composition, ``GPEstimator``) is tracked separately;
what's here is the contract every estimator must satisfy:

* Subclass :class:`EstimatorBase` with ``feature_cols`` /
  ``target_col`` plus any model-specific :class:`equinox.Module`
  fields.
* Implement ``fit(self, df, *, seed) -> FittedEstimator``. Return a
  *new* fitted record; never mutate ``self``.
* Implement ``FittedEstimator.predict(self, df, *, quantiles=None)``
  returning either ``mean`` (when ``quantiles is None``) or
  ``(mean, quantile_tuple)``.

The contract is deliberately thin so unit tests can subclass it
trivially (see ``tests/api/test_estimator.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import pandas as pd
from jaxtyping import Array, Float, PRNGKeyArray


class EstimatorBase(eqx.Module):
    """Immutable configuration for a probabilistic regression model.

    Attributes:
        feature_cols: Names of the input columns, in the order they
            should be packed into the design matrix.
        target_col: Name of the target column.
    """

    feature_cols: tuple[str, ...] = eqx.field(static=True)
    target_col: str = eqx.field(static=True)

    def fit(self, df: pd.DataFrame, *, seed: PRNGKeyArray | int) -> FittedEstimator:
        """Fit and return a new :class:`FittedEstimator`.

        Subclasses override; ``EstimatorBase.fit`` raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.fit must be implemented by subclasses"
        )


class FittedEstimator(eqx.Module):
    """Output of :meth:`EstimatorBase.fit`.

    Subclasses store fitted parameters as :class:`equinox.Module`
    fields and implement ``predict``. The base class enforces only
    the immutable-PyTree contract.
    """

    config: EstimatorBase

    def predict(
        self,
        df: pd.DataFrame,
        *,
        quantiles: Sequence[float] | None = None,
    ) -> Float[Array, " N"] | tuple[Float[Array, " N"], tuple[Float[Array, " N"], ...]]:
        """Predict targets for ``df``.

        Subclasses override; ``FittedEstimator.predict`` raises.

        Args:
            df: Inputs (must contain ``feature_cols``).
            quantiles: If given, return ``(mean, (q1, q2, ...))`` where
                ``q_i`` is the per-row predictive quantile at level
                ``quantiles[i]``. If ``None``, return only the mean.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.predict must be implemented by subclasses"
        )


# `Any` is not used internally but kept available for downstream
# subclasses that may need it.
_ = Any
