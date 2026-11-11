"""Normalizing-flow observation warps over a Gaussian state-space base.

Needs the ``flows`` extra (``pip install 'pyrox-ssm[flows]'``): importing this
module needs ``flowjax``, and a non-``None`` warp needs ``gauss_flows``.
"""

try:
    import flowjax  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "pyrox_ssm.flows requires the flows extra: pip install 'pyrox-ssm[flows]'"
    ) from exc

from pyrox_ssm.flows._markov_flow import (
    NormalizingKalmanPrior,
    normalizing_kalman_factor,
)


__all__ = [
    "NormalizingKalmanPrior",
    "normalizing_kalman_factor",
]
