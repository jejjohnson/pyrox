"""State-space models as NumPyro models.

Temporal GPs and Gaussian state-space models on top of pyrox and gaussx, in
the package whose inference recurses in time. ``pyrox_ssm`` never imports
``pyrox_gp``, ``pyrox_lgm`` or ``pyrox_nn``.
"""

__version__ = "0.1.0"  # x-release-please-version

__all__: list[str] = []
