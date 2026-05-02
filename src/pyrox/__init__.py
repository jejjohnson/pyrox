"""pyrox: probabilistic modeling with Equinox and NumPyro."""

from pyrox import _core, gp, inference, nn


__version__ = "0.0.12"
__all__ = [
    "__version__",
    "_core",
    "gp",
    "inference",
    "nn",
]

# `pyrox.api` and `pyrox.preprocessing` pull in pandas, which lives in the
# `[bnf]` optional extra. Only expose them when the extra is installed so
# a plain `pip install pyrox` + `import pyrox` still works.
try:
    from pyrox import api, preprocessing

    __all__ += ["api", "preprocessing"]
except ImportError:
    pass
