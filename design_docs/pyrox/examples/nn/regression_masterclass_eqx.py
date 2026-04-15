"""
Regression Masterclass — Equinox + NumPyro + einops Edition
=============================================================

A pedagogical reimplementation of the Aboleth "Regression Master Class"
using **Equinox** for deterministic architecture (modules, layers, feature
maps), **NumPyro** for probabilistic semantics (priors, likelihoods,
inference), and **einops** for all tensor contractions (replacing ``@``
and manual broadcasting with readable ``einsum`` patterns).

Three Libraries, Three Roles
-----------------------------
This module demonstrates a clean separation of concerns across three
JAX-based libraries, each owning a distinct responsibility:

    ┌────────────────────────────┐
    │        Equinox             │  Defines architecture (frozen PyTree modules)
    ├────────────────────────────┤
    │        einops              │  All matrix ops as named-axis einsum patterns
    ├────────────────────────────┤
    │        NumPyro             │  Probabilistic semantics & inference
    └────────────────────────────┘

**Equinox** provides *deterministic* neural-network modules as immutable
JAX PyTrees.  Each module (``LinearRegressor``, ``MLP``, ``RFFFeatureMap``,
etc.) defines the forward-pass *architecture* — layer shapes, activations,
feature maps — but its weight arrays are merely placeholders.  Because
Equinox modules are frozen PyTrees, they cannot be mutated in place.

**NumPyro** owns the *probabilistic semantics*: it declares priors over
every learnable parameter via ``numpyro.sample``, defines the likelihood
via ``numpyro.sample("obs", ..., obs=y)``, records deterministic
transformations via ``numpyro.deterministic``, and runs inference (MCMC
or SVI).  NumPyro never knows anything about the network architecture —
it only ever sees flat sample sites.

**einops** replaces all ``@`` matrix multiplications and manual broadcast
hacks with explicit, named-axis ``einsum`` patterns such as::

    einsum(A, B, "n feat, feat -> n")

This makes every contraction self-documenting: the reader can see which
axes are batch dimensions (``n``), which are being summed out (``feat``),
and which survive into the output.

The ``eqx.tree_at`` Bridge Pattern
------------------------------------
The key design pattern that connects Equinox and NumPyro is
``eqx.tree_at``.  The call signature is::

    new_module = eqx.tree_at(selector_fn, old_module, replacement_values)

where:

- **selector_fn** (``lambda m: m.weight`` or ``lambda m: (m.W1, m.b1)``)
  is a function that, given the module PyTree, returns the leaf (or tuple
  of leaves) to replace.  Equinox traces this function to identify the
  target leaves by object identity.
- **old_module** is the existing (placeholder-weight) Equinox module.
- **replacement_values** are the fresh arrays drawn from NumPyro priors.

The return value is a *new* module instance (since PyTrees are immutable)
with exactly those leaves swapped out.  This lets us:

1. Define the architecture once in pure Equinox (``MLP.__init__`` sets
   shapes; ``MLP.__call__`` defines the forward pass).
2. In the NumPyro model function, sample fresh weights from priors and
   inject them into the frozen architecture with ``eqx.tree_at``.
3. Call the now-parameterised module to compute the latent function ``f``.

This pattern cleanly avoids mixing probabilistic logic into the module
class and avoids duplicating the forward-pass logic in the model function.

Latent function
---------------
    f(x) = sin(x) / x   (implemented via ``jnp.sinc``)

Models
------
1. ``LinearRegressor``         — Bayesian linear / polynomial regression
2. ``MLP``                     — Single-hidden-layer neural network
3. ``MLPDropout``              — MLP with stochastic dropout (MC-Dropout)
4. ``RFFFeatureMap``           — Random Fourier Feature map (Rahimi & Recht 2007)
5. ``RFFRegressor``            — Linear regression in RFF space (approx. SVR / GP)
6. ``DeepRFFRegressor``        — Two-layer RFF (approx. Deep GP, Cutajar et al. 2017)
7. ``NCPNetwork``              — Noise Contrastive Prior network (Hafner et al. 2018)
8. ``RBFKernel``               — Parameterized GP kernel with learned hyperparameters

Each Equinox module has a corresponding ``model_*`` function that wraps it
in NumPyro's probabilistic semantics.

References
----------
.. [1] Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale
       Kernel Machines." NeurIPS 2007.
.. [2] Cutajar, K., Bonilla, E. V., Michiardi, P. & Filippone, M. (2017).
       "Random Feature Expansions for Deep Gaussian Processes." ICML 2017.
.. [3] Hafner, D., Tran, D., Irpan, A. & Lillicrap, T. (2018). "Noise
       Contrastive Priors for Functional Uncertainty." arXiv:1807.09289.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from einops import einsum, rearrange
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide


# ============================================================================
# 1.  Data generation
# ============================================================================


def latent_function(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the true latent (noise-free) function: f(x) = sin(x) / x.

    Uses ``jnp.sinc(x / pi)`` because NumPy's sinc is defined as
    ``sin(pi * x) / (pi * x)``, so ``sinc(x / pi) = sin(x) / x``.

    Parameters
    ----------
    x : jnp.ndarray, shape (N,)
        Input locations.

    Returns
    -------
    f : jnp.ndarray, shape (N,)
        Function values f(x) = sin(x) / x evaluated element-wise.
    """
    return jnp.sinc(x / jnp.pi)


def make_dataset(
    key: jax.Array,
    n_train: int = 100,
    n_test: int = 400,
    noise_std: float = 0.05,
    train_bounds: tuple[float, float] = (-10.0, 10.0),
    pred_bounds: tuple[float, float] = (-20.0, 20.0),
) -> dict:
    """Generate the synthetic regression dataset.

    Draws training inputs uniformly at random from ``train_bounds``,
    evaluates the latent function f(x) = sin(x)/x, and adds i.i.d.
    Gaussian noise.  Test inputs are a deterministic linspace over
    ``pred_bounds`` (wider than training, to test extrapolation).

    Parameters
    ----------
    key : jax.Array
        PRNG key for reproducibility.
    n_train : int
        Number of training points.  Default 100.
    n_test : int
        Number of test (prediction) points.  Default 400.
    noise_std : float
        Standard deviation of additive Gaussian observation noise.
    train_bounds : tuple[float, float]
        (lo, hi) for uniform training inputs.
    pred_bounds : tuple[float, float]
        (lo, hi) for the dense test grid.

    Returns
    -------
    dict with keys:
        - ``x_train`` : shape (n_train,)  — training inputs
        - ``y_train`` : shape (n_train,)  — noisy training targets
        - ``x_test``  : shape (n_test,)   — test inputs (linspace)
        - ``y_test``  : shape (n_test,)   — noise-free test targets
        - ``noise_std`` : float           — the noise std used
    """
    k1, k2 = jr.split(key)
    # x_train: (n_train,) — uniform random in [lo, hi]
    x_train = jr.uniform(k1, (n_train,), minval=train_bounds[0], maxval=train_bounds[1])
    # y_train: (n_train,) — latent function + Gaussian noise
    y_train = latent_function(x_train) + noise_std * jr.normal(k2, (n_train,))
    # x_test: (n_test,) — evenly-spaced grid for smooth predictions
    x_test = jnp.linspace(*pred_bounds, n_test)
    # y_test: (n_test,) — noise-free ground truth
    y_test = latent_function(x_test)
    return dict(
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        noise_std=noise_std,
    )


# ============================================================================
# 2.  Equinox modules  (deterministic architecture)
# ============================================================================
#
# Each class below is a frozen JAX PyTree (via ``eqx.Module``).  Its
# ``__init__`` sets up shapes and placeholder values; its ``__call__``
# defines the deterministic forward pass.
#
# Crucially, the weight arrays stored in these modules are *not* the
# values used during inference.  Instead, the companion ``model_*``
# function will:
#   1. Sample fresh weights from NumPyro priors.
#   2. Use ``eqx.tree_at(selector, module, sampled_weights)`` to produce
#      a *new* module with those weights injected.
#   3. Call the new module's ``__call__`` to compute the forward pass.
#
# This keeps all architecture logic in Equinox and all probabilistic
# logic in NumPyro, with ``eqx.tree_at`` as the thin bridge.
# ============================================================================


class LinearRegressor(eqx.Module):
    """Bayesian-ready linear model in a polynomial feature space.

    Mathematical model (forward pass)
    ----------------------------------
    Given input vector x of shape (N,), the model constructs a Vandermonde
    feature matrix and computes a linear prediction:

        Phi(x) = [1, x, x^2, ..., x^d]       # shape: (N, degree+1)
        f(x)   = Phi(x) @ w                   # shape: (N,)

    where d = ``degree`` and w is the weight vector of shape (degree+1,).

    The ``einsum`` pattern ``"n feat, feat -> n"`` contracts over the
    ``feat`` axis (the polynomial feature dimension), leaving the batch
    axis ``n``.  This is equivalent to ``Phi @ w`` but self-documenting.

    Attributes
    ----------
    weight : jax.Array, shape (degree + 1,)
        Weight vector in the polynomial feature space.  Initialised to
        zeros as a shape placeholder — the actual values are injected
        by the NumPyro model function via ``eqx.tree_at``.
    degree : int  (static, not a JAX array)
        Polynomial degree.  ``degree=1`` gives ordinary linear regression;
        ``degree=3`` gives cubic, etc.

    eqx.tree_at usage (in ``model_linear``)
    ----------------------------------------
    The companion model function replaces the weight vector like so::

        w = numpyro.sample("w", dist.Normal(zeros(degree+1), 1.0))
        net = eqx.tree_at(lambda m: m.weight, net, w)

    Here:
    - **selector** ``lambda m: m.weight`` identifies the single leaf
      (the weight array) to replace.
    - **pytree** ``net`` is the existing LinearRegressor instance.
    - **replacement** ``w`` is the freshly sampled weight vector.
    - The return value is a new LinearRegressor with ``w`` in place of
      the old zeros, while ``degree`` remains unchanged.
    """
    weight: jax.Array
    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 1):
        self.degree = degree
        # Placeholder weights — shape (degree + 1,), all zeros.
        # These will be replaced by NumPyro samples via eqx.tree_at.
        self.weight = jnp.zeros(degree + 1)  # weight: (degree + 1,)

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build the polynomial (Vandermonde) feature matrix.

        Parameters
        ----------
        x : shape (N,)
            Raw scalar inputs.

        Returns
        -------
        Phi : shape (N, degree + 1)
            Feature matrix where column p is x^p, for p = 0, 1, ..., degree.
            Column 0 is all ones (the bias / intercept term).
        """
        # Each x**p is (N,); stacking along axis=-1 gives (N, degree+1)
        return jnp.stack([x ** p for p in range(self.degree + 1)], axis=-1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: f(x) = Phi(x) @ w.

        Parameters
        ----------
        x : shape (N,)
            Input locations.

        Returns
        -------
        f : shape (N,)
            Predictions.  The einsum pattern ``"n feat, feat -> n"``
            contracts over ``feat`` (the polynomial feature dimension),
            producing one scalar prediction per data point.
        """
        # Phi: (N, degree+1)   weight: (degree+1,)
        # einsum contracts over "feat" (the degree+1 dimension)
        # result: (N,) — one prediction per input
        return einsum(self.features(x), self.weight, "n feat, feat -> n")


class MLP(eqx.Module):
    """Single-hidden-layer multi-layer perceptron with tanh activation.

    Mathematical model (forward pass)
    ----------------------------------
    Given scalar inputs x of shape (N,):

        h  = tanh(x @ W1 + b1)     # hidden activations, shape (N, H)
        f  = h @ W2 + b2           # output predictions,  shape (N,)

    where H = ``hidden_dim`` is the hidden-layer width.

    In detail, using einsum patterns:

    1. ``einsum(x, W1, "n, one h -> n h")``
       - x is (N,) and W1 is (1, H).
       - The ``one`` axis (size 1) broadcasts with ``n``, so this is
         effectively an outer-product-like broadcast: each scalar x_n
         multiplies every column of W1.
       - Contraction: the implicit ``one`` dimension is contracted
         (it has size 1, so this is really a broadcast multiply).
       - Result: (N, H) — pre-activation for each hidden unit.

    2. ``einsum(h, W2, "n h, h one -> n")``
       - h is (N, H) and W2 is (H, 1).
       - Contraction over ``h``: sums across hidden units.
       - The ``one`` dimension (size 1) is contracted away.
       - Result: (N,) — one scalar prediction per data point.

    Attributes
    ----------
    W1 : jax.Array, shape (1, hidden_dim)
        Input-to-hidden weight matrix.
    b1 : jax.Array, shape (hidden_dim,)
        Hidden layer bias.
    W2 : jax.Array, shape (hidden_dim, 1)
        Hidden-to-output weight matrix.
    b2 : jax.Array, shape ()  (scalar)
        Output bias.
    hidden_dim : int  (static)
        Number of hidden units.

    eqx.tree_at usage (in ``model_nnet``)
    ----------------------------------------
    All four weight arrays are replaced simultaneously::

        net = eqx.tree_at(
            lambda m: (m.W1, m.b1, m.W2, m.b2),  # selector: tuple of 4 leaves
            net,                                    # pytree: the MLP instance
            (W1, b1, W2, b2),                       # replacement: tuple of 4 arrays
        )

    The selector returns a *tuple* of leaves, so the replacement must
    also be a tuple of the same structure.  Equinox matches them
    positionally: m.W1 <-> W1, m.b1 <-> b1, etc.
    """
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int = 50, *, key: jax.Array):
        """Initialise with small random weights (for shape definition).

        The specific values do not matter because NumPyro will replace
        them with prior samples via ``eqx.tree_at``.  We use small
        random values rather than zeros so that if someone accidentally
        calls the module without injecting samples, the output is not
        trivially zero everywhere.

        Parameters
        ----------
        hidden_dim : int
            Width of the hidden layer.
        key : jax.Array
            PRNG key for weight initialisation.
        """
        k1, k2 = jr.split(key)
        self.hidden_dim = hidden_dim
        # Shape-defining initialisations (values will be overwritten by NumPyro)
        self.W1 = jr.normal(k1, (1, hidden_dim)) * 0.1   # W1: (1, H)
        self.b1 = jnp.zeros(hidden_dim)                   # b1: (H,)
        self.W2 = jr.normal(k2, (hidden_dim, 1)) * 0.1   # W2: (H, 1)
        self.b2 = jnp.array(0.0)                          # b2: scalar ()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the single-hidden-layer network.

        Parameters
        ----------
        x : shape (N,)
            Scalar inputs (one per data point).

        Returns
        -------
        f : shape (N,)
            Network output (one scalar per data point).

        Einsum annotation
        -----------------
        Step 1: ``einsum(x, W1, "n, one h -> n h")``
            - x: (N,), W1: (1, H)
            - "n" is the batch dim, "one" is size 1, "h" is hidden dim
            - Contracts/broadcasts over "one", producing (N, H)
            - This is x[:, None] * W1, i.e. outer-product broadcast

        Step 2: ``einsum(h, W2, "n h, h one -> n")``
            - h: (N, H), W2: (H, 1)
            - "h" is contracted (summed over hidden units)
            - "one" (size 1) disappears, yielding (N,)
            - This is the dot product of each row of h with W2's column
        """
        # x: (N,)  W1: (1, H)
        # einsum "n, one h -> n h": broadcast x across H hidden units
        # result: (N, H) pre-activations
        h = jnp.tanh(
            einsum(x, self.W1, "n, one h -> n h") + self.b1  # + b1: (H,) broadcasts
        )                                                # h: (N, H)
        # einsum "n h, h one -> n": contract over h (hidden dim), squeeze "one"
        # result: (N,) output predictions
        return einsum(h, self.W2, "n h, h one -> n") + self.b2  # f: (N,)


class MLPDropout(eqx.Module):
    """MLP with an explicit dropout mask applied to the hidden layer.

    Mathematical model (forward pass)
    ----------------------------------
    Given scalar inputs x of shape (N,) and binary mask of shape (N, H):

        h_raw  = tanh(x @ W1 + b1)                  # shape: (N, H)
        h      = h_raw * mask / (1 - dropout_rate)   # inverted dropout, (N, H)
        f      = h @ W2 + b2                         # shape: (N,)

    The inverted dropout scaling ``/ (1 - p)`` ensures that the expected
    value of each hidden unit is the same at train and test time, so no
    rescaling is needed at prediction time.

    Unlike standard dropout implementations that draw the mask internally
    using an RNG key, here the dropout mask is an *explicit argument* to
    ``__call__``.  This design lets the NumPyro model function sample the
    mask as a ``numpyro.sample`` site::

        mask = numpyro.sample("dropout_mask",
                              dist.Bernoulli(probs=(1 - p) * ones(N, H)))

    Because the mask is a first-class NumPyro sample site, calling
    ``Predictive`` automatically draws a fresh mask for each posterior
    sample, giving MC-Dropout uncertainty "for free" without any special
    predict-time logic.

    Attributes
    ----------
    W1 : jax.Array, shape (1, hidden_dim)
        Input-to-hidden weight matrix.
    b1 : jax.Array, shape (hidden_dim,)
        Hidden layer bias.
    W2 : jax.Array, shape (hidden_dim, 1)
        Hidden-to-output weight matrix.
    b2 : jax.Array, shape ()  (scalar)
        Output bias.
    hidden_dim : int  (static)
        Number of hidden units.
    dropout_rate : float  (static)
        Probability of dropping each hidden unit (0 = no dropout).

    eqx.tree_at usage (in ``model_nnet_dropout``)
    -----------------------------------------------
    Same tuple-of-leaves pattern as MLP::

        net = eqx.tree_at(
            lambda m: (m.W1, m.b1, m.W2, m.b2),
            net,
            (W1, b1, W2, b2),
        )
    """
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    hidden_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, hidden_dim: int = 50, dropout_rate: float = 0.1, *, key: jax.Array):
        """Initialise with placeholder weights.

        Parameters
        ----------
        hidden_dim : int
            Width of the hidden layer.
        dropout_rate : float
            Probability of zeroing each hidden unit (between 0 and 1).
        key : jax.Array
            PRNG key for shape-defining weight initialisation.
        """
        k1, k2 = jr.split(key)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.W1 = jr.normal(k1, (1, hidden_dim)) * 0.1   # W1: (1, H)
        self.b1 = jnp.zeros(hidden_dim)                   # b1: (H,)
        self.W2 = jr.normal(k2, (hidden_dim, 1)) * 0.1   # W2: (H, 1)
        self.b2 = jnp.array(0.0)                          # b2: scalar ()

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with an explicit dropout mask.

        Parameters
        ----------
        x : shape (N,)
            Scalar inputs (one per data point).
        mask : shape (N, hidden_dim)
            Binary dropout mask.  1 = keep the unit, 0 = drop it.
            Typically sampled as ``Bernoulli(1 - dropout_rate)``.

        Returns
        -------
        f : shape (N,)
            Network output after applying masked (dropped-out) hidden layer.

        Einsum annotation
        -----------------
        Same two-step pattern as ``MLP.__call__``; see that docstring
        for dimension-by-dimension explanation of the einsum patterns.
        The only difference is the element-wise mask application between
        the two einsum steps.
        """
        # x: (N,)  W1: (1, H)
        # einsum "n, one h -> n h": broadcast multiply, result (N, H)
        h = jnp.tanh(
            einsum(x, self.W1, "n, one h -> n h") + self.b1  # + b1: (H,) broadcasts
        )                                                      # h: (N, H)
        # Inverted dropout: zero out dropped units, scale up survivors
        # mask: (N, H), element-wise multiply, then divide by keep probability
        h = h * mask / (1.0 - self.dropout_rate)               # h: (N, H)
        # einsum "n h, h one -> n": contract over h, squeeze "one" → (N,)
        return einsum(h, self.W2, "n h, h one -> n") + self.b2  # f: (N,)


class RFFFeatureMap(eqx.Module):
    """Random Fourier Feature (RFF) map approximating an RBF kernel.

    Reference: Rahimi, A. & Recht, B. (2007). "Random Features for
    Large-Scale Kernel Machines." NeurIPS 2007.

    Mathematical background
    -----------------------
    The RBF (squared-exponential) kernel with lengthscale l is:

        k(x, x') = exp(-||x - x'||^2 / (2 * l^2))

    By Bochner's theorem, any shift-invariant kernel can be written as
    the Fourier transform of a non-negative measure.  For the RBF kernel,
    this spectral measure is a Gaussian:  p(omega) = N(0, 1/l^2).

    The key insight of Rahimi & Recht (2007) is that we can approximate
    the kernel by drawing D random frequencies from the spectral measure
    and constructing the feature map:

        phi(x) = sqrt(2/D) * cos(omega * x + b)

    where omega ~ N(0, 1/l^2) and b ~ Uniform(0, 2*pi) are drawn once
    at construction time and held fixed.  Then:

        phi(x)^T phi(x')  approx  k(x, x')

    with the approximation improving as D -> infinity.

    Forward pass
    ------------
    Given x of shape (N,):

        projection = einsum(x, omega, "n, d -> n d")   # outer product, (N, D)
        phi(x) = sqrt(2/D) * cos(projection + b)       # (N, D)

    The ``einsum`` pattern ``"n, d -> n d"`` computes the outer product
    of x (batch dim ``n``) and omega (feature dim ``d``).  No axes are
    contracted — both ``n`` and ``d`` appear in the output.  This gives
    the (N, D) matrix of projections omega_d * x_n.

    Attributes
    ----------
    omega : jax.Array, shape (D,)
        Random frequencies drawn from N(0, 1/l^2).  Fixed after construction.
    bias : jax.Array, shape (D,)
        Random phase offsets drawn from Uniform(0, 2*pi).  Fixed after
        construction.
    n_features : int  (static)
        Number of random features D.

    Note: This module has NO learnable parameters.  The omega and bias
    arrays are random but fixed — they are never replaced by NumPyro
    samples.  Only the downstream linear weights (in ``RFFRegressor``
    or in the model function) are treated as probabilistic.
    """
    omega: jax.Array
    bias: jax.Array
    n_features: int = eqx.field(static=True)

    def __init__(self, n_features: int, lengthscale: float = 1.0, *, key: jax.Array):
        """Draw the random frequencies and phases that define this feature map.

        Parameters
        ----------
        n_features : int
            Number of random Fourier features D.  Larger D gives a better
            kernel approximation but costs more computation.
        lengthscale : float
            RBF kernel lengthscale l.  Frequencies are drawn from
            N(0, 1/l^2), so smaller l means higher-frequency features.
        key : jax.Array
            PRNG key for drawing omega and bias.
        """
        k1, k2 = jr.split(key)
        self.n_features = n_features
        # omega: (D,) — spectral frequencies from N(0, 1/l^2)
        # Dividing N(0,1) samples by lengthscale gives N(0, 1/l^2)
        self.omega = jr.normal(k1, (n_features,)) / lengthscale  # omega: (D,)
        # bias: (D,) — random phases, uniform in [0, 2*pi]
        self.bias = jr.uniform(k2, (n_features,), minval=0.0, maxval=2.0 * jnp.pi)  # bias: (D,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the random Fourier feature map.

        Parameters
        ----------
        x : shape (N,)
            Scalar inputs.

        Returns
        -------
        Phi : shape (N, D)
            The D-dimensional RFF embedding of each input.  Rows
            approximate the kernel: Phi[i] @ Phi[j] ≈ k(x[i], x[j]).
        """
        D = self.n_features
        # einsum "n, d -> n d": outer product (no contraction).
        # "n" is the batch (data) axis, "d" is the feature axis.
        # Both appear in the output, so no summation occurs.
        # Result: (N, D) where entry [i, j] = x[i] * omega[j]
        projection = einsum(x, self.omega, "n, d -> n d")  # projection: (N, D)
        # cos(projection + bias): (N, D), element-wise cosine
        # sqrt(2/D) normalisation ensures phi^T phi ≈ k in expectation
        return jnp.sqrt(2.0 / D) * jnp.cos(projection + self.bias)  # Phi: (N, D)


class RFFRegressor(eqx.Module):
    """Linear model in Random Fourier Feature space (approx. kernel method).

    Mathematical model (forward pass)
    ----------------------------------
    Composes an ``RFFFeatureMap`` with a linear weight vector:

        Phi = phi(x)       # shape (N, D) — RFF feature map (fixed)
        f   = Phi @ w      # shape (N,)   — linear prediction

    This is equivalent to kernel ridge regression / SVR with an RBF
    kernel, where the kernel matrix is approximated by Phi @ Phi^T.

    The ``einsum`` pattern ``"n d, d -> n"`` contracts over ``d`` (the
    RFF feature dimension), computing the dot product of each row of Phi
    with the weight vector w.

    Attributes
    ----------
    rff : RFFFeatureMap
        The (fixed) random feature map.  Its omega and bias are drawn
        once at construction and never changed.
    weight : jax.Array, shape (D,)
        Linear weights in the RFF feature space.  Initialised to zeros
        as a placeholder — replaced by NumPyro samples during inference.

    eqx.tree_at usage (in ``model_svr_rff``)
    ------------------------------------------
    Note: For this model, ``eqx.tree_at`` is NOT used.  Instead, the
    model function calls the RFF module directly (it has no learnable
    params) and samples only the linear weight vector ``w`` and noise
    ``sigma`` as NumPyro sites.  The linear prediction is computed
    inline with ``einsum(Phi, w, "n d, d -> n")``.
    """
    rff: RFFFeatureMap
    weight: jax.Array

    def __init__(self, n_features: int, lengthscale: float = 1.0, *, key: jax.Array):
        """Construct the RFF feature map and allocate placeholder weights.

        Parameters
        ----------
        n_features : int
            Number of random Fourier features D.
        lengthscale : float
            RBF kernel lengthscale.
        key : jax.Array
            PRNG key (split internally for the feature map).
        """
        k1, k2 = jr.split(key)
        self.rff = RFFFeatureMap(n_features, lengthscale, key=k1)
        self.weight = jnp.zeros(n_features)  # weight: (D,) placeholder

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: f(x) = phi(x) @ w.

        Parameters
        ----------
        x : shape (N,)
            Input locations.

        Returns
        -------
        f : shape (N,)
            Predictions.  einsum ``"n d, d -> n"`` contracts over ``d``
            (the D random features), yielding one scalar per data point.
        """
        # rff(x): (N, D)   weight: (D,)
        # einsum "n d, d -> n": contract over d (RFF feature dim) → (N,)
        return einsum(self.rff(x), self.weight, "n d, d -> n")  # f: (N,)


class DeepRFFRegressor(eqx.Module):
    """Two-layer RFF regressor approximating a Deep Gaussian Process.

    Reference: Cutajar, K., Bonilla, E. V., Michiardi, P. & Filippone, M.
    (2017). "Random Feature Expansions for Deep Gaussian Processes." ICML 2017.

    Mathematical model (forward pass)
    ----------------------------------
    The architecture stacks two RFF layers with a learned linear map
    between them, mimicking a two-layer Deep GP:

    **Layer 1** (input → hidden):
        Phi1 = rff1(x)                   # shape: (N, D1)
        h    = Phi1 @ W1                 # shape: (N, inner_dim)

    **Layer 2** (hidden → output):
        For each hidden dimension j = 0, ..., inner_dim-1:
            Phi2_j = rff2(h[:, j])       # shape: (N, D2)
        Phi2 = mean_j(Phi2_j)           # shape: (N, D2)  — average across hidden dims
        f    = Phi2 @ w2                 # shape: (N,)

    The averaging across hidden dimensions in Layer 2 is a design choice
    that reduces the parameter count (w2 is (D2,) not (inner_dim * D2,))
    while still allowing information from all hidden dimensions to flow
    through.

    Einsum patterns
    ---------------
    1. ``einsum(Phi1, W1, "n d1, d1 inner -> n inner")``
       - Phi1: (N, D1), W1: (D1, inner_dim)
       - Contracts over ``d1`` (the layer-1 RFF features)
       - Result: (N, inner_dim) — the hidden representation

    2. ``einsum(Phi2, w2, "n d2, d2 -> n")``
       - Phi2: (N, D2), w2: (D2,)
       - Contracts over ``d2`` (the layer-2 RFF features)
       - Result: (N,) — final scalar predictions

    Attributes
    ----------
    rff1 : RFFFeatureMap
        Layer-1 random feature map with D1 features.  Fixed after construction.
    W1 : jax.Array, shape (D1, inner_dim)
        Layer-1 linear weights (input RFF space → hidden representation).
        Placeholder — replaced by NumPyro samples.
    rff2 : RFFFeatureMap
        Layer-2 random feature map with D2 features.  Fixed after construction.
    w2 : jax.Array, shape (D2,)
        Layer-2 linear weights (hidden RFF space → output).
        Placeholder — replaced by NumPyro samples.
    inner_dim : int  (static)
        Dimensionality of the hidden representation between layers.

    eqx.tree_at usage (in ``model_deep_gp_rff``)
    -----------------------------------------------
    Only the learnable weight arrays are replaced; the RFF maps stay fixed::

        mod = eqx.tree_at(
            lambda m: (m.W1, m.w2),   # selector: the two weight arrays
            mod,                       # pytree: the DeepRFFRegressor
            (W1, w2),                  # replacement: sampled arrays
        )

    The RFF maps (``mod.rff1``, ``mod.rff2``) are not selected by the
    lambda, so they pass through unchanged — their frozen random
    frequencies and phases remain intact.
    """
    rff1: RFFFeatureMap
    W1: jax.Array
    rff2: RFFFeatureMap
    w2: jax.Array
    inner_dim: int = eqx.field(static=True)

    def __init__(
        self,
        n_features_1: int = 20,
        n_features_2: int = 10,
        inner_dim: int = 5,
        lengthscale_1: float = 1.0,
        lengthscale_2: float = 1.0,
        *,
        key: jax.Array,
    ):
        """Construct both RFF maps and allocate placeholder weights.

        Parameters
        ----------
        n_features_1 : int
            Number of random features D1 for the first layer.
        n_features_2 : int
            Number of random features D2 for the second layer.
        inner_dim : int
            Width of the intermediate hidden representation.
        lengthscale_1 : float
            RBF lengthscale for the first RFF layer.
        lengthscale_2 : float
            RBF lengthscale for the second RFF layer.
        key : jax.Array
            PRNG key (split into 4 sub-keys internally).
        """
        k1, k2, k3, k4 = jr.split(key, 4)
        self.inner_dim = inner_dim
        self.rff1 = RFFFeatureMap(n_features_1, lengthscale_1, key=k1)
        # W1: (D1, inner_dim) — small random init as shape placeholder
        self.W1 = jr.normal(k2, (n_features_1, inner_dim)) * 0.1  # W1: (D1, inner_dim)
        self.rff2 = RFFFeatureMap(n_features_2, lengthscale_2, key=k3)
        # w2: (D2,) — zeros as shape placeholder
        self.w2 = jnp.zeros(n_features_2)  # w2: (D2,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the two-layer Deep RFF architecture.

        Parameters
        ----------
        x : shape (N,)
            Scalar inputs.

        Returns
        -------
        f : shape (N,)
            Output predictions from the two-layer RFF pipeline.
        """
        # ---- Layer 1: x → hidden representation h ----
        Phi1 = self.rff1(x)                                        # Phi1: (N, D1)
        # einsum "n d1, d1 inner -> n inner":
        #   contracts over d1 (layer-1 RFF features)
        #   maps from D1-dimensional RFF space to inner_dim-dimensional hidden space
        h = einsum(Phi1, self.W1, "n d1, d1 inner -> n inner")     # h: (N, inner_dim)

        # ---- Layer 2: h → output f via averaged RFF features ----
        # Apply the second RFF map independently to each hidden dimension,
        # then average.  For each j in 0..inner_dim-1:
        #   rff2(h[:, j]) is (N, D2) — the layer-2 features for hidden dim j
        # Stack gives (inner_dim, N, D2); mean over axis=0 gives (N, D2)
        Phi2 = jnp.mean(
            jnp.stack([self.rff2(h[:, j]) for j in range(self.inner_dim)], axis=0),
            axis=0,
        )                                                           # Phi2: (N, D2)
        # einsum "n d2, d2 -> n":
        #   contracts over d2 (layer-2 RFF features)
        #   yields one scalar prediction per data point
        return einsum(Phi2, self.w2, "n d2, d2 -> n")              # f: (N,)


# ============================================================================
# 3.  NumPyro model functions
# ============================================================================
#
# Each ``model_*`` function follows a five-step recipe:
#
#   1. **Instantiate** the Equinox module with placeholder weights.
#      This defines the architecture (layer shapes, activations, etc.)
#      without committing to specific parameter values.
#
#   2. **Sample** parameters from priors using ``numpyro.sample(name, dist)``.
#      Each ``numpyro.sample`` call declares a named random variable in the
#      probabilistic model.  During MCMC, NumPyro will infer the posterior
#      distribution over these variables.  During SVI, the guide will
#      approximate the posterior.
#
#   3. **Inject** sampled parameters into the Equinox module using
#      ``eqx.tree_at(selector, module, replacement)``.  This produces a
#      *new* module instance (Equinox modules are immutable PyTrees) with
#      the sampled values in place of the placeholders.  This is the
#      bridge between the deterministic (Equinox) and probabilistic
#      (NumPyro) worlds.
#
#   4. **Compute** the latent function f by calling the parameterised
#      module, and register it with ``numpyro.deterministic("f", ...)``.
#      This tells NumPyro to record f in the trace so it appears in
#      posterior samples / predictions, even though it is a deterministic
#      function of the sampled parameters (not a random variable itself).
#
#   5. **Define the likelihood** ``y ~ Normal(f, sigma)`` via
#      ``numpyro.sample("obs", dist.Normal(f, sigma), obs=y)``.
#      The ``obs=y`` keyword conditions the model on the observed data:
#      during inference, NumPyro uses this to compute the log-likelihood
#      contribution.  During prediction (obs=None), it generates
#      synthetic observations.
# ============================================================================


def model_linear(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    degree: int = 1,
) -> None:
    """Bayesian linear/polynomial regression via Equinox + NumPyro.

    Generative model (plate notation)
    ----------------------------------
        w     ~ Normal(0, I)           shape: (degree + 1,)
        sigma ~ HalfNormal(1)          shape: ()
        Phi   = [1, x, x^2, ..., x^d] shape: (N, degree + 1)
        f     = Phi @ w                shape: (N,)
        y_n   ~ Normal(f_n, sigma)     for n = 1, ..., N

    eqx.tree_at bridge
    -------------------
    The ``LinearRegressor`` module stores a placeholder ``weight`` array
    of shape (degree+1,).  We replace it with the NumPyro sample ``w``::

        net = eqx.tree_at(lambda m: m.weight, net, w)

    - **selector**: ``lambda m: m.weight`` — identifies the weight leaf.
    - **pytree**: ``net`` — the LinearRegressor with zero weights.
    - **replacement**: ``w`` — the (degree+1,) array sampled from the prior.
    - **result**: a new LinearRegressor whose ``weight`` attribute is ``w``.

    NumPyro sample sites
    ---------------------
    - ``"w"``     : the polynomial weight vector.  Shape (degree + 1,).
    - ``"sigma"`` : observation noise standard deviation.  Scalar.
    - ``"f"``     : (deterministic) the latent function values.  Shape (N,).
    - ``"obs"``   : observed targets, conditioned on ``y``.  Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets.  Pass None for prior predictive / posterior
        predictive (generates synthetic observations).
    degree : int
        Polynomial degree (1 = linear, 2 = quadratic, etc.).
    """
    # --- Architecture (shape only) ---
    # Instantiate a LinearRegressor with zero weights; degree determines
    # the number of polynomial features (degree + 1 including intercept).
    net = LinearRegressor(degree=degree)

    # --- Sample weights from prior ---
    # w: (degree+1,) — one weight per polynomial basis function [1, x, ..., x^d]
    w = numpyro.sample("w", dist.Normal(jnp.zeros(degree + 1), 1.0))

    # --- Replace placeholder weights with sampled values ---
    # eqx.tree_at(selector, pytree, replacement) returns a *new* module
    # with the selected leaf replaced.
    # selector: lambda m: m.weight — points to the weight array
    # pytree:   net — the LinearRegressor instance
    # replacement: w — the freshly sampled weight vector
    net = eqx.tree_at(lambda m: m.weight, net, w)

    # --- Forward pass ---
    # net(x) calls LinearRegressor.__call__, which computes Phi(x) @ w
    # numpyro.deterministic records f in the trace for later retrieval
    # f: (N,) — predicted function values at each input
    f = numpyro.deterministic("f", net(x))

    # --- Likelihood ---
    # sigma: () — scalar noise std, constrained to be positive via HalfNormal
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    # obs: (N,) — when obs=y, this computes log p(y | f, sigma) for inference;
    # when obs=None (prediction), this generates y ~ Normal(f, sigma)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


def model_nnet(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    hidden_dim: int = 50,
) -> None:
    """Single-hidden-layer Bayesian neural network: Equinox MLP + NumPyro priors.

    Generative model
    ----------------
        W1    ~ Normal(0, I)    shape: (1, hidden_dim)
        b1    ~ Normal(0, I)    shape: (hidden_dim,)
        W2    ~ Normal(0, I)    shape: (hidden_dim, 1)
        b2    ~ Normal(0, 1)    shape: ()
        sigma ~ HalfNormal(1)   shape: ()

        h  = tanh(x @ W1 + b1)  shape: (N, hidden_dim)
        f  = h @ W2 + b2        shape: (N,)
        y  ~ Normal(f, sigma)   shape: (N,)

    eqx.tree_at bridge
    -------------------
    All four weight arrays are replaced in a single call using a tuple
    selector::

        net = eqx.tree_at(
            lambda m: (m.W1, m.b1, m.W2, m.b2),  # selector: 4 leaves
            net,                                    # pytree
            (W1, b1, W2, b2),                       # replacement: 4 arrays
        )

    The selector returns a tuple of four leaves; the replacement is a
    matching tuple.  Equinox pairs them positionally:
    m.W1 <-> W1, m.b1 <-> b1, m.W2 <-> W2, m.b2 <-> b2.

    With ``AutoDelta`` as the guide, SVI gives MAP (point) estimates.
    With NUTS, we get full Bayesian posterior samples.

    NumPyro sample sites
    ---------------------
    - ``"W1"``    : input-to-hidden weights.    Shape (1, hidden_dim).
    - ``"b1"``    : hidden bias.                Shape (hidden_dim,).
    - ``"W2"``    : hidden-to-output weights.   Shape (hidden_dim, 1).
    - ``"b2"``    : output bias.                Scalar.
    - ``"sigma"`` : observation noise std.      Scalar.
    - ``"f"``     : (deterministic) latent predictions.  Shape (N,).
    - ``"obs"``   : observed targets.           Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    hidden_dim : int
        Width of the hidden layer.
    """
    # --- Architecture ---
    # MLP with placeholder weights; key=PRNGKey(0) is arbitrary since
    # weights will be overwritten.
    net = MLP(hidden_dim=hidden_dim, key=jr.PRNGKey(0))

    # --- Sample all weights from isotropic Gaussian priors ---
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, hidden_dim)), 1.0))   # W1: (1, H)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(hidden_dim), 1.0))        # b1: (H,)
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((hidden_dim, 1)), 1.0))   # W2: (H, 1)
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))                          # b2: ()

    # --- Replace placeholder weights in the module ---
    # Tuple selector: (m.W1, m.b1, m.W2, m.b2) identifies all 4 leaves
    # Tuple replacement: (W1, b1, W2, b2) provides matching sampled arrays
    net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    # --- Forward pass & likelihood ---
    # net(x) calls MLP.__call__, computing tanh(x W1 + b1) W2 + b2
    # f: (N,) — latent predictions, recorded as deterministic site
    f = numpyro.deterministic("f", net(x))
    # sigma: () — noise std
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    # obs: (N,) — likelihood conditioned on y (or generative if y is None)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


def model_nnet_dropout(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    hidden_dim: int = 50,
    dropout_rate: float = 0.1,
) -> None:
    """MC-Dropout neural network: Equinox MLPDropout + NumPyro mask sampling.

    The dropout mask is a first-class ``numpyro.sample`` site, so
    ``Predictive`` automatically draws fresh masks for each posterior
    sample — giving MC-Dropout uncertainty without any special
    predict-time logic.

    Generative model
    ----------------
        W1    ~ Normal(0, I)                      shape: (1, hidden_dim)
        b1    ~ Normal(0, I)                      shape: (hidden_dim,)
        W2    ~ Normal(0, I)                      shape: (hidden_dim, 1)
        b2    ~ Normal(0, 1)                      shape: ()
        sigma ~ HalfNormal(1)                     shape: ()

        mask  ~ Bernoulli(1 - dropout_rate)       shape: (N, hidden_dim)

        h_raw = tanh(x @ W1 + b1)                shape: (N, hidden_dim)
        h     = h_raw * mask / (1 - dropout_rate) shape: (N, hidden_dim)
        f     = h @ W2 + b2                       shape: (N,)
        y     ~ Normal(f, sigma)                  shape: (N,)

    eqx.tree_at bridge
    -------------------
    Same pattern as ``model_nnet`` — tuple selector replaces 4 weight
    arrays simultaneously::

        net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (...))

    The dropout mask is passed as an explicit argument to ``net(x, mask)``
    rather than being injected via ``eqx.tree_at``, because it is
    data-dependent (its shape depends on N) and varies per forward pass.

    NumPyro sample sites
    ---------------------
    - ``"W1"``           : input-to-hidden weights.   Shape (1, hidden_dim).
    - ``"b1"``           : hidden bias.               Shape (hidden_dim,).
    - ``"W2"``           : hidden-to-output weights.  Shape (hidden_dim, 1).
    - ``"b2"``           : output bias.               Scalar.
    - ``"dropout_mask"`` : binary mask.               Shape (N, hidden_dim).
    - ``"sigma"``        : observation noise std.     Scalar.
    - ``"f"``            : (deterministic) latent predictions.  Shape (N,).
    - ``"obs"``          : observed targets.          Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    hidden_dim : int
        Width of the hidden layer.
    dropout_rate : float
        Probability of dropping each hidden unit.
    """
    N = x.shape[0]  # N: number of data points
    net = MLPDropout(hidden_dim=hidden_dim, dropout_rate=dropout_rate, key=jr.PRNGKey(0))

    # --- Sample weights ---
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, hidden_dim)), 1.0))   # W1: (1, H)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(hidden_dim), 1.0))        # b1: (H,)
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((hidden_dim, 1)), 1.0))   # W2: (H, 1)
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))                          # b2: ()

    # Inject sampled weights into the MLPDropout module (same tuple pattern)
    net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    # --- Sample dropout mask ---
    # mask: (N, H) — each entry is Bernoulli(1 - dropout_rate)
    # 1 = keep the hidden unit, 0 = drop it
    # Because this is a numpyro.sample site, Predictive will re-draw
    # the mask for each posterior sample, giving MC-Dropout uncertainty.
    mask = numpyro.sample(
        "dropout_mask",
        dist.Bernoulli(probs=(1.0 - dropout_rate) * jnp.ones((N, hidden_dim))),
    )  # mask: (N, H)

    # --- Forward pass (mask is an explicit argument, not injected via tree_at) ---
    # net(x, mask) applies inverted dropout inside MLPDropout.__call__
    # f: (N,) — latent predictions
    f = numpyro.deterministic("f", net(x, mask))
    # sigma: () — noise std
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    # obs: (N,)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


def model_bayesian_nnet(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    hidden_dim: int = 50,
    prior_scale: float = 1.0,
) -> None:
    """Full Bayesian neural network with configurable prior scale.

    Identical architecture to ``model_nnet`` but with an adjustable
    ``prior_scale`` parameter that controls the width of the Gaussian
    prior on all weights.  Intended for full MCMC inference (NUTS),
    where the prior scale acts as a form of regularisation:

    - Small ``prior_scale`` (e.g. 0.1) → strong regularisation, smoother fits.
    - Large ``prior_scale`` (e.g. 5.0) → weak regularisation, more flexible.

    Generative model
    ----------------
        W1    ~ Normal(0, prior_scale * I)    shape: (1, hidden_dim)
        b1    ~ Normal(0, prior_scale * I)    shape: (hidden_dim,)
        W2    ~ Normal(0, prior_scale * I)    shape: (hidden_dim, 1)
        b2    ~ Normal(0, prior_scale)        shape: ()
        sigma ~ HalfNormal(1)                 shape: ()

        h  = tanh(x @ W1 + b1)               shape: (N, hidden_dim)
        f  = h @ W2 + b2                      shape: (N,)
        y  ~ Normal(f, sigma)                 shape: (N,)

    eqx.tree_at bridge
    -------------------
    Same tuple-of-four-leaves pattern as ``model_nnet``::

        net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    NumPyro sample sites
    ---------------------
    - ``"W1"``    : input-to-hidden weights.    Shape (1, hidden_dim).
    - ``"b1"``    : hidden bias.                Shape (hidden_dim,).
    - ``"W2"``    : hidden-to-output weights.   Shape (hidden_dim, 1).
    - ``"b2"``    : output bias.                Scalar.
    - ``"sigma"`` : observation noise std.      Scalar.
    - ``"f"``     : (deterministic) latent predictions.  Shape (N,).
    - ``"obs"``   : observed targets.           Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    hidden_dim : int
        Width of the hidden layer.
    prior_scale : float
        Standard deviation of the Gaussian prior on all weights.
    """
    net = MLP(hidden_dim=hidden_dim, key=jr.PRNGKey(0))

    # Sample weights with configurable prior_scale (controls regularisation strength)
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, hidden_dim)), prior_scale))   # W1: (1, H)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(hidden_dim), prior_scale))        # b1: (H,)
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((hidden_dim, 1)), prior_scale))   # W2: (H, 1)
    b2 = numpyro.sample("b2", dist.Normal(0.0, prior_scale))                          # b2: ()

    # Inject all four sampled arrays into the MLP module
    net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    # f: (N,) — forward pass through the now-parameterised MLP
    f = numpyro.deterministic("f", net(x))
    # sigma: () — observation noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    # obs: (N,) — likelihood
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


def model_svr_rff(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    rff_module: RFFFeatureMap,
) -> None:
    """SVR via Random Fourier Features: fixed Equinox RFF map + NumPyro linear model.

    The ``rff_module`` is an already-constructed ``RFFFeatureMap`` whose
    frequencies omega and phases b are *fixed* (drawn once at construction
    time).  Only the linear weights w and noise sigma are inferred.

    This is equivalent to kernel ridge regression / support vector
    regression with an RBF kernel, where the kernel is approximated by
    the random feature inner product:  k(x, x') ≈ phi(x)^T phi(x').

    Reference: Rahimi & Recht (2007), "Random Features for Large-Scale
    Kernel Machines."

    Generative model
    ----------------
        phi(x) = sqrt(2/D) cos(omega x + b)   — fixed RFF map (Equinox module)
        w      ~ Normal(0, I)                  shape: (D,)
        sigma  ~ HalfNormal(1)                 shape: ()

        Phi    = phi(x)                        shape: (N, D)
        f      = Phi @ w                       shape: (N,)
        y      ~ Normal(f, sigma)              shape: (N,)

    Note on eqx.tree_at
    --------------------
    This model does NOT use ``eqx.tree_at`` because the RFF module has
    no learnable parameters — its omega and bias are fixed.  The linear
    prediction is computed inline with ``einsum(Phi, w, "n d, d -> n")``,
    where ``w`` is a raw NumPyro sample (not injected into a module).

    NumPyro sample sites
    ---------------------
    - ``"w"``     : linear weights in RFF feature space.  Shape (D,).
    - ``"sigma"`` : observation noise std.                Scalar.
    - ``"f"``     : (deterministic) latent predictions.   Shape (N,).
    - ``"obs"``   : observed targets.                     Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    rff_module : RFFFeatureMap
        Pre-constructed feature map (frequencies are fixed, not inferred).
        Its ``n_features`` attribute determines D.
    """
    D = rff_module.n_features

    # --- Feature map (deterministic, no NumPyro samples) ---
    # Phi: (N, D) — the D-dimensional RFF embedding of each input
    Phi = rff_module(x)  # Phi: (N, D)

    # --- Sample linear weights ---
    # w: (D,) — one weight per random Fourier feature
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), 1.0))
    # sigma: () — observation noise std
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    # einsum "n d, d -> n": contracts over d (the D random features)
    # yielding one scalar prediction per data point
    # f: (N,)
    f = numpyro.deterministic("f", einsum(Phi, w, "n d, d -> n"))
    # obs: (N,)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


def model_gp_rff(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    rff_module: RFFFeatureMap,
) -> None:
    """Approximate Gaussian Process via RFF: hierarchical version of model_svr_rff.

    Like ``model_svr_rff`` but adds a hierarchical prior on the weight
    amplitude alpha, which acts as a learnable kernel amplitude
    (signal variance) parameter.  This makes the model closer to a true
    GP with learned hyperparameters.

    The RFF module's lengthscale is still fixed at construction time.
    To learn the lengthscale, one would construct multiple RFF modules
    with different lengthscales and use model selection (e.g. via
    marginal likelihood or cross-validation).

    Reference: Rahimi & Recht (2007), "Random Features for Large-Scale
    Kernel Machines."

    Generative model
    ----------------
        alpha  ~ HalfNormal(1)                 — weight scale (approx. kernel amplitude)
        w      ~ Normal(0, alpha * I)          shape: (D,)
        sigma  ~ HalfNormal(0.1)               shape: ()

        phi(x) = sqrt(2/D) cos(omega x + b)   — fixed RFF map
        Phi    = phi(x)                        shape: (N, D)
        f      = Phi @ w                       shape: (N,)
        y      ~ Normal(f, sigma)              shape: (N,)

    The hierarchical structure (alpha → w) lets the model learn how much
    signal variance to attribute to the function vs. noise.  With a fixed
    w ~ Normal(0, 1) prior (as in model_svr_rff), the signal amplitude
    is implicitly fixed.

    Note on eqx.tree_at
    --------------------
    Same as ``model_svr_rff`` — no ``eqx.tree_at`` is used.  The RFF
    module is called directly and the linear prediction is computed
    inline.

    NumPyro sample sites
    ---------------------
    - ``"amplitude"`` : weight prior scale (approx. kernel amplitude).  Scalar.
    - ``"w"``         : linear weights in RFF feature space.  Shape (D,).
    - ``"sigma"``     : observation noise std.                Scalar.
    - ``"f"``         : (deterministic) latent predictions.   Shape (N,).
    - ``"obs"``       : observed targets.                     Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    rff_module : RFFFeatureMap
        Pre-constructed feature map (frequencies are fixed, not inferred).
    """
    D = rff_module.n_features
    # Phi: (N, D) — fixed RFF features
    Phi = rff_module(x)

    # --- Hierarchical prior on weight scale ---
    # amplitude: () — scalar, acts as learned kernel signal variance
    amplitude = numpyro.sample("amplitude", dist.HalfNormal(1.0))
    # w: (D,) — weights scaled by amplitude (wider amplitude → larger function values)
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), amplitude))
    # sigma: () — observation noise, with a tighter prior (0.1) than model_svr_rff (1.0)
    # reflecting the expectation that the GP should explain most of the variance
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # einsum "n d, d -> n": contracts over d (RFF features) → (N,)
    f = numpyro.deterministic("f", einsum(Phi, w, "n d, d -> n"))  # f: (N,)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)  # obs: (N,)


def model_deep_gp_rff(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    deep_rff_module: DeepRFFRegressor,
) -> None:
    """Approximate Deep Gaussian Process via two-layer RFF.

    The RFF maps (omega, b in both layers) are fixed in the Equinox
    module.  We sample the inter-layer weights W1 and output weights w2
    from Gaussian priors, then inject them into the module via
    ``eqx.tree_at``.

    Reference: Cutajar, K., Bonilla, E. V., Michiardi, P. & Filippone, M.
    (2017). "Random Feature Expansions for Deep Gaussian Processes." ICML 2017.

    Generative model
    ----------------
        W1    ~ Normal(0, I)          shape: (D1, inner_dim)
        w2    ~ Normal(0, I)          shape: (D2,)
        sigma ~ HalfNormal(0.1)       shape: ()

        Phi1  = rff1(x)              shape: (N, D1)        — fixed layer-1 RFF
        h     = Phi1 @ W1            shape: (N, inner_dim) — hidden representation
        Phi2  = mean_j rff2(h_j)     shape: (N, D2)        — averaged layer-2 RFF
        f     = Phi2 @ w2            shape: (N,)           — output predictions
        y     ~ Normal(f, sigma)     shape: (N,)

    eqx.tree_at bridge
    -------------------
    Only the two learnable weight arrays (W1, w2) are replaced; the RFF
    maps (rff1, rff2) pass through unchanged::

        mod = eqx.tree_at(
            lambda m: (m.W1, m.w2),   # selector: 2 weight leaves only
            mod,                       # pytree: the DeepRFFRegressor
            (W1, w2),                  # replacement: 2 sampled arrays
        )

    The selector does NOT touch ``m.rff1`` or ``m.rff2``, so their
    frozen omega and bias arrays remain intact in the new module.

    NumPyro sample sites
    ---------------------
    - ``"W1"``    : inter-layer weights.          Shape (D1, inner_dim).
    - ``"w2"``    : output-layer weights.         Shape (D2,).
    - ``"sigma"`` : observation noise std.        Scalar.
    - ``"f"``     : (deterministic) predictions.  Shape (N,).
    - ``"obs"``   : observed targets.             Shape (N,).

    Parameters
    ----------
    x : shape (N,)
        Input locations.
    y : shape (N,) or None
        Observed targets (None for predictive sampling).
    deep_rff_module : DeepRFFRegressor
        Pre-constructed two-layer RFF module with fixed random features.
    """
    mod = deep_rff_module
    D1 = mod.rff1.n_features     # number of layer-1 RFF features
    D2 = mod.rff2.n_features     # number of layer-2 RFF features
    inner_dim = mod.inner_dim    # hidden representation width

    # --- Sample learnable weights ---
    # W1: (D1, inner_dim) — maps from layer-1 RFF space to hidden representation
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((D1, inner_dim)), 1.0))
    # w2: (D2,) — maps from (averaged) layer-2 RFF space to scalar output
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros(D2), 1.0))

    # --- Replace weights in the Equinox module ---
    # selector: lambda m: (m.W1, m.w2) — only the two weight arrays
    # The RFF maps (m.rff1, m.rff2) are NOT selected, so they stay frozen
    mod = eqx.tree_at(lambda m: (m.W1, m.w2), mod, (W1, w2))

    # --- Forward pass ---
    # mod(x) calls DeepRFFRegressor.__call__, which:
    #   1. Computes Phi1 = rff1(x)        → (N, D1)
    #   2. Computes h = Phi1 @ W1         → (N, inner_dim)
    #   3. Averages rff2 over hidden dims → (N, D2)
    #   4. Computes f = Phi2 @ w2         → (N,)
    # f: (N,) — latent predictions
    f = numpyro.deterministic("f", mod(x))
    # sigma: () — tight noise prior (0.1), reflecting deep GP's expressiveness
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
    # obs: (N,)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)


# ============================================================================
# 3b. Additional NumPyro model functions (ported from Flax masterclass)
# ============================================================================

# --- NCP: Noise Contrastive Prior ---
#
# Hafner et al. (2018): "Noise Contrastive Priors for Functional Uncertainty".
#
# The NCP pattern provides calibrated uncertainty by regularising predictions
# on noise-perturbed inputs toward a simple prior distribution.  Far from
# training data, clean and perturbed inputs look similar, so the network
# is regularised toward the prior.  Near data, the signal-to-noise ratio
# is high and the regularisation has little effect.
#
# In the Equinox edition we use a plain MLP (deterministic hidden layers)
# and implement the NCP logic inline in the model function — no special
# PyroxModule or Flax compact method needed.


class NCPNetwork(eqx.Module):
    """Neural network architecture for Noise Contrastive Prior.

    Deterministic hidden layers: the NCP perturbation and variational
    output layer are handled in the model function, not the module.

    Architecture
    ------------
    ::

        x: (2N, 1)  →  Dense(32) + selu  →  Dense(16) + selu  →  Dense(8) + selu
        → h: (2N, 8)

    Attributes
    ----------
    W1, b1 : input → hidden1 (32 units)
    W2, b2 : hidden1 → hidden2 (16 units)
    W3, b3 : hidden2 → hidden3 (8 units)
    """

    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array
    W3: jax.Array
    b3: jax.Array

    def __init__(self, *, key: jax.Array):
        k1, k2, k3 = jr.split(key, 3)
        self.W1 = jr.normal(k1, (1, 32)) * 0.1
        self.b1 = jnp.zeros(32)
        self.W2 = jr.normal(k2, (32, 16)) * 0.1
        self.b2 = jnp.zeros(16)
        self.W3 = jr.normal(k3, (16, 8)) * 0.1
        self.b3 = jnp.zeros(8)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through deterministic hidden layers.

        Parameters
        ----------
        x : shape (B, 1) — input (may be doubled batch from NCP)

        Returns
        -------
        h : shape (B, 8) — hidden representation
        """
        # h1: (B, 32)
        h = jax.nn.selu(einsum(x, self.W1, "b d, d h -> b h") + self.b1)
        # h2: (B, 16)
        h = jax.nn.selu(einsum(h, self.W2, "b d, d h -> b h") + self.b2)
        # h3: (B, 8)
        h = jax.nn.selu(einsum(h, self.W3, "b d, d h -> b h") + self.b3)
        return h


def model_nnet_ncp(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    perturb_noise: float = 10.0,
    latent_std: float = 1.0,
    prior_std: float = 0.1,
) -> None:
    """Neural network with Noise Contrastive Prior (Hafner et al. 2018).

    The NCP idea: double the batch by concatenating clean inputs with
    noise-perturbed copies.  Run both through the network.  Penalise
    the perturbed-output distribution toward a simple prior via KL.
    Far from data, this forces prior-like (uncertain) predictions.

    Generative model
    ----------------
    ::

        # Input perturbation:
        epsilon ~ Normal(0, perturb_noise^2)           (N, 1)
        x_doubled = [x; x + epsilon]                   (2N, 1)

        # Deterministic hidden layers (MAP, via eqx.tree_at):
        h = selu(x_doubled @ W1 + b1)  →  selu(... @ W2 + b2)  →  selu(... @ W3 + b3)
        h: (2N, 8)

        # Variational output layer:
        W4 ~ Normal(0, prior_std)                      (8, 1)
        b4 ~ Normal(0, prior_std)                      (1,)
        f_all = h @ W4 + b4                            (2N, 1)
        f_clean = f_all[:N]                            (N,)
        f_noisy = f_all[N:]                            (N,)

        # NCP KL penalty (function-space regularisation):
        KL( Normal(mean(f_noisy), std(f_noisy)) || Normal(0, latent_std) )

        # Observation:
        noise ~ param > 0
        y ~ Normal(f_clean, noise)

    eqx.tree_at bridge
    -------------------
    Only the hidden layers (W1..W3) use ``eqx.tree_at`` for MAP injection.
    The output layer (W4, b4) is sampled directly via ``numpyro.sample``.

    References
    ----------
    Hafner et al. (2018): "Noise Contrastive Priors for Functional
    Uncertainty", arXiv:1807.09289.
    """
    net = NCPNetwork(key=jr.PRNGKey(0))

    # --- MAP hidden layers via eqx.tree_at ---
    W1 = numpyro.param("W1", net.W1)
    b1 = numpyro.param("b1", net.b1)
    W2 = numpyro.param("W2", net.W2)
    b2 = numpyro.param("b2", net.b2)
    W3 = numpyro.param("W3", net.W3)
    b3 = numpyro.param("b3", net.b3)
    net = eqx.tree_at(
        lambda m: (m.W1, m.b1, m.W2, m.b2, m.W3, m.b3),
        net,
        (W1, b1, W2, b2, W3, b3),
    )

    # --- NCP input perturbation ---
    # x_col: (N, 1) — ensure 2D for matrix ops
    x_col = rearrange(x, "n -> n 1")
    # epsilon: (N, 1) — Gaussian noise
    epsilon = numpyro.sample(
        "epsilon",
        dist.Normal(jnp.zeros_like(x_col), perturb_noise),
    )
    # x_doubled: (2N, 1) — clean + noisy concatenated
    x_doubled = jnp.concatenate([x_col, x_col + epsilon], axis=0)

    # --- Hidden layers (deterministic, MAP) ---
    # h: (2N, 8)
    h = net(x_doubled)

    # --- Variational output layer ---
    # W4: (8, 1) — sampled (not MAP)
    W4 = numpyro.sample(
        "W4",
        dist.Normal(jnp.zeros((8, 1)), prior_std).to_event(2),
    )
    # b4: (1,) — sampled
    b4 = numpyro.sample(
        "b4",
        dist.Normal(jnp.zeros(1), prior_std).to_event(1),
    )
    # f_all: (2N, 1)
    f_all = einsum(h, W4, "b d, d one -> b one") + b4

    # --- Split clean / noisy ---
    n_half = x.shape[0]
    f_clean = f_all[:n_half].squeeze(-1)  # (N,)
    f_noisy = f_all[n_half:].squeeze(-1)  # (N,)

    # --- NCP KL penalty ---
    # Fit Normal to perturbed outputs, penalise toward N(0, latent_std)
    f_prior = dist.Normal(0.0, latent_std)
    ncp_kl = dist.kl_divergence(
        dist.Normal(f_noisy.mean(), f_noisy.std() + 1e-6),
        f_prior,
    ).sum()
    numpyro.deterministic("ncp_kl", ncp_kl)
    numpyro.factor("ncp_penalty", -ncp_kl)

    # --- Observation ---
    f = numpyro.deterministic("f", f_clean)
    noise = numpyro.param("noise", 0.5, constraint=dist.constraints.positive)
    numpyro.sample("obs", dist.Normal(f, noise), obs=y)


# --- Parameterized GP kernel (Equinox edition) ---
#
# This demonstrates the Parameterized-style API ported to pure Equinox:
# an RBF kernel where hyperparameters are registered with priors and
# the model/guide pair manages inference.  Unlike the Flax version which
# uses class-level registries and mode switching, the Equinox version
# keeps things explicit: the model function samples hyperparams and
# passes them to the kernel via eqx.tree_at.


class RBFKernel(eqx.Module):
    """RBF (squared exponential) kernel as an Equinox module.

    Kernel function
    ---------------
    ::

        k(x, x') = variance * exp( -0.5 * ||x - x'||^2 / lengthscale^2 )

    Attributes
    ----------
    log_variance : jax.Array
        Log of the signal variance (unconstrained parameterisation).
    log_lengthscale : jax.Array
        Log of the lengthscale (unconstrained parameterisation).
    """

    log_variance: jax.Array
    log_lengthscale: jax.Array

    def __init__(
        self,
        variance: float = 1.0,
        lengthscale: float = 1.0,
    ):
        self.log_variance = jnp.log(jnp.array(variance))
        self.log_lengthscale = jnp.log(jnp.array(lengthscale))

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        """Compute the RBF Gram matrix.

        Parameters
        ----------
        X1 : shape (N1, D)
        X2 : shape (N2, D)

        Returns
        -------
        K : shape (N1, N2)
        """
        variance = jnp.exp(self.log_variance)
        lengthscale = jnp.exp(self.log_lengthscale)
        X1_s = X1 / lengthscale
        X2_s = X2 / lengthscale
        sq_dist = (
            jnp.sum(X1_s**2, axis=-1, keepdims=True)
            - 2.0 * X1_s @ X2_s.T
            + jnp.sum(X2_s**2, axis=-1)
        )
        return variance * jnp.exp(-0.5 * sq_dist)


def model_gp_parameterized(
    x: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
) -> None:
    """Full GP model with Equinox RBF kernel and sampled hyperparameters.

    Generative model
    ----------------
    ::

        variance    ~ LogNormal(0, 1)
        lengthscale ~ LogNormal(0, 1)
        K           = k_RBF(x, x; variance, lengthscale) + 1e-4 * I
        y           ~ MultivariateNormal(0, K)

    eqx.tree_at bridge
    -------------------
    The kernel's ``log_variance`` and ``log_lengthscale`` are replaced
    with values sampled from their priors via ``eqx.tree_at``.

    Parameters
    ----------
    x : shape (N, D)
        Input locations.
    y : shape (N,) or None
        Observed function values.
    """
    kernel = RBFKernel()

    # Sample kernel hyperparameters (unconstrained → log space)
    log_var = numpyro.sample("log_variance", dist.Normal(0.0, 1.0))
    log_ls = numpyro.sample("log_lengthscale", dist.Normal(0.0, 1.0))

    # Inject into kernel via eqx.tree_at
    kernel = eqx.tree_at(
        lambda k: (k.log_variance, k.log_lengthscale),
        kernel,
        (log_var, log_ls),
    )

    # Compute Gram matrix + jitter
    N = x.shape[0]
    K = kernel(x, x) + 1e-4 * jnp.eye(N)

    # GP observation model
    numpyro.sample(
        "obs",
        dist.MultivariateNormal(jnp.zeros(N), covariance_matrix=K),
        obs=y,
    )


# ============================================================================
# 4.  Inference utilities
# ============================================================================


def run_mcmc(
    model,
    key: jax.Array,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    num_warmup: int = 1_000,
    num_samples: int = 2_000,
    **model_kwargs,
) -> MCMC:
    """Run NUTS (No-U-Turn Sampler) MCMC on a NumPyro model.

    This is a convenience wrapper around NumPyro's MCMC interface.  It
    constructs a NUTS kernel, creates an MCMC runner, executes warmup
    and sampling, and returns the fitted MCMC object (which holds
    posterior samples).

    Parameters
    ----------
    model : callable
        A NumPyro model function (e.g. ``model_linear``, ``model_nnet``).
        Must accept ``(x, y=..., **model_kwargs)`` as arguments.
    key : jax.Array
        PRNG key for MCMC randomness (proposal, initialisation, etc.).
    x : shape (N,)
        Training input locations.
    y : shape (N,)
        Training targets.
    num_warmup : int
        Number of warmup (burn-in / adaptation) steps.  During warmup,
        NUTS adapts the step size and mass matrix.
    num_samples : int
        Number of post-warmup posterior samples to collect.
    **model_kwargs
        Additional keyword arguments forwarded to the model function
        (e.g. ``degree=3``, ``hidden_dim=100``).

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        The fitted MCMC object.  Access posterior samples via
        ``mcmc.get_samples()`` which returns a dict mapping site names
        to arrays of shape ``(num_samples, *site_shape)``.
    """
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(key, x, y=y, **model_kwargs)
    return mcmc


def run_svi(
    model,
    guide,
    key: jax.Array,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    num_steps: int = 5_000,
    lr: float = 0.01,
    **model_kwargs,
):
    """Run Stochastic Variational Inference (SVI) on a NumPyro model.

    Optimises the ELBO (Evidence Lower BOund) using Adam.  Commonly
    used with ``autoguide.AutoDelta`` for MAP estimation or
    ``autoguide.AutoNormal`` for mean-field variational Bayes.

    Parameters
    ----------
    model : callable
        A NumPyro model function.
    guide : callable or autoguide
        The variational guide / approximate posterior.  Common choices:
        - ``autoguide.AutoDelta(model)`` for MAP (point estimates).
        - ``autoguide.AutoNormal(model)`` for mean-field Gaussian VI.
    key : jax.Array
        PRNG key for SVI randomness.
    x : shape (N,)
        Training input locations.
    y : shape (N,)
        Training targets.
    num_steps : int
        Number of optimisation steps.
    lr : float
        Adam learning rate.
    **model_kwargs
        Additional keyword arguments forwarded to the model function.

    Returns
    -------
    svi_result : numpyro.infer.SVIRunResult
        Contains ``svi_result.params`` (optimised guide parameters) and
        ``svi_result.losses`` of shape ``(num_steps,)`` (ELBO loss trace).
        To extract posterior samples, use::

            samples = guide.sample_posterior(key, svi_result.params)
    """
    from numpyro.optim import Adam
    optimizer = Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    return svi.run(key, num_steps, x, y=y, **model_kwargs)


def predict(
    model,
    key: jax.Array,
    samples: dict,
    x_new: jnp.ndarray,
    **model_kwargs,
) -> dict:
    """Generate posterior predictive samples via ``Predictive``.

    For each set of posterior parameter values in ``samples``, this
    runs the model forward on ``x_new`` to produce predictions.

    Parameters
    ----------
    model : callable
        A NumPyro model function.
    key : jax.Array
        PRNG key for any stochastic sites in the model (e.g. dropout
        masks, observation noise).
    samples : dict
        Posterior samples, mapping site names to arrays of shape
        ``(S, *site_shape)`` where S is the number of posterior samples.
        Typically obtained from ``mcmc.get_samples()`` or
        ``guide.sample_posterior(...)``.
    x_new : shape (M,)
        New input locations at which to make predictions.
    **model_kwargs
        Additional keyword arguments forwarded to the model function.

    Returns
    -------
    predictions : dict
        Maps site names to predictive arrays.  Key entries:
        - ``"f"``   : shape (S, M) — latent function samples (no noise).
        - ``"obs"`` : shape (S, M) — observation samples (with noise).
        Other deterministic sites may also appear depending on the model.
    """
    predictive = Predictive(model, posterior_samples=samples)
    return predictive(key, x_new, **model_kwargs)


def compute_r2(y_true: jnp.ndarray, y_pred_mean: jnp.ndarray) -> float:
    """Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot

    where SS_res = sum((y_true - y_pred_mean)^2) is the residual sum of
    squares and SS_tot = sum((y_true - mean(y_true))^2) is the total sum
    of squares.

    R^2 = 1.0 means perfect prediction; R^2 = 0.0 means the model is no
    better than predicting the mean; R^2 < 0.0 means worse than the mean.

    Parameters
    ----------
    y_true : shape (M,)
        Ground truth target values.
    y_pred_mean : shape (M,)
        Predicted mean values (e.g. posterior predictive mean).

    Returns
    -------
    r2 : float
        The R^2 score (scalar).
    """
    ss_res = jnp.sum((y_true - y_pred_mean) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def summarise_predictions(
    predictions: dict,
    y_true: jnp.ndarray,
    quantile: float = 0.90,
) -> dict:
    """Compute summary statistics from posterior predictive samples.

    Takes the raw output of ``predict()`` and computes means, highest
    posterior density intervals (HPDI), and the R^2 score.

    Parameters
    ----------
    predictions : dict
        Output of ``predict()``.  Must contain:
        - ``"f"``   : shape (S, M) — latent function samples.
        - ``"obs"`` : shape (S, M) — observation samples (with noise).
        where S = number of posterior samples and M = number of test points.
    y_true : shape (M,)
        Ground truth target values at the test points.
    quantile : float
        Probability mass for the HPDI interval (e.g. 0.90 for a 90%
        credible interval).

    Returns
    -------
    summary : dict with keys:
        - ``"f_mean"``   : shape (M,)    — posterior mean of latent function.
        - ``"f_hpdi"``   : shape (2, M)  — [lower, upper] HPDI bounds for f.
        - ``"obs_mean"`` : shape (M,)    — posterior mean of observations.
        - ``"obs_hpdi"`` : shape (2, M)  — [lower, upper] HPDI bounds for obs.
        - ``"r2"``       : float         — R^2 of f_mean vs y_true.
    """
    # f_samples: (S, M) — S posterior draws of the latent function at M points
    f_samples = predictions["f"]
    # obs_samples: (S, M) — S posterior draws of noisy observations
    obs_samples = predictions["obs"]
    # f_mean: (M,) — point estimate (posterior predictive mean)
    f_mean = jnp.mean(f_samples, axis=0)
    return dict(
        f_mean=f_mean,                                  # (M,)
        f_hpdi=hpdi(f_samples, prob=quantile),          # (2, M)
        obs_mean=jnp.mean(obs_samples, axis=0),         # (M,)
        obs_hpdi=hpdi(obs_samples, prob=quantile),      # (2, M)
        r2=compute_r2(y_true, f_mean),                  # scalar
    )
