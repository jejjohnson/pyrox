---
name: add-bayesian-layer
description: Add a Bayesian or uncertainty-aware layer to pyrox-nn — a PyroxModule whose weights are NumPyro sites, either sampled from a prior or swapped into a wrapped geonnax core (dense / conv variants, random features, SIREN / MFN, SNGP, ensembles, heteroscedastic heads, conditioning). Use when asked to add, port or implement a Bayesian / variational / probabilistic neural-network layer in packages/pyrox-nn.
---

# Add a Bayesian layer (pyrox-nn)

Read "The contracts" in the root `AGENTS.md` (sites) and
`packages/pyrox-nn/AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md` for the layer and its synonyms (Flipout,
  NCP, rank-1 / BatchEnsemble, SNGP, HSGP, …) in `pyrox_nn` **and** in
  `geonnax` (the deterministic core may exist already).
- The deterministic network belongs in **geonnax**; pyrox-nn only adds the
  probabilistic wrapper. If the core is missing, it goes to geonnax first
  (ask before adding it here).

## 2. Pick the shape

- **Pure prior** (the weights *are* the random variables): exemplars
  `DenseReparameterization` (`_dense.py`), `BayesianSIREN` (`_siren.py`).
- **Wrapped core** (a geonnax module whose arrays become sites): exemplars
  `_sngp.py`, `_vssgp.py`, `_heteroscedastic.py`.
- **GP-flavoured features** (spectral densities, HSGP): exemplar
  `HSGPFeatures` in `_features.py` (uses `pyrox_gp._basis` and the kernel
  context).

## 3. Write it (`src/pyrox_nn/_<family>.py`)

- Subclass `PyroxModule` (`from pyrox._core import PyroxModule,
  pyrox_method`). Every configuration field is `eqx.field(static=True)`;
  add `pyrox_name: str | None = eqx.field(static=True, default=None)`.
- Build with `@classmethod init(cls, ..., *, pyrox_name=None)`: keyword-only
  options, `ValueError` on bad values (see `_require_positive` in
  `_siren.py`). A pure-prior layer takes no PRNG key.
- In `@pyrox_method def __call__(self, x)`:
  - pure prior: `W = self.pyrox_sample("weight",
    dist.Normal(0, s).expand([d_in, d_out]).to_event(2))`, then
    `einx.dot("... i, i o -> ... o", x, W)`;
  - wrapped core: register replacements (`self.pyrox_sample(...)` or
    `self.pyrox_param("W_loc", self.core.W_loc)`), splice them in with
    `eqx.tree_at`, then run the core;
  - a per-example core runs over `(*batch, D)` through
    `_batching.vmap_over_flat_batch(core, x)`, with the weights sampled
    once, outside the vmap;
  - indexed sites in a loop: `f"layer_{i}.W"`;
  - a raw `numpyro.factor` (e.g. a KL term) is named
    `self._pyrox_fullname("kl")`; `numpyro.prng_key()` needs a `seed`
    handler, so say so in the docstring.
- Docstring (Google, MathJax): the model, the priors, **the site names it
  registers** (`<pyrox_name>.weight`, …), `Args:` / `Returns:` with shapes,
  and an `Examples:` block that runs. No Sphinx markup
  (`tests/nn/test_docstrings.py`).
- End the module with `__all__`.

## 4. Export and document

- `src/pyrox_nn/__init__.py`: import it and add it to `__all__`.
- A `::: pyrox_nn.Name` entry in `docs/api/nn.md` or the topic page under
  `docs/api/nn/`.
- `make capabilities`. If the new name equals a geonnax core's name (the
  usual case for a wrapper), add it to `ALLOWED_SHARED_NAMES` in
  `scripts/capabilities.py` (the `_GEONNAX` reason already covers it).

## 5. Tests (`packages/pyrox-nn/tests/nn/test_<family>.py`)

- The exact site set under `with handlers.trace() as tr,
  handlers.seed(rng_seed=0):` (see
  `test_siren.py::test_bayesian_siren_registers_sites`).
- Two instances with distinct `pyrox_name`s in one model don't collide; the
  default name collides loudly under `trace`.
- Output shapes for `(D,)`, `(N, D)` and `(B, N, D)` inputs.
- It runs under SVI with `AutoNormal` and under `Predictive` (a short run;
  mark longer fits `slow`).
- For a wrapped core: with the sites substituted by the core's own values
  (`handlers.substitute`), the output equals the deterministic core's.

## 6. Verify

`uv run pytest --no-cov packages/pyrox-nn/tests -m "not slow"`, the slow
tests you added, then the `pre-pr-check` skill.
