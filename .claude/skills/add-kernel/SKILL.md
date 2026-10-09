---
name: add-kernel
description: Add a GP kernel to pyrox-gp — a Parameterized wrapper over a kernellib kernel, with constrained hyperparameters that accept priors and guides, a frozen() form, and its tests and docs. Use when asked to add, port or wrap a covariance function / kernel (stationary, periodic, dot-product, spectral, …) in packages/pyrox-gp.
---

# Add a kernel (pyrox-gp)

Read the "Parameterized" contract in the root `AGENTS.md` and
`packages/pyrox-gp/AGENTS.md` first.

## 1. Make sure it does not exist yet

- Search `docs/capabilities.md`: the `pyrox_gp` kernels, and the
  `kernellib` / `kernellib.functional` sections.
- **The kernel math lives in kernellib.** If `kernellib.functional` has no
  function for it, it goes to kernellib first (a separate PR there, then a
  pin bump with the `bump-geoml-deps` skill). Don't write kernel math here.

## 2. The class (`src/pyrox_gp/_kernels.py`)

Copy `RBF` (or `Matern` when there is a static structural field such as
`nu`):

- `class NewKernel(_ParameterizedKernel):` with fields
  `pyrox_name: str = "NewKernel"`, `init_<param>: float = …` for each
  hyperparameter, and `input_dim: int | None = None` if it supports ARD.
  Structural settings (`nu`, `degree`, `period` if fixed) are plain fields,
  not registered params.
- `setup()`: `self.register_param("<param>", jnp.asarray(self.init_<param>),
  constraint=dist.constraints.positive)` (or the right support) for each.
- `@pyrox_method def __call__(self, X1, X2)`: call the
  `kernellib.functional` function with `self.get_param("<param>")`.
- Override `diag(X)` unless the default (`variance * ones`) is right
  (non-stationary kernels need it).
- Set `_frozen_cls = kl.NewKernel`, `_frozen_params` and `_frozen_static`
  so `frozen()` returns the kernellib kernel; the matrix-free path
  (`freeze_kernel`), `init_inducing` and `_basis` rely on it.
- If it has a spectral density usable by inducing features or HSGP, check
  `_STATIONARY_KERNELS` in `_inducing.py` and `spectral_density` in
  `_basis/_spectral_density.py`.
- Docstring: the formula in MathJax, the registered sites
  (`NewKernel.variance`, …), the constraints, an `Examples:` block.

## 3. Export and document

- `src/pyrox_gp/__init__.py`: import + `__all__`.
- `::: pyrox_gp.NewKernel` in `docs/api/gp.md`.
- `make capabilities`; a name equal to the kernellib kernel's goes into
  `ALLOWED_SHARED_NAMES` (the `_KERNEL` reason) in `scripts/capabilities.py`.

## 4. Tests (`packages/pyrox-gp/tests/gp/test_kernel_classes.py`, `test_frozen.py`)

- Under `handlers.seed(rng_seed=0)`, the Gram equals the
  `kernellib.functional` function at the init values.
- `diag(X)` equals the Gram's diagonal.
- The sites appear in `handlers.trace()` (`param` without a prior,
  `sample` after `set_prior`; after `autoguide` + `set_mode("guide")`, a
  `Delta` site (`"delta"`) or a transformed-Normal site (`"normal"`, which
  maps back onto the positive support)).
- `frozen()` returns the kernellib kernel with the same Gram.
- Inside `GPPrior` + `gp_factor`, a model with priors on the
  hyperparameters runs under SVI for a few steps.

## 5. Verify

`uv run pytest --no-cov packages/pyrox-gp/tests -m "not slow"`, then the
`pre-pr-check` skill.
