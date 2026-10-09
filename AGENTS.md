# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here, and each package
adds its own rules in `packages/<package>/AGENTS.md`.

## What this repo is

pyrox is a uv workspace of four packages for probabilistic modelling with
Equinox and NumPyro. Its core is a **bridge**: an Equinox module that owns
NumPyro sample and param sites, so the same module runs under `handlers.trace`,
NUTS, SVI and `Predictive` unchanged. People build models on top of it, so
**the packages are primitives**: new code composes what is here (and what
gaussx, kernellib and geonnax provide), and anything genuinely new lands where
the next person will find and reuse it.

| Package (dist) | Import | Use it for | Depends on (internal) |
|---|---|---|---|
| `pyrox` | `pyrox._core`, `pyrox.inference` | The bridge (`PyroxModule`, `pyrox_method`, `Parameterized`) and ensemble MAP / VI | — |
| `pyrox-gp` | `pyrox_gp` | GP kernels (with priors), guides, likelihoods; exact, sparse, Markov, multi-output and warped GPs; non-Gaussian inference; pathwise sampling; the shared basis helpers (`_basis`) | pyrox |
| `pyrox-nn` | `pyrox_nn` (+ `pyrox_nn.api`, `pyrox_nn.preprocessing` behind `[bnf]`) | Bayesian / uncertainty-aware layers on geonnax cores, the Bayesian Neural Field estimator | pyrox, pyrox-gp |
| `pyrox-lgm` | `pyrox_lgm` | Latent Gaussian models in precision form: GMRF components, PC priors, `inla()` | pyrox (**never** pyrox-gp) |

Dependencies point one way: nothing imports upward, and `pyrox-lgm` never
imports `pyrox-gp` (`packages/pyrox-lgm/tests/test_import_guard.py`).
`pyrox` exports only `__version__`, `_core` and `inference`: the bridge's
public home is `pyrox._core` despite the underscore.

## What pyrox is built on

Each foundation brings a rule. Breaking one runs fine on one example and
fails under a handler, a transform or a second instance.

| Library | pyrox uses it for | The rule it brings |
|---|---|---|
| **NumPyro** | Sample / param / factor sites, handlers, MCMC, SVI, autoguides, distributions | Sites inside a module go through `pyrox_sample` / `pyrox_param` so they are scoped, cached per call and guarded; models stay plain NumPyro functions that every handler composes with. |
| **Equinox** | Immutable `eqx.Module` pytrees, `filter_*` transforms, `eqx.tree_at` | Modules, guides, likelihoods, results are `eqx.Module`s, never dataclasses. Configuration (shapes, names, flags) is `eqx.field(static=True)`. Rebuilding a pytree skips `__init__`. |
| **gaussx** | Every solve, logdet, Cholesky, Gaussian density, KL, GP conditioning, Kalman filter, quadrature, GMRF | Linear algebra on a covariance or precision goes through gaussx on a PSD-tagged operator, never `jnp.linalg` / `cho_solve`. Models carry `solver: AbstractSolverStrategy \| None` (default `DenseSolver()`). |
| **kernellib** | Kernel math (`kernellib.functional`), `AbstractKernel` (= `pyrox_gp.Kernel`), kernel operators, landmark selection, graphs | pyrox-gp kernels wrap kernellib; new kernel math goes to kernellib, not here. |
| **geonnax** | Deterministic network cores, encoders, basis functions | pyrox-nn wraps a geonnax core and swaps its parameters for sites; new deterministic architecture goes to geonnax. |
| **lineax** | Operators and tags | Dense Grams enter as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`. |
| **matfree**, **optax** | Lanczos (PC priors), L-BFGS (`inla`), optimisers (ensemble MAP) | optax is optional in `pyrox` (`[optax]`): import it lazily (`_require_optax`). |
| **einx**, **jaxtyping** | Axis-naming array ops; shape annotations | Contractions are written with einx (`einx.dot("... i, i o -> ... o", …)`); public arrays are jaxtyping-annotated. |
| **jax** | `jit`, `grad`, `vmap`, PRNG, x64 | Pure functions, explicit keys, no Python control flow on traced values; the test suite runs with x64 on. |

## Reuse before you write

1. **Search the capability index.** [`docs/capabilities.md`](docs/capabilities.md)
   lists every public name in the four packages, grouped by the module that
   defines it, with a one-line summary; then the public API of gaussx,
   kernellib and geonnax. It is generated (`make capabilities`) and checked
   by `packages/pyrox/tests/test_capabilities.py`.
2. **Search the shared helpers** in the table below and the package's
   `AGENTS.md`.
3. **If it is missing, add it at the right level.** Kernel math → kernellib;
   deterministic network cores and basis functions → geonnax; structured
   linear algebra and Gaussian algebra → gaussx; the probabilistic wrapper →
   here, in the lowest package every caller already depends on.
4. **One object, one home.** A name means one thing across the workspace; the
   deliberate exceptions (a pyrox kernel named like the kernellib kernel it
   wraps, …) are listed with their reason in `ALLOWED_SHARED_NAMES` in
   `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| `numpyro.sample` / `numpyro.param` inside a module | `self.pyrox_sample` / `self.pyrox_param` in a `@pyrox_method` (scoped, cached, guarded) |
| A raw `numpyro.factor` / `deterministic` inside a module | name it `self._pyrox_fullname("<site>")` |
| Constrained hyperparameters with priors and per-parameter guides | `Parameterized` (`register_param`, `set_prior`, `autoguide`, `set_mode`, `get_param`) |
| `jnp.linalg.solve` / `cho_solve` / `solve_triangular` / `slogdet` on a covariance | `gaussx.solve`, `cholesky`, `logdet`, `solve_columns`, `cholesky_logdet` on a PSD-tagged operator |
| A Gaussian log-density, KL, whitening, conditioning | `gaussx.gaussian_log_prob`, `gaussian_kl`, `whitened_svgp_predict`, `whiten_covariance`, `conditional` |
| A Kalman filter / RTS smoother / SDE autocovariance | `gaussx.kalman_filter`, `rts_smoother`, `sde_autocovariance`; the `MarkovGPPrior` family here |
| A kernel function | `kernellib.functional`; as a module with priors, the `pyrox_gp` kernels |
| Calling one kernel several times in one model (Gram + diag, several blocks) | `with pyrox_gp._context._kernel_context(kernel):` (`_kernel_contexts` for several) |
| Fourier / spherical-harmonic / Slepian bases, spectral densities, RFF draws | `pyrox_gp._basis` (re-exports `geonnax.basis` + `spectral_density`, RFF draws) |
| A network core | `geonnax`; wrap it with `eqx.tree_at` + sites (see `pyrox_nn._sngp`, `_vssgp`) |
| Flattening `(*batch, D)` around a single-example core | `pyrox_nn._batching.vmap_over_flat_batch` |
| An ensemble of MAP / VI fits, a per-group optimiser | `pyrox.inference` (`ensemble_map`, `ensemble_vi`, `EnsembleMAP`, `EnsembleVI`, `param_group_optimizer`) |
| GMRF precisions (iid, RW1/2, AR1, Besag, BYM2, SPDE) | gaussx GMRF builders; as LGM components, `pyrox_lgm` |
| A dense-reference check in a test | gaussx / `jnp.linalg` on the materialised matrix (fine in tests) |

## The contracts

### 1. Sites: `PyroxModule` owns its sites

- **Subclass `PyroxModule`** and register sites only through
  `self.pyrox_sample(name, prior)` and `self.pyrox_param(name, init, *,
  constraint=, event_dim=)`, inside a method decorated with
  **`@pyrox_method`**. The decorator opens a per-call cache: a site read twice
  in one call is one site; without it NumPyro rejects the duplicate (sample)
  or the trace aliases it (param).
- **Site names are `"<scope>.<name>"`**, where the scope is `pyrox_name` if
  set, else the class name. Two instances of one class in one model need
  distinct `pyrox_name`s, or the trace rejects the duplicate sample site and
  `pyrox_param` raises (under a bare `handlers.seed` the collision is silent).
  Declare it as a field (`pyrox_name: str | None = eqx.field(static=True,
  default=None)` in layers; a fixed default such as `"RBF"` in kernels).
- **Priors are built in the call**, at full shape (`dist.Normal(0, 1)
  .expand([d_in, d_out]).to_event(2)`); a callable prior `(self) -> Distribution`
  expresses a dependent prior. No sampling and no PRNG key at construction for
  pure-prior layers.
- **Raw NumPyro primitives** inside a module (`factor`, `deterministic`) are
  named with `self._pyrox_fullname(...)`; `numpyro.prng_key()` needs a `seed`
  handler.
- **Composable with every handler.** `trace`, `seed`, `substitute`,
  `condition`, `block`, `scope`, `mask`, `scale`, `reparam`, `do`, `lift`,
  `plate`, MCMC, SVI, `Predictive`, `jit` and `vmap` must keep working
  (`packages/pyrox/tests/test_core_numpyro_integration.py`). Under `jit`,
  close over the module (`jax.jit(handlers.seed(model, 0))`) rather than
  passing it as an argument.
- **The private helpers are API across packages.** `_get_context`
  (`pyrox_gp._context`), `_pyrox_scope_name` (`pyrox_gp._multi_output`) and
  `_pyrox_fullname` (`pyrox_nn`) are used downstream: don't rename them.

### 2. `Parameterized`: priors and guides declared once

- **Declare in `setup()`** (called from `__post_init__`; never override
  `__post_init__`, validate in `__check_init__`): `register_param(name,
  value, constraint=)`, then `set_prior`, `autoguide(name, "delta" |
  "normal")`. `set_mode("model")` makes `get_param` sample the prior,
  `set_mode("guide")` the guide, so the module is its own guide.
- **The registry is not in the pytree.** It lives in a class-level registry
  keyed by `id(self)`, so anything that rebuilds the module (`eqx.tree_at`,
  `eqx.apply_updates`, passing it through `filter_jit`, flatten / unflatten,
  checkpoint load) yields a copy with an empty registry and a `KeyError`.
  Keep the original instance, or close over it.
- **Reserved names.** Guide sites use `<name>_loc` / `<name>_scale`; a user
  param with that name raises.
- Structural settings (`nu`, `degree`, `input_dim`) are fields, not
  registered params.

### 3. Numerics: JAX, gaussx and the kernel context

- **Pure and explicit.** Arrays and modules in, arrays and modules out; keys
  are explicit (`jr.key`, split before reuse); no Python control flow on
  traced values (the eager `fit` loops of the non-Gaussian inference
  strategies are the documented exception).
- **Linear algebra through gaussx.** Wrap Grams as PSD-tagged lineax
  operators and call gaussx; respect the model's `solver=`. A few modules
  still hand-roll `cho_solve` / `solve_triangular`
  (`pyrox_gp._inference_nongauss`, `pyrox_gp._latent_factor`); new code adds
  none, and a change that touches one is a chance to convert it.
- **One draw per model call.** Evaluate a kernel with priors more than once
  only inside `_kernel_context(kernel)`, or the second call resamples
  (`seed`) or collides (`trace`).
- **Dtypes.** The suite runs with x64 on (root `conftest.py`); build
  constants with the input's dtype so float32 callers stay float32.

### The public API

- Export a public name from its package `__init__.py` (import + `__all__`),
  add a `::: package.Name` entry to its page in `docs/api/` (no test enforces
  this: check it), and run `make capabilities`.
- Google-style docstrings with jaxtyping shapes, the model in MathJax
  (`$…$`, `$$…$$`), **the site names the module registers**, and an
  `Examples:` block. No Sphinx / RST markup (`:math:`, `.. math::`,
  `:class:` roles, `.. note::` directives): `tests/**/test_docstrings.py` in
  pyrox, pyrox-gp and pyrox-nn scan the source. Doctests run only in
  pyrox-lgm (`test_doctests.py`); elsewhere examples are illustrative, but
  keep them correct.
- Breaking changes go through a deprecation (a `DeprecationWarning` naming
  the replacement; see `pyrox_gp._src.kernels`) before removal.

## What enforces them

| Test | Enforces |
|---|---|
| `packages/pyrox/tests/test_core_numpyro_integration.py` | The bridge under every NumPyro handler, MCMC, SVI, `Predictive`, `jit`, `vmap`, `grad` |
| `packages/pyrox/tests/test_core_pyrox_module.py`, `test_core_parameterized.py` | Site naming, the per-call cache, the duplicate-param guard, `Parameterized` priors / guides / rebuild errors |
| `packages/pyrox/tests/test_pyrox.py` | `pyrox._core`'s exports and types |
| `packages/pyrox/tests/test_capabilities.py` | `docs/capabilities.md` is current; no name bound to two objects |
| `tests/**/test_docstrings.py` (pyrox, pyrox-gp, pyrox-nn) | No Sphinx / RST markup in source |
| `packages/pyrox-lgm/tests/test_import_guard.py` | pyrox-lgm never imports pyrox-gp |
| `packages/pyrox-lgm/tests/test_public_api.py`, `test_doctests.py` | lgm `__all__` names exist, no duplicates; lgm doctests run |
| `packages/pyrox-lgm/tests/test_rinla_fixtures.py` | `inla()` against R-INLA reference fixtures (slow) |
| `packages/pyrox-gp/tests/gp/test_src_kernels_deprecation.py` | The deprecated `pyrox_gp._src.kernels` shim |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add a Bayesian / uncertainty-aware layer (pyrox-nn) | `add-bayesian-layer` |
| Add a GP kernel (pyrox-gp) | `add-kernel` |
| Add a guide, likelihood, inducing features, inference strategy or multi-output kernel (pyrox-gp) | `add-gp-component` |
| Add an LGM component, PC prior or observation model (pyrox-lgm) | `add-lgm-component` |
| Change `PyroxModule`, `Parameterized` or `pyrox.inference` | `change-core-bridge` |
| Bump gaussx / kernellib / geonnax | `bump-geoml-deps` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `pyrox-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `model-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

Downstream users get pyrox's guidance through the Claude Code plugin in
`plugins/pyrox/` (published by `.claude-plugin/marketplace.json`) and
`docs/llms.txt`; see `docs/agents.md`. When the public API or the modelling
patterns change, update `plugins/pyrox/skills/bayesian-models-with-pyrox/`
too: `packages/pyrox/tests/test_plugin_skill.py` runs its worked example
(slow tier) and checks that every pyrox name it and the plugin reviewer
mention still exists.

## Working in the repo

Always run Python tools through `uv run` (never the system Python).

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test                 # every test, no coverage (uv run pytest -v -o addopts=)
make test-cov             # every test with coverage (fail_under = 80)
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check on all four package src dirs
make capabilities         # regenerate docs/capabilities.md
make docs-serve           # local MkDocs preview
```

Run tests from the repo root (the root `conftest.py` and pytest config apply).
`addopts` turns coverage on, and a subset never reaches the 80 % gate, so pass
`--no-cov` (or `-o addopts=`) when running part of the suite:

```bash
uv run pytest --no-cov packages/pyrox-gp/tests/gp/test_kernel_classes.py -v
```

### Test tiers

- CI ("Tests", `ci.yml`) runs `uv run pytest -v -m "not slow"` on Python 3.12
  and 3.13, with coverage gated at 80 % across all four packages.
- `@pytest.mark.slow` marks convergence, NUTS / SVI and dense-equivalence
  sweeps. **No workflow runs them**, so run the slow tests of what you touched
  locally (`uv run pytest --no-cov -m slow packages/<pkg>/tests/...`).
- Keep one unmarked, tiny smoke test per feature so the fast tier still
  exercises it.
- The root `conftest.py` enables x64 for the whole session and turns
  `GaussxDeprecationWarning` into an error.
- Seed NumPyro with `handlers.seed(rng_seed=0)` and assert site names under
  `handlers.trace()`; keys come from `jr.key(n)`; compare against a dense or
  closed-form reference, and say in a comment where a tolerance comes from.

### Before every commit

All of these must pass, from the repo root:

1. `uv run pytest -m "not slow"` (what CI runs, with the coverage gate), plus
   the slow tests of what you touched.
2. `uv run --group lint ruff check .` — the **entire** repo, which includes
   every package's `tests/` and `scripts/`. Never lint a subdirectory.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (all four packages; CI checks the same four). In an
   environment with every extra installed (`make install` plus extras), ty
   reports three unused `ty: ignore` comments on optional imports
   (`gauss_flows`, `xarray`); CI's environment does not have them, so those
   three are expected locally.
5. After changing a public API: `make capabilities`, and the `docs/api` page.
6. After changing a dependency: `uv lock`, and commit `uv.lock`.
7. After changing docs: `make docs` (it must build; CI deploys from `main`).

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change; remove
   only what your change made unused.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced
   bug, a dense reference to match) and loop until it passes.

Also: Python 3.12+, `from __future__ import annotations` where the module
uses it, type hints on every public function, `eqx.Module` (not dataclasses)
for anything that flows through JAX.

## Package rules

Each package's `AGENTS.md` holds its layout, extension points and the tests
that enforce them. Read it before changing that package:

- [`packages/pyrox/AGENTS.md`](packages/pyrox/AGENTS.md) — the bridge and ensemble inference
- [`packages/pyrox-gp/AGENTS.md`](packages/pyrox-gp/AGENTS.md) — kernels, guides, likelihoods, inference, the kernel context
- [`packages/pyrox-nn/AGENTS.md`](packages/pyrox-nn/AGENTS.md) — Bayesian layers on geonnax cores, the `[bnf]` estimator
- [`packages/pyrox-lgm/AGENTS.md`](packages/pyrox-lgm/AGENTS.md) — components, PC priors, `inla()`, R-INLA fixtures

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to ("push to
  main", "merge to main"). Work on a feature branch, commit locally, and push
  only when asked. "Merge the branch" means push the feature branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(gp): add …`); CI validates PR titles. Types:
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
  `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please per package (`pyrox-gp-vX.Y.Z`, …, in
  one combined release PR); don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** Read it
  first; only append checklist items or update their status.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).
- Issues follow the label taxonomy and epic model in
  [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MkDocs + Material + mkdocstrings + mkdocs-jupyter, one site for the workspace
(`mkdocs.yml`); `pages.yml` deploys on every push to `main`.

- **API pages** (`docs/api/*.md`, `docs/api/nn/*.md`) list members by hand,
  one `::: package.Name` per symbol under prose headings.
- **Notebooks** in `docs/notebooks/` are committed as **executed `.ipynb`
  only**; the jupytext `.py` you author from is a local artefact
  (`docs/notebooks/*.py` is gitignored). Figures render inline with
  `plt.show()`: no `savefig`, no separate image files (the
  `docs/images/readme/` PNGs are README figures). Full standards:
  `.github/instructions/docs-examples.instructions.md`.
- **Diagrams and icons** come from `docs/assets/render.py`
  (`uv run --no-project python docs/assets/render.py`).
- **Design references** live in `design_docs/pyrox/`; parts of it predate the
  workspace split (single-package layout, older pattern names), so the code
  and this file win where they disagree.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never committed);
track work in GitHub issues.
