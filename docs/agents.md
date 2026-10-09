# Building with agents

pyrox exists so that a model's random parameters live in the module that
uses them, with stable site names, under every NumPyro handler. Coding
agents tend to re-implement what they cannot see — a `numpyro.sample` wired
into an Equinox module by hand, a GP marginal likelihood written out with
`cho_solve`, a deep-ensemble loop — and each one loses scoping, caching or
structure. pyrox ships three things that let them find it.

## The capability index

The [capability index](capabilities.md) lists every public name in the four
packages, grouped by the module that defines it, with a one-line summary;
then the public API of gaussx, kernellib and geonnax, which pyrox builds on.
It is regenerated from the code and checked in the test suite, so it never
drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/pyrox
/plugin install pyrox@pyrox
```

The plugin adds:

- **`bayesian-models-with-pyrox`** (skill) — loads whenever a task puts
  priors on an Equinox module's parameters, writes a GP or Bayesian NN in
  NumPyro, or fits a latent Gaussian model: which package does what, the
  three modelling patterns, the rules (sites through the bridge, unique
  scopes, don't rebuild a `Parameterized` module), a worked example, and a
  "don't write it — use pyrox" table.
- **`pyrox-reuse-reviewer`** (subagent) — a read-only check of a diff for
  modelling code pyrox already provides, and for bridge misuse.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/pyrox/llms.txt): a curated map of
pyrox and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on pyrox:

```markdown
## Probabilistic models: build on pyrox

This project uses pyrox (Equinox ↔ NumPyro bridge), pyrox-gp (GPs),
pyrox-nn (Bayesian layers) and pyrox-lgm (INLA). Before writing a module
with random parameters, a GP likelihood or prediction, a Bayesian layer, an
ensemble loop or a latent Gaussian model, search the capability index
(https://jejjohnson.github.io/pyrox/capabilities/) or the packages'
`__all__`, and compose what exists:

- Inside an Equinox module, register sites with `self.pyrox_sample` /
  `self.pyrox_param` in a method decorated with `@pyrox_method`
  (`from pyrox._core import PyroxModule, pyrox_method`), never
  `numpyro.sample` directly.
- Give two instances of one class in one model distinct `pyrox_name`s.
- Declare constrained parameters with priors and guides once, in a
  `Parameterized.setup()`; don't rebuild such a module with `eqx.tree_at` or
  pass it as a `jit` argument (close over it instead).
- GPs: `pyrox_gp.GPPrior` + `gp_factor` / `.condition(...).predict(...)`,
  the `pyrox_gp` kernels with `set_prior`; never `cho_solve` a Gram by hand.
- Check every model's site names under
  `numpyro.handlers.trace(numpyro.handlers.seed(model, 0))`, and that it
  recovers known parameters on simulated data.
```

## Working on pyrox itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/pyrox/blob/main/AGENTS.md) in the
repository and the `AGENTS.md` of each package: the package map, what pyrox
is built on, "reuse before you write", the contracts and the tests that
enforce them, and recipe skills for adding layers, kernels, GP and LGM
components.
