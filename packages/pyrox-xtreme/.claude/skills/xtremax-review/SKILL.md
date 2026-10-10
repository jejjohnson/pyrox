---
name: xtremax-review
description: Review a change or pull request in xtremax against the repo's own rules — CODE_REVIEW.md, the primitive / EVT distribution / point-process / extraction contracts in AGENTS.md, the boundaries with NumPyro, pipekit and scikit-learn, reuse of existing primitives, and EVT numerics. Use when asked to review a diff, branch or PR in this repo.
---

# Review an xtremax change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which layers it touches (primitives, distributions, point-process
   primitives / operators / wrappers, extraction, simulations, docs).
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: hand-written
   densities, quantiles, return levels, compensators, samplers, thinning
   loops, quadrature or GOF statistics are the main way xtremax grows a
   second copy of something it already has.
3. **Numerics** — run the `distribution-reviewer` subagent in parallel with
   step 2: the ξ → 0 limit, tails and supports, the NumPyro contract,
   padded point-process buffers, keys, gradients and float32.
4. **Boundaries** — no pipekit import in `src/xtremax/`; no hand-written
   inference (NumPyro owns it); scikit-learn only behind the `[threshold]`
   guard; GP priors left to a GP library; layers point one way.
5. **Contracts** — for each layer touched, the rules in `AGENTS.md`'s
   contracts: maths in the primitives with the classes as thin routers, the
   `concrete_sign` support, `check_prng_key`, contiguous masks and typed
   sample results, `GoodnessOfFitMixin` / `effective_lambda_max` /
   `SeparableMarkedPP`, xarray in and out.
6. **Checklist** — the rest of `CODE_REVIEW.md` (public API, `members:`
   entries, capability index, docstrings, tests against scipy oracles,
   lane markers), skipping what ruff, ty or the tests already enforce.
7. **Verify claims** — for anything you flag as a bug, run it: the value
   against `scipy.stats` (mind `genextreme(c=-ξ)`), `jax.grad` at ξ = 0 and
   at a support endpoint, a float32 input, `biject_to(d.support)`, a padded
   buffer with trailing slots, `jax.jit` around the call.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
