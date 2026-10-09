---
name: add-notebook
description: Add or update an example notebook in pyrox's docs — authored as a local jupytext .py, executed into the committed .ipynb (the only file committed), with the Colab install cell, watermark and nav entry. Use when asked to write a tutorial, example, demo, walkthrough or benchmark notebook, or after changing an API a notebook uses.
---

# Add an example notebook

The full standards are `.github/instructions/docs-examples.instructions.md`;
this is the workflow. The committed artefact is the **executed `.ipynb`
only**: `docs/notebooks/*.py` is gitignored, and there are no separate image
files (figures render inline with `plt.show()`).

## 1. Plan it

- One question per notebook, using the public API (`pyrox._core`,
  `pyrox.inference`, `pyrox_gp`, `pyrox_nn`, `pyrox_lgm`); check the existing
  notebooks in `mkdocs.yml` ("Examples", "Tutorials") so you extend one
  rather than duplicate it.
- Show the model and its sites (the trace's site names), fit it (SVI / NUTS /
  `inla`), and check the answer against the truth or a reference.

## 2. Author the source locally (`docs/notebooks/<name>.py`)

- The jupytext percent header, `# %%` / `# %% [markdown]` cells; each
  markdown paragraph on **one** line (soft wraps render as breaks).
- First markdown cell: title + Colab badge. First code cell: Colab
  detection and a per-package install
  (`"pyrox-gp[colab] @ git+https://github.com/jejjohnson/pyrox@main#subdirectory=packages/pyrox-gp"`,
  plus every other pyrox package it imports).
- `jax.config.update("jax_enable_x64", True)`, the IProgress warning
  filter, the `%watermark` readout cell, matplotlib defaults only.
- Smoke-run while iterating: `uv run --group docs python docs/notebooks/<name>.py`.

## 3. Convert, execute, delete the source

```bash
uv run --group docs jupytext --to notebook docs/notebooks/<name>.py
uv run --group docs jupyter nbconvert --to notebook \
  --execute docs/notebooks/<name>.ipynb --inplace \
  --ExecutePreprocessor.timeout=180
rm docs/notebooks/<name>.py
```

Commit the `.ipynb`. To change an existing notebook, regenerate the `.py`
(`jupytext --to py:percent`), edit, and repeat the three steps.

## 4. Publish it

- Add it to `nav` in `mkdocs.yml` ("Examples" or "Tutorials").
- `make docs` must build (mkdocs-jupyter renders the executed notebook with
  `execute: false`; the build is slow, give it several minutes).
