---
name: add-notebook
description: Add or update an executed example notebook in xtremax's docs — jupytext percent .py while drafting, an executed .ipynb as the committed source, a toc entry in docs/myst.yml, xref links into the API reference, ruff on the notebook cells, and a strict two-tool docs build. Use when asked to write, extend, re-run or fix an example notebook under docs/notebooks or a tutorial under docs/tutorials.
---

# Add or update a notebook

The full standard is `.github/instructions/docs-examples.instructions.md`;
read it first. This is the checklist.

## 1. Plan it

- One question per notebook; check the existing ones in `docs/notebooks/`
  (`evt_*` for the EVT families, `<family>_from_scratch_and_numpyro` for
  each point process) and the toc in `docs/myst.yml`, so you extend rather
  than duplicate.
- The basename must be unique across the whole prose tree (`docs/`,
  `docs/notebooks/`, `docs/tutorials/`, `docs/design_docs/`): mystmd builds
  flat URLs from basenames, minus a leading `NN_` prefix, underscores
  becoming hyphens (`evt_gpd_pot.ipynb` → `/evt-gpd-pot/`), and a clash
  gets an unstable suffix.
- Use the public API (`import xtremax`, `from xtremax.distributions import
  …`) the way a user would; no private names. Fitting loops use NumPyro
  (and the docs group's optax / diffrax where a notebook needs them); the
  library itself never does.
- A longer real-data curriculum belongs in `docs/tutorials/` with its
  helper code in the `_helpers/` folder next to the notebooks; those
  notebooks need the `tutorials` dependency group (and CDS data) to
  re-run, and ruff skips their `.ipynb` (`extend-exclude` in
  `pyproject.toml`).

## 2. Draft in `.py`, commit the `.ipynb`

1. Write `docs/notebooks/<name>.py` in jupytext percent format (header in
   the instructions file); first markdown cell: title + Colab badge
   pointing at `jejjohnson/xtremax`.
2. Smoke-run: `uv run --group docs python docs/notebooks/<name>.py`.
3. Convert and execute:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/<name>.py
   uv run --group docs jupyter nbconvert --to notebook --execute \
     docs/notebooks/<name>.ipynb --inplace --ExecutePreprocessor.timeout=180
   ```

4. Delete the `.py`; the executed `.ipynb` is the source of truth. To fix
   something later, regenerate the `.py` (`jupytext --to py:percent`),
   edit, re-convert and **re-execute**; never hand-edit committed source
   cells without re-running them.

## 3. Content rules

- Figures inline with `plt.show()`: no `savefig`, no committed PNGs.
  Pre-commit's `check-added-large-files` hook rejects new files over its
  500 kB default, so keep figures few and small (`dpi` around 100).
- Link API names with ``[`temporal_block_maxima`](xref:api#xtremax.extraction.temporal_block_maxima)``
  — a missing target fails the strict build.
- Random draws from explicit keys (`jax.random.PRNGKey(0)` as the existing
  notebooks do, split per use) so a re-run reproduces the outputs.
- ruff lints and formats the `.ipynb` code cells (88 columns), with the
  per-file ignores for `docs/notebooks/*.ipynb` in `pyproject.toml` (E501
  for long URLs, RUF001 for typographic dashes, B905).
- MyST maths: `\lambda_{\max}`, not `\lambda_\max`.

## 4. Wire it in and verify

- Add `- file: notebooks/<name>.ipynb` under the right section of the
  `toc` in `docs/myst.yml`.
- `uv run --group lint ruff format docs/notebooks/` and
  `uv run --group lint ruff check .`
- `make docs-check` (strict validation of both halves) and `make docs`
  (builds, assembles `public/` and checks every internal link) both need
  the mystmd CLI (`npm install -g mystmd`); if the theme download is
  blocked, follow "If the theme download is blocked" in `docs/README.md`
  and restore `docs/myst.yml` afterwards. Without mystmd, run
  `make docs-api` and say in the PR that the prose half was not checked.
