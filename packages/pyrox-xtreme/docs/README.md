# Documentation build

This project's documentation is built by **two tools**, deployed as one site.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, example notebooks, tutorials, design docs | [mystmd](https://mystmd.org) | `docs/*.md`, `docs/notebooks/`, `docs/tutorials/`, `docs/design_docs/` (toc in `docs/myst.yml`) | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

Build both and assemble them with:

```bash
npm install -g mystmd   # once: mystmd is a Node CLI, not a uv dependency
make docs               # build + assemble into public/
make docs-serve         # assemble, then serve public/ locally
make docs-check         # strict validation only, no themed HTML render
make docs-api           # MkDocs API reference alone (mkdocs build --strict)
```

Notebooks under `docs/notebooks/` and `docs/tutorials/` are committed
**pre-executed**; the docs build renders their stored outputs and never runs
them. A new notebook only appears on the site once it is listed in the `toc`
of `docs/myst.yml`.

## URLs are flat

mystmd derives a page's URL from its file basename (minus any leading `NN_`
ordering prefix, with `_` turned into `-`), not from its directory. So
`docs/design_docs/vision.md` is served at `/vision/`, and two files that share
a basename anywhere in the prose tree collide. Keep basenames unique — that is
why the design docs are named `api_primitives.md` / `examples_primitives.md`
rather than two `primitives.md`. Equation and section labels (`(label)=`,
`:label:`) are likewise project-global, so keep those unique too.

## Why the split

mystmd is much better at prose: real cross-references, first-class notebook
execution, exports to PDF/LaTeX, and MyST's directive syntax. What it does not
have is an autodoc equivalent — there is no mature way to render Python
docstrings into a MyST site today.

MkDocs + mkdocstrings already does that well, and it publishes a
Sphinx-compatible `objects.inv`, which is exactly what mystmd needs to
cross-reference *into* it. So each tool does the half it is good at.

## Site-nav URLs must be absolute

The theme re-renders the site nav from the config it embeds for hydration, and
prepends `BASE_URL` to any nav URL starting with `/`. A nav entry that already
carries the deployment prefix is therefore doubled —
`/xtremax/xtremax/reference/` — and 404s.

This bites only *after* hydration: the server-rendered HTML is correct, so
`curl` and `verify_links` both see a healthy link. Keep `site.nav` URLs
absolute (mystmd rejects `/reference/` and `reference/` anyway) and never
rewrite them at build time. `nav_base_url_problems` in
`scripts/build_docs.py` fails the build if one becomes root-relative.

The trade-off is that a local preview's nav button points at the deployed
site. In-page `xref:` links are unaffected — the theme does not re-prefix
those, so they are rewritten to `{BASE_URL}/reference/...` as normal.

## If the theme download is blocked

`myst build --html` fetches the site template as a zip from GitHub. Behind a
corporate proxy or a restrictive egress policy that request can fail with a
403 while ordinary git access still works. Clone the template and point at it
locally instead:

```bash
git clone --depth 1 https://github.com/myst-templates/book-theme.git /tmp/book-theme
```

Then set `site.template` in `docs/myst.yml` to `/tmp/book-theme` for that
build. Everything else is unchanged.

Note that the rendered pages still load KaTeX, Font Awesome, and
jupyter-matplotlib stylesheets from CDNs at view time. Without network access
in the *browser*, maths renders doubled — KaTeX ships an accessibility MathML
copy that its stylesheet is responsible for hiding. That is a viewing
artefact, not a build problem.

## Cross-references from prose into the API

In any MyST page, link to an API object with the `xref:` protocol and the
object's full dotted path, as it appears on its API page:

```markdown
[`temporal_block_maxima`](xref:api#xtremax.extraction.temporal_block_maxima)
[`GeneralizedExtremeValueDistribution`](xref:api#xtremax.distributions.GeneralizedExtremeValueDistribution)
```

A target that does not exist in the inventory fails `myst build --strict`, so
broken API links are caught on the pull request rather than in production.

## Known upstream issue: the `$` anchor abbreviation

A Sphinx inventory may record an object's anchor as the literal `$`, meaning
"the anchor is the object's own name". mystmd 1.11.0 **lowercases the name
when it expands that abbreviation**, which breaks links into mkdocstrings'
case-sensitive anchors:

| Inventory entry | mystmd emits | Correct? |
|---|---|---|
| `xtremax.distributions.GumbelType1GEVD` -> `distributions/#$` | `#xtremax.distributions.gumbeltype1gevd` | **no** |
| `xtremax.extraction.temporal_block_maxima` -> `extraction/#$` | `#xtremax.extraction.temporal_block_maxima` | yes (already lower case) |

The failure is silent: the link resolves, the page loads, and the browser
simply cannot find the anchor.

**This does bite in xtremax.** Every object documented under its canonical
path uses `$`, and any link to a class or other mixed-case name comes out
lowercased. Two safety nets handle it in
`scripts/build_docs.py`:

- `restore_anchor_case` repairs the lowercased anchors after the build, using
  the inventory as the source of truth. Delete it once mystmd fixes the
  expansion upstream.
- `verify_links` checks that **every** internal link in the assembled site
  resolves to a file that exists and, when it carries a fragment, to an anchor
  that is really in that file. It sees both generators' output at once, so it
  catches this class of bug regardless of cause — keep it either way.

Both operate on the **static** HTML. Anything the theme re-renders on
hydration — the site nav above being the case that actually bit us — is
invisible to them, which is why that one needs its own check.
