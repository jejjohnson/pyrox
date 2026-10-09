"""Render the pyrox README assets: icons, logos and the two diagrams.

Every output is a plain SVG written next to this file. The icons share the
gaussx tile (a 120-unit rounded square, r = 26, white glyph) in the pyrox
ember gradient; each package gets its own glyph:

    pyrox      plate notation (latent node over an observed node)
    pyrox-gp   GP mean with a credible band and data
    pyrox-nn   a small network with a distribution over its weights
    pyrox-lgm  a GMRF lattice with one node's neighbourhood ringed

Diagrams come in a light and a dark variant for GitHub's
``<picture>`` / ``prefers-color-scheme`` switch.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

from pathlib import Path


OUT = Path(__file__).parent

DOT = " \N{MIDDLE DOT} "
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Ember: the pyrox gradient and its wordmark accents.
C1, C2 = "#F97316", "#E11D48"
ACCENT = {"light": "#C2410C", "dark": "#FDBA74"}

THEME = {
    "light": {
        "text": "#1F2937",
        "muted": "#6B7280",
        "card": "#F9FAFB",
        "line": "#D1D5DB",
        "arrow": "#9CA3AF",
        "hi_fill": "#FFF7ED",
        "hi_line": "#EA580C",
        "site": "#C2410C",
        "band": "#F3F4F6",
    },
    "dark": {
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "card": "#111827",
        "line": "#374151",
        "arrow": "#6B7280",
        "hi_fill": "#2A1408",
        "hi_line": "#EA580C",
        "site": "#FDBA74",
        "band": "#1F2937",
    },
}

W = "#FFFFFF"


# ---------------------------------------------------------------- glyphs
# Each glyph draws on the 128-unit icon canvas; the tile spans 4..124.


def glyph_plate() -> str:
    """Latent θ above a plate holding an observed node: the core bridge."""
    return (
        f'<rect x="24" y="54" width="80" height="56" rx="10" fill="none" '
        f'stroke="{W}" stroke-width="4.5"/>'
        f'<circle cx="64" cy="31" r="12" fill="none" stroke="{W}" stroke-width="4.5"/>'
        f'<path d="M64 44 L64 63" stroke="{W}" stroke-width="4" '
        f'stroke-linecap="round"/>'
        f'<path d="M58 58 L64 65 L70 58" fill="none" stroke="{W}" '
        f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="64" cy="82" r="12" fill="{W}"/>'
    )


def glyph_band() -> str:
    """GP mean, its credible band and four observations."""
    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="4.5" fill="{W}"/>'
        for x, y in [(30, 74), (50, 63), (78, 78), (98, 59)]
    )
    return (
        '<path d="M18 70 C34 46 48 40 64 52 C80 64 92 46 110 36 L110 70 '
        'C92 82 80 96 64 86 C48 76 34 84 18 100 Z" '
        f'fill="{W}" fill-opacity="0.28"/>'
        '<path d="M18 85 C34 66 48 58 64 69 C80 80 92 64 110 53" fill="none" '
        f'stroke="{W}" stroke-width="4.5" stroke-linecap="round"/>' + dots
    )


def glyph_network() -> str:
    """A 3-2-1 network under a bell curve: a prior over the weights."""
    nodes = [(32, 48), (32, 72), (32, 96), (64, 60), (64, 84), (96, 72)]
    edges = (
        "M32 48 L64 60 M32 48 L64 84 M32 72 L64 60 M32 72 L64 84 "
        "M32 96 L64 60 M32 96 L64 84 M64 60 L96 72 M64 84 L96 72"
    )
    return (
        '<path d="M44 32 C54 32 58 18 64 18 C70 18 74 32 84 32" fill="none" '
        f'stroke="{W}" stroke-width="4" stroke-linecap="round"/>'
        f'<path d="{edges}" stroke="{W}" stroke-opacity="0.55" '
        'stroke-width="3" stroke-linecap="round"/>'
        + "".join(f'<circle cx="{x}" cy="{y}" r="7" fill="{W}"/>' for x, y in nodes)
    )


def glyph_lattice() -> str:
    """A 3-by-3 GMRF lattice with the centre node's neighbourhood ringed."""
    grid = (
        "M36 36 L92 36 M36 64 L92 64 M36 92 L92 92 "
        "M36 36 L36 92 M64 36 L64 92 M92 36 L92 92"
    )
    nodes = [(x, y) for y in (36, 64, 92) for x in (36, 64, 92) if (x, y) != (64, 64)]
    return (
        f'<path d="{grid}" stroke="{W}" stroke-opacity="0.6" stroke-width="3.5" '
        'stroke-linecap="round"/>'
        + "".join(f'<circle cx="{x}" cy="{y}" r="7" fill="{W}"/>' for x, y in nodes)
        + f'<circle cx="64" cy="64" r="13" fill="none" stroke="{W}" stroke-width="3"/>'
        + f'<circle cx="64" cy="64" r="7" fill="{W}"/>'
    )


GLYPHS = {
    "pyrox": glyph_plate,
    "pyrox-gp": glyph_band,
    "pyrox-nn": glyph_network,
    "pyrox-lgm": glyph_lattice,
}


def tile(gid: str, glyph: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with ``glyph`` on it, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph}</g>"
    )


def svg(width: int, height: int, label: str, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{title}</title>\n{body}\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    weight: int = 400,
    fill: str,
    family: str = SANS,
    anchor: str = "start",
) -> str:
    # xml:space keeps leading spaces, so indented code lines stay indented.
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'xml:space="preserve">{s}</text>'
    )


def arrow_defs(mid: str, colour: str) -> str:
    return (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker></defs>'
    )


def line(points: list[tuple[float, float]], colour: str, head: str | None) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    end = f' marker-end="url(#{head})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round"{end}/>'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    t: dict[str, str],
    *,
    highlight: bool = False,
) -> str:
    fill, stroke, sw = (
        (t["hi_fill"], t["hi_line"], 2) if highlight else (t["card"], t["line"], 1)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


# ---------------------------------------------------------------- icons and logo


def write_icons() -> None:
    for name, glyph in GLYPHS.items():
        stem = "icon" if name == "pyrox" else f"icon-{name.removeprefix('pyrox-')}"
        body = tile(f"{stem}-g", glyph())
        (OUT / f"{stem}.svg").write_text(svg(128, 128, name, name, body))


def write_logos() -> None:
    for mode, t in THEME.items():
        body = tile(f"logo-{mode}", glyph_plate(), x=6, y=6) + (
            f'<text x="154" y="92" font-family="{SANS}" font-size="72" '
            f'font-weight="700" letter-spacing="-1.5" fill="{t["text"]}">'
            f'pyro<tspan fill="{ACCENT[mode]}">x</tspan></text>'
        )
        (OUT / f"logo-{mode}.svg").write_text(svg(400, 140, "pyrox", "pyrox", body))


# ---------------------------------------------------------------- hero


ENGINES = [
    ("handlers.seed" + DOT + "trace", "inspect every site"),
    ("MCMC(NUTS(model))", "sample the posterior"),
    ("SVI(model, AutoNormal)", "fit a guide"),
    ("Predictive", "posterior predictive draws"),
    ("jit" + DOT + "vmap" + DOT + "grad", "plain JAX transforms"),
]


def write_hero() -> None:
    """One PyroxModule fanning out to every NumPyro handler and engine."""
    width, height = 960, 356
    for mode, t in THEME.items():
        head = f"hero-{mode}-head"
        parts = [arrow_defs(head, t["arrow"])]

        # Module card, centred on the fan-out trunk.
        mx, my, mw, mh = 20, 88, 340, 180
        parts.append(card(mx, my, mw, mh, t, highlight=True))
        parts.append(tile(f"hero-{mode}-icon", glyph_plate(), mx + 18, my + 16, 36))
        parts.append(
            text(
                mx + 64,
                my + 40,
                "BayesianLinear(PyroxModule)",
                size=14.5,
                weight=700,
                fill=t["text"],
                family=MONO,
            )
        )
        code = [
            "@pyrox_method",
            "def __call__(self, x):",
            '    W = self.pyrox_sample("weight", prior)',
            '    b = self.pyrox_param("bias", zeros)',
        ]
        for i, s in enumerate(code):
            parts.append(
                text(mx + 18, my + 74 + 19 * i, s, size=12, fill=t["text"], family=MONO)
            )
        parts.append(
            text(
                mx + 18,
                my + 162,
                "sites: BayesianLinear.weight, .bias",
                size=12.5,
                weight=600,
                fill=t["site"],
                family=MONO,
            )
        )

        # Trunk, then one arrow into each engine card.
        cy = my + mh / 2
        trunk_x, ex, ew, eh, gap = 410, 470, 470, 52, 14
        centres = [20 + eh / 2 + i * (eh + gap) for i in range(len(ENGINES))]
        parts.append(line([(mx + mw, cy), (trunk_x, cy)], t["arrow"], None))
        parts.append(
            line([(trunk_x, centres[0]), (trunk_x, centres[-1])], t["arrow"], None)
        )
        for c in centres:
            parts.append(line([(trunk_x, c), (ex - 3, c)], t["arrow"], head))

        for (name, what), c in zip(ENGINES, centres, strict=True):
            parts.append(card(ex, c - eh / 2, ew, eh, t))
            parts.append(
                text(
                    ex + 18,
                    c + 5,
                    name,
                    size=14,
                    weight=700,
                    fill=t["text"],
                    family=MONO,
                )
            )
            parts.append(
                text(
                    ex + ew - 18, c + 5, what, size=13.5, fill=t["muted"], anchor="end"
                )
            )

        label = (
            "One PyroxModule with named sample and param sites runs unchanged "
            "under NumPyro handlers, NUTS, SVI, Predictive and JAX transforms."
        )
        (OUT / f"hero-{mode}.svg").write_text(
            svg(
                width,
                height,
                label,
                "One module, every inference engine",
                "\n".join(parts),
            )
        )


# ---------------------------------------------------------------- layers


PACKAGES = {
    # name: (x, y, w, description, extra dependencies)
    "pyrox-nn": (
        30,
        20,
        400,
        "Bayesian layers" + DOT + "SIREN" + DOT + "SNGP" + DOT + "BNF",
        "+ geonnax",
    ),
    "pyrox-gp": (
        30,
        160,
        400,
        "kernels" + DOT + "guides" + DOT + "likelihoods" + DOT + "GP models",
        "+ gaussx" + DOT + "kernellib" + DOT + "geonnax" + DOT + "flowjax",
    ),
    "pyrox-lgm": (
        530,
        90,
        400,
        "GMRF components" + DOT + "PC priors" + DOT + "inla()",
        "+ gaussx" + DOT + "kernellib" + DOT + "matfree",
    ),
    "pyrox": (
        270,
        300,
        420,
        "PyroxModule" + DOT + "Parameterized" + DOT + "ensemble MAP / VI",
        "+ jax" + DOT + "equinox" + DOT + "numpyro",
    ),
}


def write_layers() -> None:
    """The four packages; arrows point at what each one imports."""
    width, height, h = 960, 484, 92
    for mode, t in THEME.items():
        head = f"layers-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]

        # nn → gp, gp → pyrox, lgm → pyrox (pyrox-lgm never imports pyrox-gp).
        parts.append(line([(230, 112), (230, 157)], a, head))
        parts.append(line([(230, 252), (230, 346), (267, 346)], a, head))
        parts.append(line([(730, 182), (730, 346), (693, 346)], a, head))

        for name, (x, y, w, desc, deps) in PACKAGES.items():
            core = name == "pyrox"
            parts.append(card(x, y, w, h, t, highlight=core))
            parts.append(
                tile(f"layers-{mode}-{name}", GLYPHS[name](), x + 16, y + 22, 48)
            )
            parts.append(
                text(
                    x + 80,
                    y + 32,
                    name,
                    size=16,
                    weight=700,
                    fill=t["site"] if core else t["text"],
                    family=MONO,
                )
            )
            parts.append(text(x + 80, y + 56, desc, size=13, fill=t["text"]))
            parts.append(text(x + 80, y + 77, deps, size=12.5, fill=t["muted"]))

        parts.append(
            f'<rect x="20" y="420" width="920" height="44" rx="10" fill="{t["band"]}"/>'
        )
        for x, head_word, names in [
            (44, "Foundations", ["jax", "equinox", "numpyro", "einx"]),
            (500, "GeoML", ["gaussx", "kernellib", "geonnax", "matfree", "flowjax"]),
        ]:
            parts.append(
                f'<text x="{x}" y="447" font-family="{SANS}" font-size="13" '
                f'fill="{t["muted"]}"><tspan font-weight="700" fill="{t["text"]}">'
                f"{head_word}</tspan>  {DOT.join(names)}</text>"
            )

        label = (
            "pyrox-nn builds on pyrox-gp, which builds on pyrox; pyrox-lgm builds "
            "on pyrox directly. All sit on JAX, Equinox and NumPyro and on the "
            "GeoML packages gaussx, kernellib and geonnax."
        )
        (OUT / f"layers-{mode}.svg").write_text(
            svg(width, height, label, "pyrox package layers", "\n".join(parts))
        )


if __name__ == "__main__":
    write_icons()
    write_logos()
    write_hero()
    write_layers()
