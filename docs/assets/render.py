"""Render the pyrox README assets: the icons and the logo.

Every output is a plain SVG written next to this file. The icons share the
gaussx tile (a 120-unit rounded square, r = 26, white glyph) in the pyrox
ember gradient; each package gets its own glyph:

    pyrox      plate notation (latent node over an observed node)
    pyrox-gp   GP mean with a credible band and data
    pyrox-nn   a small network with a distribution over its weights
    pyrox-lgm  a GMRF lattice with one node's neighbourhood ringed
    pyrox-xtreme  a long-tailed density, its return level and one event past it
    pyrox-ssm  a Markov chain of states, each observed below, one observation missing
    pyrox-gauss  a bell curve over a banded (tridiagonal) precision matrix

The logo comes in a light and a dark variant for GitHub's
``<picture>`` / ``prefers-color-scheme`` switch. The diagrams are drawn in geostack.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

import math
from pathlib import Path


OUT = Path(__file__).parent

SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Ember: the pyrox gradient and its wordmark accents.
C1, C2 = "#F97316", "#E11D48"
ACCENT = {"light": "#C2410C", "dark": "#FDBA74"}

TEXT = {"light": "#1F2937", "dark": "#E5E7EB"}

W = "#FFFFFF"


# ---------------------------------------------------------------- glyphs
# Each glyph draws on the 128-unit icon canvas; the tile spans 4..124.


def glyph_plate() -> str:
    """Latent θ above a plate holding an observed node: the core bridge."""
    return (
        f'<rect x="24" y="54" width="80" height="56" rx="10" fill="none" '
        f'stroke="{W}" stroke-width="4.5"/>'
        f'<circle cx="64" cy="31" r="12" fill="none" stroke="{W}" stroke-width="4.5"/>'
        f'<path d="M64 44 L64 69" stroke="{W}" stroke-width="4" '
        f'stroke-linecap="round"/>'
        f'<path d="M58 63 L64 69 L70 63" fill="none" stroke="{W}" '
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


def glyph_tail() -> str:
    """A long-tailed density, its return level and one event past it."""
    curve = "M16 100 C30 100 36 34 50 34 C62 34 70 78 86 90 C96 97 104 99 112 100"
    return (
        f'<path d="{curve} Z" fill="{W}" fill-opacity="0.28"/>'
        f'<path d="{curve}" fill="none" stroke="{W}" stroke-width="4.5" '
        'stroke-linecap="round"/>'
        f'<path d="M92 28 L92 100" stroke="{W}" stroke-width="3.5" '
        'stroke-dasharray="6 5" stroke-linecap="round"/>'
        f'<circle cx="104" cy="44" r="7" fill="{W}"/>'
    )


def glyph_chain() -> str:
    """Three latent states in a Markov chain, each observed below; one gap."""
    xs = (26, 64, 102)
    states = "".join(
        f'<circle cx="{x}" cy="44" r="9.5" fill="none" stroke="{W}" '
        'stroke-width="4.5"/>'
        for x in xs
    )
    arrows = "".join(
        f'<path d="M{x0} 44 L{x0 + 13} 44" stroke="{W}" stroke-width="4" '
        'stroke-linecap="round"/>'
        f'<path d="M{x0 + 7} 38.5 L{x0 + 13} 44 L{x0 + 7} 49.5" fill="none" '
        f'stroke="{W}" stroke-width="4" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        for x0 in (39, 77)
    )
    links = "".join(
        f'<path d="M{x} 56 L{x} 76" stroke="{W}" stroke-opacity="0.6" '
        'stroke-width="3.5" stroke-linecap="round"/>'
        for x in xs
    )
    observed = "".join(
        f'<circle cx="{x}" cy="88" r="8.5" fill="{W}"/>' for x in (xs[0], xs[2])
    )
    missing = (
        f'<circle cx="{xs[1]}" cy="88" r="8.5" fill="none" stroke="{W}" '
        'stroke-width="3" stroke-dasharray="4 4"/>'
    )
    return states + arrows + links + observed + missing


def glyph_bell_band() -> str:
    """A Gaussian density over a 5-by-5 tridiagonal precision, zeros as dots."""
    pts = []
    for k in range(41):
        t = -3 + 6 * k / 40
        pts.append(f"{64 + 13 * t:.1f} {44 - 27 * math.exp(-0.5 * t * t):.1f}")
    curve = "M" + " L".join(pts)
    cells = []
    for i in range(5):
        for j in range(5):
            x, y = 35.25 + 12 * j, 52 + 12 * i
            if i == j:
                cells.append(
                    f'<rect x="{x}" y="{y}" width="9.5" height="9.5" rx="2" '
                    f'fill="{W}"/>'
                )
            elif abs(i - j) == 1:
                cells.append(
                    f'<rect x="{x}" y="{y}" width="9.5" height="9.5" rx="2" '
                    f'fill="{W}" fill-opacity="0.55"/>'
                )
            else:
                cells.append(
                    f'<circle cx="{x + 4.75}" cy="{y + 4.75}" r="1.7" fill="{W}" '
                    'fill-opacity="0.45"/>'
                )
    return (
        f'<path d="{curve} Z" fill="{W}" fill-opacity="0.28"/>'
        f'<path d="{curve}" fill="none" stroke="{W}" stroke-width="4.5" '
        'stroke-linecap="round" stroke-linejoin="round"/>' + "".join(cells)
    )


GLYPHS = {
    "pyrox": glyph_plate,
    "pyrox-gp": glyph_band,
    "pyrox-nn": glyph_network,
    "pyrox-lgm": glyph_lattice,
    "pyrox-xtreme": glyph_tail,
    "pyrox-ssm": glyph_chain,
    "pyrox-gauss": glyph_bell_band,
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


# ---------------------------------------------------------------- icons and logo


def write_icons() -> None:
    for name, glyph in GLYPHS.items():
        stem = "icon" if name == "pyrox" else f"icon-{name.removeprefix('pyrox-')}"
        body = tile(f"{stem}-g", glyph())
        (OUT / f"{stem}.svg").write_text(svg(128, 128, name, name, body))


def write_logos() -> None:
    for mode, colour in TEXT.items():
        body = tile(f"logo-{mode}", glyph_plate(), x=6, y=6) + (
            f'<text x="154" y="92" font-family="{SANS}" font-size="72" '
            f'font-weight="700" letter-spacing="-1.5" fill="{colour}">'
            f'pyro<tspan fill="{ACCENT[mode]}">x</tspan></text>'
        )
        (OUT / f"logo-{mode}.svg").write_text(svg(400, 140, "pyrox", "pyrox", body))


if __name__ == "__main__":
    write_icons()
    write_logos()
