"""MedEvolve method overview v6-vector — true editable SVG matching AI draft layout.

Layout: Fixed dual-explore LEFT | Generate·Verify·Evaluate CENTER |
        thin orange Feedback bottom | gray Held-out bottom-right.
Notation: W_r, g_r, K_D, K_E, F_r, theta_{r+1}, theta*.
Exports: medevolve-overview-v6-vector.{svg,pdf,png}
Keeps v4/v5 assets untouched.
"""

from pathlib import Path
from math import cos, sin, pi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse,
    FancyBboxPatch as FBP,
)
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent

# Palette (per task)
INK = "#17324D"
MUTED = "#52677B"
TEAL = "#087F83"
ORANGE = "#C45B25"
EDGE = "#CEDCE5"
PALE = "#EFF7F8"
WARM = "#FFF5ED"
WHITE = "#FFFFFF"
TEAL_DARK = "#066568"
BAND_TEAL = "#D5EEEE"
BAND_ORANGE = "#F8E8DC"
GRAY_FILL = "#F5F7F9"
GRAY_EDGE = "#D0D8E0"
FAIL_GRAY = "#8A97A5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",   # real <text> nodes, editable
    "pdf.fonttype": 42,
})

# Canvas ~3:2 like the AI draft (1536x1024)
W, H = 1536, 1024
fig = plt.figure(figsize=(15.36, 10.24))
ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
fig.patch.set_facecolor(WHITE)


def rbox(x, y, w, h, fill=WHITE, edge=EDGE, radius=10, lw=1.2, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
        mutation_aspect=1,
    ))


def header_bar(x, y, w, h, fill=TEAL, radius=8):
    """Rounded top header strip inside a panel."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor="none", zorder=2,
    ))
    # square off bottom of header so it meets panel body cleanly
    ax.add_patch(Rectangle((x, y + h - radius), w, radius + 1,
                           facecolor=fill, edgecolor="none", zorder=2))


def txt(x, y, s, size=12, color=INK, bold=False, ha="center", va="center", z=6):
    ax.text(
        x, y, s, fontsize=size, color=color,
        fontweight="bold" if bold else "normal",
        ha=ha, va=va, zorder=z, linespacing=1.2,
    )


def icon(kind, x, y, color=TEAL, size=14):
    def line(pts, width=1.3):
        ax.plot(
            [x + px * size for px, _ in pts],
            [y + py * size for _, py in pts],
            color=color, lw=width, zorder=5,
            solid_capstyle="round", solid_joinstyle="round",
        )

    def circ(cx, cy, r, fill="none", ec=None):
        ax.add_patch(Circle(
            (x + cx * size, y + cy * size), r * size,
            facecolor=fill, edgecolor=ec or color, lw=1.2, zorder=5,
        ))

    if kind == "search":
        circ(-.08, -.08, .34)
        line([(.18, .18), (.48, .48)], 1.8)
    elif kind == "record":
        line([(-.36, -.48), (.12, -.48), (.36, -.24), (.36, .48), (-.36, .48), (-.36, -.48)])
        line([(.12, -.48), (.12, -.24), (.36, -.24)])
        for dy in (-.12, .08, .28):
            line([(-.2, dy), (.18, dy)])
    elif kind == "person":
        circ(0, -.28, .22, WHITE)
        ax.add_patch(Ellipse(
            (x, y + .22 * size), .7 * size, .5 * size,
            facecolor=WHITE, edgecolor=color, lw=1.2, zorder=5,
        ))
    elif kind == "book":
        line([(-.35, -.45), (.25, -.45), (.35, -.35), (.35, .45), (-.35, .45), (-.35, -.45)])
        line([(-.05, -.45), (-.05, .45)])
        line([(.05, -.2), (.25, -.2)])
        line([(.05, 0), (.25, 0)])
    elif kind == "doc":
        line([(-.32, -.48), (.16, -.48), (.36, -.28), (.36, .48), (-.32, .48), (-.32, -.48)])
        line([(.16, -.48), (.16, -.28), (.36, -.28)])
        for dy in (-.1, .1, .3):
            line([(-.18, dy), (.2, dy)])
    elif kind == "sliders":
        for i, (cx, cy) in enumerate([(-.28, -.15), (0, .2), (.28, -.05)]):
            line([(cx, -.45), (cx, .45)], 1.4)
            circ(cx, cy, .16, color)
    elif kind == "gear":
        for i in range(8):
            a = i * pi / 4
            # tooth as thick short radial stroke
            line([(.26 * cos(a), .26 * sin(a)), (.52 * cos(a), .52 * sin(a))], 3.2)
        circ(0, 0, .30, color)
        circ(0, 0, .12, WHITE)
    elif kind == "database":
        ax.add_patch(Rectangle(
            (x - .38 * size, y - .28 * size), .76 * size, .6 * size,
            facecolor=WHITE, edgecolor=color, lw=1.1, zorder=4,
        ))
        for dy in (-.28, 0, .32):
            ax.add_patch(Ellipse(
                (x, y + dy * size), .76 * size, .28 * size,
                facecolor=WHITE if dy == -.28 else "none",
                edgecolor=color, lw=1.1, zorder=5,
            ))
    elif kind == "cylinder":
        # knowledge cylinder (taller)
        ax.add_patch(Rectangle(
            (x - .4 * size, y - .35 * size), .8 * size, .7 * size,
            facecolor=PALE, edgecolor=color, lw=1.3, zorder=4,
        ))
        for dy in (-.35, .35):
            ax.add_patch(Ellipse(
                (x, y + dy * size), .8 * size, .28 * size,
                facecolor=PALE if dy < 0 else "none",
                edgecolor=color, lw=1.3, zorder=5,
            ))
    elif kind == "star":
        pts = []
        for i in range(10):
            angle = -pi / 2 + i * pi / 5
            radius = .48 if i % 2 == 0 else .22
            pts.append((x + cos(angle) * size * radius, y + sin(angle) * size * radius))
        ax.add_patch(Polygon(pts, facecolor=color, edgecolor=color, lw=.5, zorder=5))
    elif kind == "bars":
        for dx, height in [(-.32, .35), (-.02, .55), (.28, .75)]:
            ax.add_patch(Rectangle(
                (x + dx * size, y + (.4 - height) * size),
                .18 * size, height * size,
                facecolor=color, edgecolor="none", zorder=5,
            ))
    elif kind == "brain":
        circ(-.12, -.05, .32, WHITE)
        circ(.18, -.05, .28, WHITE)
        circ(0, .2, .22, WHITE)
    elif kind == "checklist":
        line([(-.3, -.45), (.3, -.45), (.3, .45), (-.3, .45), (-.3, -.45)])
        for dy in (-.2, 0, .2):
            line([(-.12, dy), (.18, dy)])
            line([(-.22, dy), (-.16, dy - .06), (-.08, dy + .08)], 1.4)
    elif kind == "check":
        circ(0, 0, .42, color)
        ax.plot(
            [x - .2 * size, x - .04 * size, x + .22 * size],
            [y + .02 * size, y + .18 * size, y - .2 * size],
            color=WHITE, lw=2.0, zorder=6,
            solid_capstyle="round", solid_joinstyle="round",
        )
    elif kind == "cross":
        circ(0, 0, .42, FAIL_GRAY)
        ax.plot(
            [x - .2 * size, x + .2 * size],
            [y - .2 * size, y + .2 * size],
            color=WHITE, lw=2.0, zorder=6, solid_capstyle="round",
        )
        ax.plot(
            [x - .2 * size, x + .2 * size],
            [y + .2 * size, y - .2 * size],
            color=WHITE, lw=2.0, zorder=6, solid_capstyle="round",
        )
    elif kind == "list":
        for i, dy in enumerate([-.3, 0, .3]):
            circ(-.28, dy, .1, color)
            line([(-.1, dy), (.35, dy)])
    elif kind == "snowflake":
        for i in range(6):
            a = i * pi / 3
            line([(0, 0), (.45 * cos(a), .45 * sin(a))], 1.4)
            line([(.25 * cos(a) - .1 * sin(a), .25 * sin(a) + .1 * cos(a)),
                  (.25 * cos(a), .25 * sin(a))], 1.1)
    elif kind == "people":
        for ox in (-.28, .28):
            circ(ox, -.22, .16, WHITE)
            ax.add_patch(Ellipse(
                (x + ox * size, y + .2 * size), .4 * size, .35 * size,
                facecolor=WHITE, edgecolor=color, lw=1.1, zorder=5,
            ))
        circ(0, -.28, .18, WHITE)
        ax.add_patch(Ellipse(
            (x, y + .18 * size), .48 * size, .4 * size,
            facecolor=WHITE, edgecolor=color, lw=1.1, zorder=5,
        ))
    elif kind == "clipboard":
        line([(-.32, -.4), (.32, -.4), (.32, .45), (-.32, .45), (-.32, -.4)])
        ax.add_patch(Rectangle(
            (x - .14 * size, y - .52 * size), .28 * size, .18 * size,
            facecolor=WHITE, edgecolor=color, lw=1.1, zorder=5,
        ))
        line([(-.12, 0), (-.02, .12), (.18, -.12)], 1.6)


def solid(pts, color=TEAL, lw=1.6, arrow=True, ms=11):
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, lw=lw, zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1],
        arrowstyle="-|>" if arrow else "-",
        mutation_scale=ms, linewidth=lw, color=color,
        shrinkA=0, shrinkB=0, zorder=3,
    ))


def dashed(pts, color=ORANGE, lw=1.3, arrow=True, ms=10):
    if len(pts) > 2:
        ax.plot(
            [p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
            color=color, lw=lw, ls=(0, (5, 3.5)), zorder=3, solid_capstyle="butt",
        )
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1],
        arrowstyle="-|>" if arrow else "-",
        mutation_scale=ms, linewidth=lw, color=color,
        linestyle=(0, (5, 3.5)), shrinkA=0, shrinkB=0, zorder=3,
    ))


# =====================================================================
# TITLE
# =====================================================================
txt(W / 2, 36, "MedEvolve", size=28, color=INK, bold=True)

# =====================================================================
# LEFT: Fixed dual explore
# =====================================================================
FX, FY, FW, FH = 28, 70, 300, 560
rbox(FX, FY, FW, FH, WHITE, TEAL, 12, 1.6, z=1)
header_bar(FX, FY, FW, 40, TEAL, 10)
txt(FX + FW / 2, FY + 20, "Fixed", size=15, color=WHITE, bold=True)

# --- Data explore ---
DX, DY, DW, DH = FX + 14, FY + 56, FW - 28, 230
rbox(DX, DY, DW, DH, WHITE, EDGE, 8, .9)
txt(DX + DW / 2, DY + 18, "Data explore", size=12, color=TEAL, bold=True)

# stacked record cards
for i, (ox, oy) in enumerate([(12, 8), (6, 4), (0, 0)]):
    rbox(DX + 40 + ox, DY + 42 + oy, 78, 52, WHITE if i < 2 else PALE, EDGE, 4, .7, z=3 + i)
icon("person", DX + 79, DY + 68, TEAL, 13)
# timeline
tl_y = DY + 120
ax.plot([DX + 28, DX + DW - 28], [tl_y, tl_y], color=EDGE, lw=1.2, zorder=3)
for i, lab in enumerate([r"$t_1$", r"$t_2$", r"$\cdots$", r"$t_n$"]):
    px = DX + 45 + i * 55
    ax.add_patch(Circle((px, tl_y), 4.5, facecolor=TEAL, edgecolor="none", zorder=4))
    txt(px, tl_y + 18, lab, size=10, color=MUTED)

# arrow down to KD cylinder
solid([(DX + DW / 2, DY + 155), (DX + DW / 2, DY + 175)], TEAL, 1.3, ms=9)
# KD knowledge chip
rbox(DX + 40, DY + 175, DW - 80, 42, PALE, TEAL, 6, 1.1)
icon("cylinder", DX + 62, DY + 196, TEAL, 12)
txt(DX + DW / 2 + 10, DY + 196, r"$K_D$", size=14, color=TEAL, bold=True)

# --- Environment explore ---
EX, EY, EW, EH = FX + 14, FY + 300, FW - 28, 240
rbox(EX, EY, EW, EH, WHITE, EDGE, 8, .9)
txt(EX + EW / 2, EY + 18, "Environment explore", size=12, color=TEAL, bold=True)

for px, lab, kind in [
    (EX + 50, "Search", "search"),
    (EX + EW / 2, "Read", "record"),
    (EX + EW - 50, "FHIR", "database"),
]:
    icon(kind, px, EY + 55, TEAL, 14)
    txt(px, EY + 78, lab, size=9, color=MUTED)

solid([(EX + EW / 2, EY + 100), (EX + EW / 2, EY + 120)], TEAL, 1.3, ms=9)
rbox(EX + 40, EY + 120, EW - 80, 42, PALE, TEAL, 6, 1.1)
icon("cylinder", EX + 62, EY + 141, TEAL, 12)
txt(EX + EW / 2 + 10, EY + 141, r"$K_E$", size=14, color=TEAL, bold=True)

# note under Fixed
txt(FX + FW / 2, FY + FH - 18, "reuse across rounds", size=9, color=MUTED)

# =====================================================================
# CENTER: Generate · Verify · Evaluate
# =====================================================================
MX, MY, MW, MH = 350, 70, 1160, 560
rbox(MX, MY, MW, MH, WHITE, TEAL, 12, 1.6, z=1)
header_bar(MX, MY, MW, 40, TEAL, 10)
txt(MX + MW / 2, MY + 20, "Generate  ·  Verify  ·  Evaluate", size=15, color=WHITE, bold=True)

# three panels
GAP = 16
PW = (MW - 40 - 2 * GAP) / 3
PY, PH = MY + 56, MH - 72

# ---- Generate ----
G1X = MX + 20
rbox(G1X, PY, PW, PH, WHITE, EDGE, 8, 1.0)
header_bar(G1X + 1, PY + 1, PW - 2, 32, BAND_TEAL, 6)
txt(G1X + PW / 2, PY + 17, "Generate", size=13, color=TEAL, bold=True)

# inputs column
inputs = [
    (r"$X_p$", "person", "patient"),
    (r"$K_D$", "book", "data knowledge"),
    (r"$K_E$", "book", "env knowledge"),
    (r"$g_r$", "doc", "guidance"),
    (r"$W_r$", "sliders", "weights"),
]
for i, (lab, kind, sub) in enumerate(inputs):
    iy = PY + 50 + i * 48
    rbox(G1X + 14, iy, 118, 40, PALE, EDGE, 5, .6)
    icon(kind, G1X + 32, iy + 20, TEAL, 11)
    txt(G1X + 78, iy + 13, lab, size=12, color=INK, bold=True)
    txt(G1X + 78, iy + 28, sub, size=7.5, color=MUTED)

# brace / gather into gear
gear_x = G1X + PW - 95
gear_y = PY + PH / 2 - 10
# horizontal gather lines from inputs mid to gear
for i in range(5):
    iy = PY + 50 + i * 48 + 20
    ax.plot([G1X + 132, gear_x - 28], [iy, gear_y], color=EDGE, lw=0.9, zorder=2)
rbox(gear_x - 28, gear_y - 28, 56, 56, PALE, TEAL, 8, 1.2)
icon("gear", gear_x, gear_y, TEAL, 16)

# Candidates dashed box
cand_y = PY + PH - 130
rbox(G1X + 14, cand_y, PW - 28, 110, "#F8FBFC", EDGE, 6, .8)
# dashed border overlay hint
ax.add_patch(FancyBboxPatch(
    (G1X + 14, cand_y), PW - 28, 110,
    boxstyle="round,pad=0,rounding_size=6",
    facecolor="none", edgecolor=TEAL, linewidth=1.0,
    linestyle=(0, (4, 3)), zorder=4,
))
txt(G1X + PW / 2, cand_y + 16, "Candidates", size=11, color=TEAL, bold=True)
for i, (lab, kind, muted) in enumerate([
    ("Question", "doc", False),
    ("Answer (hidden)", "doc", True),
    ("Evidence (hidden)", "record", True),
]):
    cx = G1X + 50 + i * ((PW - 60) / 3)
    icon(kind, cx, cand_y + 55, MUTED if muted else ORANGE, 11)
    txt(cx, cand_y + 82, lab, size=8, color=MUTED if muted else INK, bold=not muted)

solid([(gear_x, gear_y + 30), (gear_x, cand_y)], TEAL, 1.3, ms=9)

# ---- Evidence verification ----
G2X = G1X + PW + GAP
rbox(G2X, PY, PW, PH, WHITE, EDGE, 8, 1.0)
header_bar(G2X + 1, PY + 1, PW - 2, 32, BAND_TEAL, 6)
txt(G2X + PW / 2, PY + 17, "Evidence verification", size=12, color=TEAL, bold=True)

# against original records
rbox(G2X + 30, PY + 70, PW - 60, 100, PALE, TEAL, 8, 1.1)
icon("record", G2X + PW / 2, PY + 105, TEAL, 22)
txt(G2X + PW / 2, PY + 145, "Against original records", size=11, color=TEAL, bold=True)

# fork to Pass / Fail
fork_y = PY + 200
solid([(G2X + PW / 2, PY + 170), (G2X + PW / 2, fork_y)], TEAL, 1.3, ms=0, arrow=False)

# Pass
pass_x = G2X + 40
rbox(pass_x, fork_y + 20, 130, 70, PALE, TEAL, 8, 1.3)
icon("check", pass_x + 30, fork_y + 55, TEAL, 14)
txt(pass_x + 85, fork_y + 55, "Pass", size=13, color=TEAL, bold=True)
solid([(G2X + PW / 2, fork_y), (pass_x + 65, fork_y), (pass_x + 65, fork_y + 20)], TEAL, 1.3, ms=9)

# Fail
fail_x = G2X + PW - 170
rbox(fail_x, fork_y + 20, 130, 70, GRAY_FILL, GRAY_EDGE, 8, 1.0)
icon("cross", fail_x + 30, fork_y + 55, FAIL_GRAY, 14)
txt(fail_x + 85, fork_y + 55, "Fail", size=13, color=MUTED, bold=True)
solid([(G2X + PW / 2, fork_y), (fail_x + 65, fork_y), (fail_x + 65, fork_y + 20)], FAIL_GRAY, 1.2, ms=9)

# note
txt(G2X + PW / 2, PY + PH - 40, "source–claim check", size=9, color=MUTED)
txt(G2X + PW / 2, PY + PH - 22, "strict filter → verified set", size=9, color=MUTED)

# ---- Evaluate ----
G3X = G2X + PW + GAP
rbox(G3X, PY, PW, PH, WHITE, EDGE, 8, 1.0)
header_bar(G3X + 1, PY + 1, PW - 2, 32, BAND_TEAL, 6)
txt(G3X + PW / 2, PY + 17, "Evaluate", size=13, color=TEAL, bold=True)

evals = [
    ("Quality", "star"),
    ("Depth", "bars"),
    ("Blind solve", "brain"),
    ("Complete", "checklist"),
]
for i, (lab, kind) in enumerate(evals):
    row, col = divmod(i, 2)
    cx = G3X + 24 + col * ((PW - 40) / 2)
    cy = PY + 70 + row * 160
    cw = (PW - 52) / 2
    rbox(cx, cy, cw, 130, WHITE, EDGE, 8, .8)
    icon(kind, cx + cw / 2, cy + 48, TEAL, 20)
    txt(cx + cw / 2, cy + 95, lab, size=12, color=INK, bold=True)

# spine arrows between panels
mid_y = PY + PH / 2
solid([(G1X + PW, mid_y), (G2X, mid_y)], TEAL, 2.0, ms=13)
solid([(G2X + PW, mid_y), (G3X, mid_y)], TEAL, 2.0, ms=13)
txt((G2X + G3X) / 2, mid_y - 18, "pass", size=9, color=MUTED)

# KD/KE feed arrows into Generate
solid([(FX + FW, DY + 196), (MX, DY + 196), (MX + 10, DY + 196), (G1X, PY + 50 + 1 * 48 + 20)],
      TEAL, 1.4, ms=10)
solid([(FX + FW, EY + 141), (MX - 8, EY + 141), (MX - 8, PY + 50 + 2 * 48 + 20), (G1X, PY + 50 + 2 * 48 + 20)],
      TEAL, 1.4, ms=10)

# =====================================================================
# BOTTOM: Feedback (orange, thin) + Held-out (gray)
# =====================================================================
BY = 660
BH = 320

# Feedback panel
FBX, FBW = 28, 900
rbox(FBX, BY, FBW, BH, WARM, ORANGE, 12, 1.5)
header_bar(FBX, BY, FBW, 36, ORANGE, 10)
txt(FBX + 20, BY + 18, r"Feedback ($F_r$)", size=14, color=WHITE, bold=True, ha="left")
txt(FBX + FBW - 20, BY + 18, "secondary · mixed effect", size=9, color="#FFE8D6", ha="right")

# three chips in feedback
chip_y = BY + 70
chip_h = 140
# Invalid reasons
c1x, c1w = FBX + 24, 220
rbox(c1x, chip_y, c1w, chip_h, WHITE, "#E8C4A8", 8, .9)
icon("list", c1x + c1w / 2, chip_y + 45, ORANGE, 16)
txt(c1x + c1w / 2, chip_y + 90, "Invalid reasons", size=11, color=ORANGE, bold=True)
txt(c1x + c1w / 2, chip_y + 112, "from Fail", size=9, color=MUTED)

# Scores / traces
c2x, c2w = FBX + 270, 220
rbox(c2x, chip_y, c2w, chip_h, WHITE, "#E8C4A8", 8, .9)
icon("bars", c2x + c2w / 2, chip_y + 45, ORANGE, 16)
txt(c2x + c2w / 2, chip_y + 90, "Scores / traces", size=11, color=ORANGE, bold=True)
txt(c2x + c2w / 2, chip_y + 112, "from Evaluate", size=9, color=MUTED)

txt((c1x + c1w + c2x) / 2, chip_y + chip_h / 2, "+", size=18, color=ORANGE, bold=True)

# arrow to Update
solid([(c2x + c2w + 8, chip_y + chip_h / 2), (FBX + 560, chip_y + chip_h / 2)],
      ORANGE, 1.5, ms=11)

# Update chip
c3x, c3w = FBX + 560, 310
rbox(c3x, chip_y, c3w, chip_h, WHITE, ORANGE, 8, 1.2)
icon("gear", c3x + 40, chip_y + 50, ORANGE, 14)
txt(c3x + c3w / 2 + 10, chip_y + 40, "Update", size=12, color=ORANGE, bold=True)
txt(c3x + c3w / 2 + 10, chip_y + 70, r"$g_{r+1},\ W_{r+1}$", size=13, color=INK, bold=True)
txt(c3x + c3w / 2 + 10, chip_y + 100, r"$\theta_{r+1}$  next round", size=11, color=MUTED)

# dashed feedback up to Generate
dashed([
    (c3x + c3w / 2, chip_y),
    (c3x + c3w / 2, BY - 8),
    (gear_x, BY - 8),
    (gear_x, MY + MH),
], ORANGE, 1.4, ms=11)
txt(gear_x + 50, BY - 20, r"$\theta_{r+1}$", size=11, color=ORANGE, bold=True)

# dashed down from Fail and Evaluate into Feedback
dashed([
    (fail_x + 65, fork_y + 90),
    (fail_x + 65, MY + MH + 10),
    (c1x + c1w / 2, MY + MH + 10),
    (c1x + c1w / 2, BY),
], ORANGE, 1.1, ms=9)
txt(c1x + c1w / 2 + 40, MY + MH + 22, "invalid", size=8, color=ORANGE)

dashed([
    (G3X + PW / 2, MY + MH),
    (G3X + PW / 2, BY - 4),
    (c2x + c2w / 2, BY - 4),
    (c2x + c2w / 2, BY),
], ORANGE, 1.1, ms=9)
txt(G3X + PW / 2 + 45, MY + MH + 14, "scores", size=8, color=ORANGE)

# footnote in feedback
txt(FBX + 24, BY + BH - 28, "Strategy update is descriptive / mixed — not a guaranteed improvement.",
    size=9, color=MUTED, ha="left")

# ---- Held-out ----
HX, HW = 950, 560
rbox(HX, BY, HW, BH, GRAY_FILL, GRAY_EDGE, 12, 1.2)
header_bar(HX, BY, HW, 36, MUTED, 10)
txt(HX + 20, BY + 18, "Held-out", size=14, color=WHITE, bold=True, ha="left")
txt(HX + HW - 20, BY + 18, "after evolution", size=9, color="#D8DEE5", ha="right")

held_steps = [
    (r"Freeze $\theta^*$", "snowflake"),
    ("Held-out patients", "people"),
    ("Independent Eval", "clipboard"),
]
sw = 150
gap_h = 28
total_w = 3 * sw + 2 * gap_h
start_x = HX + (HW - total_w) / 2
for i, (lab, kind) in enumerate(held_steps):
    sx = start_x + i * (sw + gap_h)
    sy = BY + 90
    rbox(sx, sy, sw, 150, WHITE, GRAY_EDGE, 8, .8)
    icon(kind, sx + sw / 2, sy + 50, MUTED, 18)
    txt(sx + sw / 2, sy + 110, lab, size=10, color=MUTED, bold=True)
    if i < 2:
        solid([(sx + sw + 4, sy + 75), (sx + sw + gap_h - 4, sy + 75)], MUTED, 1.2, ms=9)

# dashed from Update to Freeze
dashed([
    (c3x + c3w, chip_y + chip_h / 2),
    (HX - 10, chip_y + chip_h / 2),
    (HX - 10, BY + 165),
    (start_x, BY + 165),
], ORANGE, 1.2, ms=10)
txt((c3x + c3w + HX) / 2, chip_y + chip_h / 2 - 16, "final round", size=8, color=ORANGE)

txt(HX + HW / 2, BY + BH - 28, "Held-out results do not update strategy.",
    size=9, color=MUTED)

# =====================================================================
# Export
# =====================================================================
stem = "medevolve-overview-v6-vector"
for ext in ("svg", "pdf", "png"):
    path = OUT / f"{stem}.{ext}"
    fig.savefig(path, facecolor=WHITE, dpi=200, bbox_inches=None)
    print(path)
plt.close(fig)
