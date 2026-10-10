"""MedEvolve method overview v5 — glanceable L→R spine for NAACL short.

Spine: Explore (KD+KE) → Generate → Verify → Feedback → Freeze+held-out.
Main path dominates; (c)/(d) sit in a thin secondary band.
Solid arrows = data flow; dashed = feedback.
"""

from pathlib import Path
from math import cos, sin, pi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse,
)
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
INK, MUTED = "#17324D", "#5A6E80"
TEAL = "#087F83"
ORANGE = "#B87A55"          # soft, secondary only
EDGE = "#C9D6DF"
PALE = "#EFF7F8"
ORANGE_FILL = "#FFFBF7"
ORANGE_EDGE = "#E0C4AE"
BAND_TEAL = "#DDEFF0"
BAND_ORANGE = "#F3E6DA"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Wide ~2:1 figure* friendly
fig = plt.figure(figsize=(17.2, 7.2))
ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
ax.set(xlim=(0, 1720), ylim=(720, 0))
ax.axis("off")


def box(x, y, w, h, fill=WHITE, edge=EDGE, radius=8, lw=1.0, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
    ))


def band(x, y, w, h, fill):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
        facecolor=fill, edgecolor="none", zorder=2,
    ))
    ax.add_patch(Rectangle((x, y + h - 7), w, 7, facecolor=fill, edgecolor="none", zorder=2))


def text(x, y, s, size=13, color=INK, bold=False, align="center", z=5):
    ax.text(
        x, y, s, fontsize=size, color=color,
        fontweight="bold" if bold else "normal",
        ha=align, va="center", zorder=z, linespacing=1.25,
    )


def icon(kind, x, y, color=TEAL, size=16):
    def line(pts, width=1.2):
        ax.plot(
            [x + px * size for px, _ in pts],
            [y + py * size for _, py in pts],
            color=color, lw=width, zorder=4,
            solid_capstyle="round", solid_joinstyle="round",
        )

    def circ(cx, cy, r, fill="none"):
        ax.add_patch(Circle(
            (x + cx * size, y + cy * size), r * size,
            facecolor=fill, edgecolor=color, lw=1.2, zorder=4,
        ))

    if kind == "search":
        circ(-.10, -.10, .36)
        line([(.18, .18), (.48, .48)], 1.7)
    elif kind in ("record", "question", "answer", "evidence"):
        line([(-.36, -.5), (.14, -.5), (.36, -.28), (.36, .5), (-.36, .5), (-.36, -.5)])
        line([(.14, -.5), (.14, -.28), (.36, -.28)])
        if kind == "question":
            line([(-.12, -.10), (-.06, -.23), (.1, -.23), (.16, -.10), (0, .04), (0, .14)])
            circ(0, .32, .025, color)
        elif kind == "answer":
            line([(-.18, .02), (-.03, .18), (.23, -.09)], 1.5)
        else:
            for dy in (-.15, .05, .25):
                line([(-.2, dy), (.18, dy)])
            if kind == "evidence":
                circ(.28, .27, .22)
                line([(.44, .44), (.58, .58)])
    elif kind == "clinical":
        line([(-.38, -.48), (.38, -.48), (.38, .48), (-.38, .48), (-.38, -.48)])
        line([(-.19, -.04), (.19, -.04)], 1.6)
        line([(0, -.23), (0, .15)], 1.6)
        line([(-.20, .31), (.20, .31)])
    elif kind in ("trend", "feedback"):
        for dx, height in [(-.35, .28), (-.05, .48), (.25, .70)]:
            ax.add_patch(Rectangle(
                (x + dx * size, y + (.4 - height) * size),
                .16 * size, height * size,
                facecolor=color, edgecolor="none", zorder=4,
            ))
        if kind == "feedback":
            circ(.39, .30, .24)
            line([(.55, .46), (.72, .63)], 1.5)
    elif kind == "tools":
        line([
            (-.43, .37), (-.28, .49), (.22, -.01), (.41, -.06), (.5, -.24), (.46, -.44),
            (.29, -.25), (.14, -.38), (.31, -.55), (.08, -.5), (-.03, -.28), (.01, -.12), (-.43, .37),
        ])
    elif kind == "quality":
        pts = []
        for i in range(10):
            angle = -pi / 2 + i * pi / 5
            radius = .5 if i % 2 == 0 else .23
            pts.append((x + cos(angle) * size * radius, y + sin(angle) * size * radius))
        ax.add_patch(Polygon(pts, facecolor=color, edgecolor=color, lw=.6, zorder=4))
    elif kind == "reasoning":
        for a, b in [((-.36, .28), (0, -.3)), ((0, -.3), (.37, .05)), ((-.36, .28), (.37, .05))]:
            line([a, b])
        for cx, cy in [(-.36, .28), (0, -.3), (.37, .05)]:
            circ(cx, cy, .12, WHITE)
    elif kind in ("solve", "update"):
        circ(0, 0, .32)
        circ(0, 0, .12)
        for i in range(8):
            a = i * pi / 4
            line([(.34 * cos(a), .34 * sin(a)), (.5 * cos(a), .5 * sin(a))], 1.5)
    elif kind == "complete":
        circ(0, 0, .48, color)
        ax.plot(
            [x - .25 * size, x - .06 * size, x + .28 * size],
            [y, y + .19 * size, y - .22 * size],
            color=WHITE, lw=1.6, zorder=5,
        )
    elif kind == "database":
        ax.add_patch(Rectangle(
            (x - .4 * size, y - .3 * size), .8 * size, .65 * size,
            facecolor=WHITE, edgecolor=color, lw=1.1, zorder=3,
        ))
        for dy in (-.3, 0, .35):
            ax.add_patch(Ellipse(
                (x, y + dy * size), .8 * size, .3 * size,
                facecolor=WHITE if dy == -.3 else "none",
                edgecolor=color, lw=1.1, zorder=4,
            ))
    elif kind == "arrow_r":
        line([(-.4, 0), (.25, 0)], 1.6)
        line([(.05, -.28), (.4, 0), (.05, .28)], 1.6)


def solid(points, color=TEAL, lw=1.7, arrow=True, ms=12):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot((a[0], b[0]), (a[1], b[1]), color=color, lw=lw, zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1],
        arrowstyle="-|>" if arrow else "-",
        mutation_scale=ms, linewidth=lw, color=color,
        shrinkA=0, shrinkB=0, zorder=3,
    ))


def dashed(points, color=ORANGE, lw=1.2, arrow=True, ms=10):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=(0, (4, 3)), zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1],
        arrowstyle="-|>" if arrow else "-",
        mutation_scale=ms, linewidth=lw, color=color,
        linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0, zorder=3,
    ))


def chip(x, y, w, h, label, kind, color=TEAL, sub=None):
    box(x, y, w, h, WHITE, EDGE, 5, .7)
    cy = y + h / 2
    icon(kind, x + 18, cy - (6 if sub else 0), color, 14)
    text(x + w / 2 + 6, cy - (7 if sub else 0), label, 11, bold=True)
    if sub:
        text(x + w / 2 + 6, cy + 10, sub, 9, MUTED)


# =========================================================================
# MAIN SPINE — (a) Explore → (b) Generate → Verify → Eval
# =========================================================================
MAIN_Y, MAIN_H = 18, 470

# ---- (a) Dual exploration ----
AX, AW = 18, 360
box(AX, MAIN_Y, AW, MAIN_H, "#FBFDFE", EDGE, 10, .9)
text(AX + 18, MAIN_Y + 22, "(a) Dual exploration", 15, TEAL, True, "left")
text(AX + AW - 18, MAIN_Y + 22, "fixed", 10, MUTED, align="right")

# Data branch
box(AX + 18, 58, AW - 36, 175, WHITE, EDGE, 7, .7)
text(AX + AW / 2, 78, "Records", 12, INK, True)
# mini timeline dots
for i, lab in enumerate([r"$t_1$", r"$t_2$", r"$t_n$"]):
    px = AX + 70 + i * 90
    icon("clinical" if i == 0 else "record", px, 108, TEAL, 16)
    text(px, 132, lab, 9, MUTED)
solid([(AX + AW / 2, 148), (AX + AW / 2, 162)], TEAL, 1.3)
box(AX + 36, 162, AW - 72, 52, PALE, TEAL, 6, 1.0)
icon("search", AX + 58, 188, TEAL, 15)
text(AX + AW / 2 + 8, 178, "Data explore", 12, TEAL, True)
text(AX + AW / 2 + 8, 198, r"$K_D$", 13, TEAL, True)

# Env branch
box(AX + 18, 255, AW - 36, 200, WHITE, EDGE, 7, .7)
text(AX + AW / 2, 275, "Tools", 12, INK, True)
for px, lab, kind in [
    (AX + 70, "Search", "search"),
    (AX + AW / 2, "Read", "record"),
    (AX + AW - 70, "FHIR", "database"),
]:
    icon(kind, px, 305, TEAL, 15)
    text(px, 328, lab, 9, MUTED)
solid([(AX + AW / 2, 345), (AX + AW / 2, 360)], TEAL, 1.3)
box(AX + 36, 360, AW - 72, 70, PALE, TEAL, 6, 1.0)
icon("tools", AX + 58, 395, TEAL, 15)
text(AX + AW / 2 + 8, 385, "Env explore", 12, TEAL, True)
text(AX + AW / 2 + 8, 408, r"$K_E$", 13, TEAL, True)

# ---- (b) Generate + verify + eval as sequential stages ----
BX = 400
# outer section
box(BX, MAIN_Y, 1300, MAIN_H, "#FBFDFE", EDGE, 10, .9)
text(BX + 18, MAIN_Y + 22, "(b) Generate · verify · evaluate", 15, TEAL, True, "left")

# Stage widths aligned for equal visual weight
# Patient+Gen | Candidates | Verify | Evaluate
S1X, S1W = BX + 20, 300
S2X, S2W = BX + 340, 230
S3X, S3W = BX + 600, 260
S4X, S4W = BX + 900, 370
STAGE_Y, STAGE_H = 55, 400

# Stage 1: Generate
box(S1X, STAGE_Y, S1W, STAGE_H, WHITE, EDGE, 8, .8)
band(S1X + 1, STAGE_Y + 1, S1W - 2, 44, TEAL)
text(S1X + S1W / 2, STAGE_Y + 24, "Generate", 14, WHITE, True)
# patient
box(S1X + 16, 115, S1W - 32, 70, WHITE, EDGE, 5, .6)
icon("clinical", S1X + 36, 150, TEAL, 16)
text(S1X + S1W / 2 + 8, 140, r"Patient $X_p$", 12, bold=True)
text(S1X + S1W / 2 + 8, 160, r"$t_1 \cdots t_n$", 10, MUTED)
# inputs as short nouns
for i, (lab, kind) in enumerate([
    (r"$K_D,K_E$", "record"),
    (r"$g_r$", "answer"),
    (r"$W_r$", "trend"),
]):
    cy = 215 + i * 48
    box(S1X + 16, cy - 18, S1W - 32, 36, PALE, "#D4E9E9", 4, .5)
    icon(kind, S1X + 40, cy, TEAL, 13)
    text(S1X + S1W / 2 + 10, cy, lab, 12, bold=True)
# tiny example strip
box(S1X + 16, 380, S1W - 32, 55, "#F7FAFC", EDGE, 4, .5)
text(S1X + S1W / 2, 395, "e.g. event relation", 9, MUTED)
text(S1X + S1W / 2, 415, r"$\rightarrow$ question", 10, TEAL, True)

# Stage 2: Candidates (compact chips, short nouns)
box(S2X, STAGE_Y, S2W, STAGE_H, WHITE, EDGE, 8, .8)
band(S2X + 1, STAGE_Y + 1, S2W - 2, 44, BAND_TEAL)
text(S2X + S2W / 2, STAGE_Y + 24, "Candidates", 13, TEAL, True)
for i, (lab, kind, sub, col) in enumerate([
    ("Question", "question", None, ORANGE),
    ("Answer", "answer", "hidden", TEAL),
    ("Evidence", "evidence", "hidden", TEAL),
]):
    cy = 130 + i * 95
    chip(S2X + 18, cy, S2W - 36, 70, lab, kind, col, sub)

# Stage 3: Verify
box(S3X, STAGE_Y, S3W, STAGE_H, WHITE, EDGE, 8, .8)
band(S3X + 1, STAGE_Y + 1, S3W - 2, 44, BAND_TEAL)
text(S3X + S3W / 2, STAGE_Y + 24, "Verify", 13, TEAL, True)
box(S3X + 20, 140, S3W - 40, 100, PALE, TEAL, 6, 1.0)
icon("evidence", S3X + S3W / 2, 170, TEAL, 22)
text(S3X + S3W / 2, 205, "Evidence check", 12, TEAL, True)
# re-read records
box(S3X + 30, 280, S3W - 60, 90, WHITE, EDGE, 5, .6)
text(S3X + S3W / 2, 300, "Re-read records", 11, bold=True)
for j, dx in enumerate((-40, 0, 40)):
    icon("record", S3X + S3W / 2 + dx, 340, TEAL, 14)
solid([(S3X + S3W / 2, 280), (S3X + S3W / 2, 240)], TEAL, 1.2)
text(S3X + 24, 255, "loop", 9, MUTED, align="left")

# Stage 4: Evaluate — 2x2 icon grid, short nouns
box(S4X, STAGE_Y, S4W, STAGE_H, WHITE, EDGE, 8, .8)
band(S4X + 1, STAGE_Y + 1, S4W - 2, 44, BAND_TEAL)
text(S4X + S4W / 2, STAGE_Y + 24, "Evaluate", 13, TEAL, True)
evals = [
    ("Quality", "quality", TEAL),
    ("Depth", "reasoning", TEAL),
    ("Blind solve", "solve", TEAL),
    ("Complete", "complete", TEAL),
]
for i, (lab, kind, col) in enumerate(evals):
    row, col_i = divmod(i, 2)
    cx = S4X + 30 + col_i * 170
    cy = 130 + row * 140
    box(cx, cy, 150, 110, WHITE, EDGE, 6, .7)
    icon(kind, cx + 75, cy + 40, col, 22)
    text(cx + 75, cy + 82, lab, 12, bold=True)

# Solid spine arrows between stages
solid([(S1X + S1W, 255), (S2X, 255)], TEAL, 2.0, ms=14)
solid([(S2X + S2W, 255), (S3X, 255)], TEAL, 2.0, ms=14)
solid([(S3X + S3W, 255), (S4X, 255)], TEAL, 2.0, ms=14)
text((S3X + S4X) / 2, 230, "pass", 9, MUTED)

# KD/KE feed into Generate
solid([(AX + AW, 188), (BX, 188), (BX + 20, 188), (S1X, 230)], TEAL, 1.5, ms=11)
solid([(AX + AW, 395), (BX - 10, 395), (BX - 10, 280), (S1X, 280)], TEAL, 1.5, ms=11)

# =========================================================================
# THIN SECONDARY BAND — (c) Feedback + (d) Held-out
# =========================================================================
STRIP_Y, STRIP_H = 510, 190

box(18, STRIP_Y, 1100, STRIP_H, ORANGE_FILL, ORANGE_EDGE, 10, .55)
text(36, STRIP_Y + 20, "(c) Feedback loop", 13, ORANGE, True, "left")
pass  # was: secondary tag

# Feedback Fr
box(40, 555, 320, 120, WHITE, ORANGE_EDGE, 6, .6)
band(41, 556, 318, 32, BAND_ORANGE)
icon("feedback", 62, 572, ORANGE, 14)
text(200, 572, r"Feedback $F_r$", 12, ORANGE, True)
text(200, 615, "defects · traces", 11, MUTED)
text(200, 640, "retrieval challenges", 10, MUTED)

# Update
box(400, 555, 320, 120, WHITE, ORANGE_EDGE, 6, .6)
band(401, 556, 318, 32, BAND_ORANGE)
icon("update", 422, 572, ORANGE, 14)
text(560, 572, "Update", 12, ORANGE, True)
text(560, 615, r"$g_{r+1}$  ·  $W_{r+1}$", 13, MUTED, True)
text(560, 645, r"$\theta_{r+1}$ next round", 11, MUTED)

# dashed feedback from eval/verify down, and update back to generate
dashed([(S4X + S4W / 2, STAGE_Y + STAGE_H), (S4X + S4W / 2, STRIP_Y + 20),
        (360, STRIP_Y + 20), (360, 555)], ORANGE, 1.15, arrow=True)
dashed([(S3X + S3W / 2, STAGE_Y + STAGE_H), (S3X + S3W / 2, 500),
        (200, 500), (200, 555)], ORANGE, 1.0, arrow=True)
text(S3X + S3W / 2 + 40, 485, "invalid", 9, ORANGE)
text(S4X + S4W / 2 + 50, 485, "scores", 9, ORANGE)

# dashed loop back to generate (up)
dashed([(560, 555), (560, 500), (S1X + S1W / 2, 500), (S1X + S1W / 2, STAGE_Y + STAGE_H)],
       ORANGE, 1.3, arrow=True)
text(S1X + S1W / 2 + 55, 490, r"$\theta_{r+1}$", 10, ORANGE, True)

# connector between Fr and Update
solid([(360, 615), (400, 615)], ORANGE, 1.2, ms=10)

# ---- (d) Held-out — thin right strip ----
box(1140, STRIP_Y, 560, STRIP_H, "#FCFCFD", "#D5DEE6", 10, .5)
text(1160, STRIP_Y + 20, "(d) Held-out", 13, MUTED, True, "left")
text(1280, STRIP_Y + 20, "after evolution", 9, MUTED, align="left")

steps = [
    (1180, r"Freeze $\theta^*$", "record"),
    (1360, "Held-out", "clinical"),
    (1540, "Eval", "complete"),
]
for i, (px, lab, kind) in enumerate(steps):
    box(px, 575, 140, 90, WHITE, "#D5DEE6", 6, .55)
    icon(kind, px + 70, 605, MUTED, 16)
    text(px + 70, 640, lab, 11, MUTED, True)
    if i < 2:
        solid([(px + 140, 620), (steps[i + 1][0], 620)], MUTED, 1.1, ms=9)

# from update to freeze
solid([(720, 615), (720, 680), (1160, 680), (1160, 620), (1180, 620)], MUTED, 1.0, ms=9)
text(920, 668, "final round", 9, MUTED)

for ext in ("svg", "pdf", "png"):
    path = OUT / f"medevolve-overview-v5.{ext}"
    fig.savefig(path, facecolor=WHITE, dpi=240)
    print(path)
plt.close(fig)
