"""绘制 MedEvolve 方法主图，导出可编辑 SVG、矢量 PDF 和 PNG。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import cos, sin, pi

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse

OUT = Path(__file__).resolve().parent
INK, MUTED = "#17324D", "#52677B"
TEAL, ORANGE = "#087F83", "#C45B25"
EDGE, PALE, WARM = "#CEDCE5", "#EFF7F8", "#FFF5ED"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(17, 7.6))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, 1700), ylim=(760, 0))
ax.axis("off")


def box(x, y, w, h, fill="white", edge=EDGE, radius=9, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))


def title_band(x, y, w, h, fill):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                          facecolor=fill, edgecolor="none", zorder=2)
    ax.add_patch(patch)
    ax.add_patch(Rectangle((x, y+h-8), w, 8, facecolor=fill, edgecolor="none", zorder=2))


def icon(kind, x, y, color=TEAL, size=18):
    """以路径绘制语义图标，避免字体符号和位图依赖。"""
    def line(points, width=1.25):
        ax.plot([x+px*size for px, _ in points], [y+py*size for _, py in points],
                color=color, lw=width, zorder=4, solid_capstyle="round", solid_joinstyle="round")

    def circle(cx, cy, r, fill="none"):
        ax.add_patch(Circle((x+cx*size, y+cy*size), r*size,
                            facecolor=fill, edgecolor=color, lw=1.25, zorder=4))

    if kind == "search":
        circle(-.10, -.10, .36)
        line([(.18,.18),(.48,.48)], 1.8)
    elif kind in ("record", "question", "answer", "evidence"):
        line([(-.36,-.5),(.14,-.5),(.36,-.28),(.36,.5),(-.36,.5),(-.36,-.5)])
        line([(.14,-.5),(.14,-.28),(.36,-.28)])
        if kind == "question":
            line([(-.12,-.10),(-.06,-.23),(.1,-.23),(.16,-.10),(0,.04),(0,.14)])
            circle(0,.32,.025,color)
        elif kind == "answer":
            line([(-.18,.02),(-.03,.18),(.23,-.09)], 1.6)
        else:
            for dy in (-.15,.05,.25):
                line([(-.2,dy),(.18,dy)])
            if kind == "evidence":
                circle(.28,.27,.23)
                line([(.44,.44),(.58,.58)])
    elif kind == "clinical":
        line([(-.38,-.48),(.38,-.48),(.38,.48),(-.38,.48),(-.38,-.48)])
        line([(-.19,-.04),(.19,-.04)], 1.7)
        line([(0,-.23),(0,.15)], 1.7)
        line([(-.20,.31),(.20,.31)])
    elif kind in ("trend", "feedback"):
        for dx, height in [(-.35,.28),(-.05,.48),(.25,.70)]:
            ax.add_patch(Rectangle((x+dx*size,y+(.4-height)*size), .16*size,height*size,
                                  facecolor=color, edgecolor="none", zorder=4))
        if kind == "feedback":
            circle(.39,.30,.25)
            line([(.55,.46),(.72,.63)], 1.6)
    elif kind == "tools":
        line([(-.43,.37),(-.28,.49),(.22,-.01),(.41,-.06),(.5,-.24),(.46,-.44),
              (.29,-.25),(.14,-.38),(.31,-.55),(.08,-.5),(-.03,-.28),(.01,-.12),(-.43,.37)])
    elif kind == "quality":
        pts=[]
        for i in range(10):
            angle=-pi/2+i*pi/5
            radius=.5 if i%2==0 else .23
            pts.append((x+cos(angle)*size*radius,y+sin(angle)*size*radius))
        ax.add_patch(Polygon(pts, facecolor=color, edgecolor=color, lw=.7, zorder=4))
    elif kind == "reasoning":
        for a,b in [((-.36,.28),(0,-.3)),((0,-.3),(.37,.05)),((-.36,.28),(.37,.05))]:
            line([a,b])
        for cx,cy in [(-.36,.28),(0,-.3),(.37,.05)]:
            circle(cx,cy,.12,"white")
    elif kind in ("solve", "update"):
        circle(0,0,.32)
        circle(0,0,.12)
        for i in range(8):
            angle=i*pi/4
            line([(.34*cos(angle),.34*sin(angle)),(.5*cos(angle),.5*sin(angle))],1.6)
    elif kind == "complete":
        circle(0,0,.48,color)
        ax.plot([x-.25*size,x-.06*size,x+.28*size],
                [y,y+.19*size,y-.22*size],color="white",lw=1.7,zorder=5)
    elif kind == "database":
        ax.add_patch(Rectangle((x-.4*size,y-.3*size), .8*size,.65*size,
                              facecolor="white",edgecolor=color,lw=1.2,zorder=3))
        for dy in (-.3,0,.35):
            ax.add_patch(Ellipse((x,y+dy*size),.8*size,.3*size,
                                facecolor="white" if dy==-.3 else "none",edgecolor=color,lw=1.2,zorder=4))


def text(x, y, value, size=14, color=INK, bold=False, align="center"):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight="bold" if bold else "normal",
                   ha=align, va="center", zorder=5, linespacing=1.35)


def route(points, color=INK, lw=1.6, arrow=True):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot((a[0], b[0]), (a[1], b[1]), color=color, lw=lw, zorder=3,
                solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>" if arrow else "-",
                                mutation_scale=13, linewidth=lw, color=color,
                                shrinkA=0, shrinkB=0, zorder=3))


def record(x, y, color=TEAL):
    ax.add_patch(Rectangle((x, y), 20, 26, facecolor="white", edgecolor=color, lw=1, zorder=3))
    for d in (7, 13, 19):
        ax.plot((x+5, x+15), (y+d, y+d), color=color, lw=1, zorder=3)


def timeline(x, y, width):
    route([(x, y+35), (x+width, y+35)], lw=1)
    for i, fraction in enumerate((0.07, 0.34, 0.61, 0.88)):
        px = x+fraction*width
        icon(["clinical","record","trend","record"][i], px, y+13, size=24)
        ax.plot(px, y+35, "o", color=INK, markersize=3, zorder=4)
        text(px, y+53, [r"$t_1$", r"$t_2$", r"$t_3$", r"$t_n$"][i], 11, MUTED)


def section(x, y, w, h, title, color=TEAL):
    box(x, y, w, h, "#FBFDFE" if color == TEAL else "#FFFCF9", EDGE, 12, .8)
    text(x+20, y+25, title, 18, color, True, "left")


section(20, 20, 400, 720, "(a) Dual exploration")
section(460, 20, 1220, 405, "(b) Task generation & evaluation")
section(460, 465, 850, 275, "(c) Strategy evolution", ORANGE)
box(1340, 465, 340, 275, "#FBFDFE", EDGE, 12, .8)
text(1510, 493, "Held-out evaluation", 17, INK, True)
text(1510, 518, "After evolution", 12, MUTED)

# 初始探索知识仅来自进化患者，后续固定复用。
box(45, 78, 350, 103)
text(220, 100, "Evolution patient records", 16, INK, True)
timeline(70, 116, 300)
route([(220,181), (220,205)], TEAL)
box(65, 205, 310, 77, PALE, TEAL)
title_band(66, 206, 308, 38, "#DDEFF0")
icon("search", 90, 229, size=19)
text(234, 229, "Data exploration", 16, TEAL, True)
text(220, 256, "Event relations · evidence patterns", 12, MUTED)
route([(220,282), (220,305)], TEAL)
box(85, 305, 270, 52, PALE, TEAL)
icon("record", 109, 331, size=22)
text(235, 331, r"Data knowledge  $K_D$", 14, TEAL, True)

box(45, 395, 350, 87)
text(220, 417, "Read-only tools", 16, INK, True)
for px, label, kind in [(110,"Search","search"),(220,"Read","record"),(330,"FHIR records","database")]:
    icon(kind, px, 447, size=24)
    text(px, 470, label, 12, MUTED)
route([(220,482), (220,505)], TEAL)
box(65, 505, 310, 77, PALE, TEAL)
title_band(66, 506, 308, 38, "#DDEFF0")
icon("tools", 89, 529, size=18)
text(234, 529, "Environment exploration", 13, TEAL, True)
text(220, 556, "Tool capabilities · access constraints", 11, MUTED)
route([(220,582), (220,605)], TEAL)
box(85, 605, 270, 52, PALE, TEAL)
icon("record", 107, 631, size=22)
text(235, 621, "Environment knowledge", 12, TEAL, True)
text(220, 645, r"$K_E$", 14, TEAL, True)
text(220, 702, "Fixed across rounds", 13, TEAL, True)

# 患者输入紧贴生成器，任务卡与评价模块共享顶部基线。
box(495, 78, 280, 94)
text(635, 98, r"Current patient  $X_p$", 16, INK, True)
timeline(520, 114, 230)
route([(635,172), (635,198)], TEAL)
box(495, 198, 280, 177, PALE, TEAL, lw=1.6)
title_band(496, 199, 278, 62, TEAL)
icon("clinical", 518, 230, "white", 22)
text(645, 223, "Knowledge-guided", 15, "white", True)
text(645, 247, "generation", 16, "white", True)
for py, label, kind in [(285,r"Knowledge  $K_D, K_E$","record"),
                       (318,r"Guidance  $g_r$","answer"),
                       (351,r"Direction weights  $w_r$","trend")]:
    box(507, py-14, 256, 28, "white", "#D4E9E9", 4, .5)
    icon(kind, 525, py, size=14)
    text(647, py, label, 12)
route([(355,331),(438,331),(438,285),(495,285)], TEAL)
route([(355,631),(438,631),(438,331)], TEAL, arrow=False)

box(816, 98, 220, 283, "#E6EFF3", EDGE, 9, .7)
box(813, 95, 220, 283, "#F1F6F8", EDGE, 9, .7)
box(810, 92, 220, 283)
title_band(811, 93, 218, 43, "#E8F0F5")
text(920, 117, "Candidate tasks", 16, INK, True)
for py, title, subtitle in [(149,"Question",""),(216,"Reference answer","Withheld from solver"),
                             (290,"Evidence","Withheld from solver")]:
    box(825, py, 190, 57, "#F7FAFC", EDGE, 5, .7)
    icon("question" if not subtitle else "answer" if title == "Reference answer" else "evidence",
         841, py+18 if subtitle else py+27, ORANGE if not subtitle else TEAL, 18)
    text(934, py+18 if subtitle else py+28, title, 11 if subtitle else 14, bold=True)
    if subtitle:
        text(920, py+41, subtitle, 10, MUTED)
route([(775,270),(810,270)])

box(1065, 150, 225, 90, PALE, TEAL)
icon("evidence", 1092, 195, TEAL, 28)
text(1192, 183, "Evidence", 17, TEAL, True)
text(1192, 210, "verification", 16, TEAL, True)
route([(1030,195),(1065,195)])
box(1080, 295, 195, 80)
text(1177.5, 313, "Original records", 13, bold=True)
for px in (1130,1168,1206):
    record(px, 332)
route([(1177.5,295),(1177.5,240)])
text(1085, 270, "Re-read", 10, MUTED, align="left")

box(1330, 92, 325, 283, PALE, TEAL)
title_band(1331, 93, 323, 38, "#DDEFF0")
text(1492.5, 117, "Task evaluation", 16, TEAL, True)
for py, label, kind in [(156,"Quality","quality"),(201,"Reasoning depth","reasoning"),
                       (246,"Blind solving","solve"),(324,"Completion","complete")]:
    box(1347, py-18, 291, 38 if py != 246 else 65, "white", EDGE, 5, .7)
    icon(kind, 1371, py, ORANGE if kind in ("quality","solve") else TEAL, 20)
    text(1507, py, label, 14, bold=True)
text(1492.5, 274, "Question only · fixed solver", 11, MUTED)
route([(1290,195),(1330,195)])
text(1310, 171, "Pass", 10, MUTED)

# 评价结果与核验失败均进入反馈，独立评测没有回流边。
box(495, 550, 315, 110, WARM, ORANGE)
title_band(496, 551, 313, 37, "#F8E1CF")
icon("update", 521, 575, ORANGE, 21)
text(667.5, 575, "Strategy update", 16, ORANGE, True)
for py, label, kind in [(610,r"Generation guidance  $g_{r+1}$","answer"),
                       (641,r"Direction weights  $w_{r+1}$","trend")]:
    icon(kind, 521, py, ORANGE, 16)
    text(668, py, label, 12)
box(955, 550, 325, 110, WARM, ORANGE)
title_band(956, 551, 323, 37, "#F8E1CF")
icon("feedback", 978, 575, ORANGE, 19)
text(1130, 575, r"Feedback attribution  $F_r$", 14, ORANGE, True)
icon("evidence", 977, 611, ORANGE, 15)
icon("reasoning", 977, 640, ORANGE, 15)
text(1130, 611, "Generation defects", 14)
text(1130, 640, "Retrieval & reasoning challenges", 12)
route([(955,605),(810,605)], ORANGE)
route([(800,550),(800,397),(720,397),(720,375)], ORANGE, 2)
text(910, 514, r"Next round: $\theta_{r+1}$", 11, ORANGE, True)
route([(1290,230),(1310,230),(1310,450)], ORANGE, arrow=False)
text(1225, 429, "Invalid reasons", 11, ORANGE)
route([(1492.5,375),(1492.5,450)], ORANGE, arrow=False)
route([(1492.5,450),(1210,450),(1210,550)], ORANGE)
ax.add_patch(Circle((1310,450), 3, facecolor=ORANGE, edgecolor="none", zorder=4))
text(1510, 429, "Scores & solve traces", 11, ORANGE)

box(1360, 550, 300, 42)
icon("record", 1382, 571, size=19)
text(1523, 571, r"Frozen strategy  $\theta^*$", 14, bold=True)
box(1360, 616, 300, 42)
text(1510, 637, "Held-out patients", 14)
box(1360, 682, 300, 42)
icon("complete", 1382, 703, size=19)
text(1523, 703, "Independent evaluation", 13, bold=True)
route([(1510,592),(1510,616)])
route([(1510,658),(1510,682)])
route([(652.5,660),(652.5,705),(1325,705),(1325,571),(1360,571)])
text(985, 692, "After final round", 12, MUTED)

for extension in ("svg", "pdf", "png"):
    path = OUT / f"medevolve-overview-v4.{extension}"
    fig.savefig(path, facecolor="white", dpi=240)
    print(path)
plt.close(fig)
