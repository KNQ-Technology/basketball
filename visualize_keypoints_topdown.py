#!/usr/bin/env python3
"""
生成篮球场俯视图，显示 91 个关键点（7×13 网格）的位置与序号。
坐标单位：cm；X 轴沿球场长度方向（0→2800），Y 轴沿宽度方向（0→1500）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

FL = 2800   # court length (cm)
FW = 1500   # court width  (cm)


def _get_field_points():
    """从 viewds.py 内联的关键点生成逻辑，避免依赖 yacs 等训练时库。"""
    points = []
    u0, r, u, s = 175, 30, 175, 0
    for _ in range(7):
        for i in range(13):
            points.append([i * FL / 12, FW - s, 0])
        s += u
        u += r
    return np.array(points, dtype=float)


# 91 grid keypoints (7 rows × 13 cols)
pts = _get_field_points()   # (91, 3): x, y, z=0

# ── Row colours (7 distinct hues) ────────────────────────────────────────────
COLORS = [plt.cm.hsv(h / 7.0) for h in range(7)]

# ── Row Y-coordinates ─────────────────────────────────────────────────────────
row_y = []
u, s = 175, 0
for _ in range(7):
    row_y.append(FW - s)
    s += u
    u += 30

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 12))
ax.set_facecolor('#2e7d32')          # dark green court
fig.patch.set_facecolor('#1b1b1b')   # dark background

LC = 'white'
LW = 2.2

def line(x0, y0, x1, y1, **kw):
    ax.plot([x0, x1], [y0, y1], color=LC, lw=LW, zorder=2,
            solid_capstyle='round', **kw)

def arc(cx, cy, r, a1, a2, **kw):
    """Counterclockwise arc from a1° to a2° (a2 > a1 always)."""
    if a2 <= a1:
        a2 += 360
    th = np.linspace(np.radians(a1), np.radians(a2), 600)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
            color=LC, lw=LW, zorder=2, **kw)

# ── Court boundary ────────────────────────────────────────────────────────────
line(0, 0, FL, 0)
line(FL, 0, FL, FW)
line(FL, FW, 0, FW)
line(0, FW, 0, 0)

# ── Centre line & circle ──────────────────────────────────────────────────────
line(FL/2, 0, FL/2, FW)
arc(FL/2, FW/2, 180, 0, 360)

# ── Three-point arc angle (exact) ─────────────────────────────────────────────
# Straight section ends at x=299, y=90 from left basket at (157.5, 750)
# dx=141.5, dy=660 → angle = arctan2(660, 141.5)
A3 = np.degrees(np.arctan2(FW/2 - 90, 299 - 157.5))   # ≈ 77.87°

# ── LEFT half ────────────────────────────────────────────────────────────────
# Key area (5.8 m long, 4.9 m wide centred at midline)
line(0,  505, 580,  505)
line(0, FW-505, 580, FW-505)
line(580, 505, 580, FW-505)
# Free throw circle
arc(580, FW/2, 180, 0, 360)
# Three-point straight sections (90 cm from each sideline)
line(0,    90, 299,    90)
line(0, FW-90, 299, FW-90)
# Three-point arc: from -A3° to +A3° through 0° (facing centre court)
arc(157.5, FW/2, 675, -A3, A3)
# Basket dot
ax.plot(157.5, FW/2, 'o', color='orange', ms=8, zorder=4)

# ── RIGHT half ────────────────────────────────────────────────────────────────
line(FL,   505, FL-580,   505)
line(FL, FW-505, FL-580, FW-505)
line(FL-580, 505, FL-580, FW-505)
arc(FL-580, FW/2, 180, 0, 360)
line(FL,    90, FL-299,    90)
line(FL, FW-90, FL-299, FW-90)
# Three-point arc: from 180°-A3 to 180°+A3 through 180°
arc(FL-157.5, FW/2, 675, 180-A3, 180+A3)
ax.plot(FL-157.5, FW/2, 'o', color='orange', ms=8, zorder=4)

# ── Keypoints ─────────────────────────────────────────────────────────────────
stroke = [withStroke(linewidth=2.5, foreground='black')]
for i, (x, y, _) in enumerate(pts):
    row = i // 13
    c = COLORS[row]
    ax.scatter(x, y, s=480, color=c, zorder=5, edgecolors='black', linewidths=1.0)
    ax.text(x, y, str(i), fontsize=7.5, ha='center', va='center',
            color='white', fontweight='bold', zorder=6,
            path_effects=stroke)

# ── Axis labels / ticks ───────────────────────────────────────────────────────
ax.set_xlim(-150, FL + 150)
# Y 轴翻转：y=1500 在底部（pt 0 在左下角），y=0 在顶部
ax.set_ylim(FW + 150, -150)
ax.set_aspect('equal')

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(colors='#aaa', labelsize=8)
ax.set_xticks(np.arange(0, FL+1, 200))
ax.set_yticks(np.arange(0, FW+1, 100))
ax.tick_params(axis='x', colors='#888')
ax.tick_params(axis='y', colors='#888')
ax.set_xlabel('X  (cm,  left baseline -> right baseline)', color='#aaa', fontsize=9, labelpad=6)
ax.set_ylabel('Y  (cm,  pt0=1500 at bottom, pt78=0 at top)', color='#aaa', fontsize=9, labelpad=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Basketball Court Keypoints — Top-Down View  |  91 pts = 7 rows x 13 cols',
             color='white', fontsize=14, pad=12)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keypoints_topdown.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.close(fig)
