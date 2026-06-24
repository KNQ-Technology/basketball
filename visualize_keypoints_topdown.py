#!/usr/bin/env python3
"""
生成篮球场俯视图，显示 99 个关键点的位置与序号。
0~90 为 7×13 网格点，91~98 为左右三秒区角点。
坐标单位：cm；X 轴沿球场长度方向，Y 轴沿宽度方向。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

CM_PER_METER = 100.0
CM_PER_FOOT = 30.48

FL = 28.65 * CM_PER_METER
FW = 15.24 * CM_PER_METER
BASKET_X = 5.25 * CM_PER_FOOT
FREE_THROW_X = 5.80 * CM_PER_METER
LANE_WIDTH = 4.90 * CM_PER_METER
CIRCLE_RADIUS = 1.80 * CM_PER_METER
THREE_POINT_RADIUS = 7.24 * CM_PER_METER
THREE_POINT_CORNER_DISTANCE = 6.70 * CM_PER_METER
THREE_POINT_SIDELINE_MARGIN = FW / 2 - THREE_POINT_CORNER_DISTANCE
THREE_POINT_LINE_X = BASKET_X + np.sqrt(
    THREE_POINT_RADIUS ** 2 - THREE_POINT_CORNER_DISTANCE ** 2
)
THREE_ARC_STOP_ANGLE = np.degrees(
    np.arcsin(THREE_POINT_CORNER_DISTANCE / THREE_POINT_RADIUS)
)


def _get_field_points():
    """Return the 99-point court template used by process_image_heuristic.py."""
    points = []
    original_length = 2800.0
    original_width = 1500.0
    original_row_y = np.array([1500.0, 1325.0, 1120.0, 885.0, 620.0, 325.0, 0.0])
    row_y = original_row_y / original_width * FW
    for y in row_y:
        for i in range(13):
            points.append([i * FL / 12, y, 0])

    lane_top_y = FW / 2 - LANE_WIDTH / 2
    lane_bottom_y = FW / 2 + LANE_WIDTH / 2
    points.extend(
        [
            [0.0, lane_top_y, 0.0],
            [FREE_THROW_X, lane_top_y, 0.0],
            [FREE_THROW_X, lane_bottom_y, 0.0],
            [0.0, lane_bottom_y, 0.0],
            [FL, lane_top_y, 0.0],
            [FL - FREE_THROW_X, lane_top_y, 0.0],
            [FL - FREE_THROW_X, lane_bottom_y, 0.0],
            [FL, lane_bottom_y, 0.0],
        ]
    )
    return np.array(points, dtype=float)


# 0~90 grid keypoints; 91~98 paint-corner keypoints.
pts = _get_field_points()   # (99, 3): x, y, z=0

# ── Row colours (7 distinct hues) ────────────────────────────────────────────
COLORS = [plt.cm.hsv(h / 7.0) for h in range(7)]

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

# ── LEFT half ────────────────────────────────────────────────────────────────
# Key area (5.8 m long, 4.9 m wide centred at midline)
lane_top_y = FW / 2 - LANE_WIDTH / 2
lane_bottom_y = FW / 2 + LANE_WIDTH / 2
line(0, lane_top_y, FREE_THROW_X, lane_top_y)
line(0, lane_bottom_y, FREE_THROW_X, lane_bottom_y)
line(FREE_THROW_X, lane_top_y, FREE_THROW_X, lane_bottom_y)
# Free throw circle
arc(FREE_THROW_X, FW/2, CIRCLE_RADIUS, 0, 360)
# Three-point straight sections (90 cm from each sideline)
line(0, THREE_POINT_SIDELINE_MARGIN, THREE_POINT_LINE_X, THREE_POINT_SIDELINE_MARGIN)
line(0, FW-THREE_POINT_SIDELINE_MARGIN, THREE_POINT_LINE_X, FW-THREE_POINT_SIDELINE_MARGIN)
# Three-point arc: from -A3° to +A3° through 0° (facing centre court)
arc(BASKET_X, FW/2, THREE_POINT_RADIUS, -THREE_ARC_STOP_ANGLE, THREE_ARC_STOP_ANGLE)
# Basket dot
ax.plot(BASKET_X, FW/2, 'o', color='orange', ms=8, zorder=4)

# ── RIGHT half ────────────────────────────────────────────────────────────────
line(FL, lane_top_y, FL-FREE_THROW_X, lane_top_y)
line(FL, lane_bottom_y, FL-FREE_THROW_X, lane_bottom_y)
line(FL-FREE_THROW_X, lane_top_y, FL-FREE_THROW_X, lane_bottom_y)
arc(FL-FREE_THROW_X, FW/2, CIRCLE_RADIUS, 0, 360)
line(FL, THREE_POINT_SIDELINE_MARGIN, FL-THREE_POINT_LINE_X, THREE_POINT_SIDELINE_MARGIN)
line(FL, FW-THREE_POINT_SIDELINE_MARGIN, FL-THREE_POINT_LINE_X, FW-THREE_POINT_SIDELINE_MARGIN)
# Three-point arc: from 180°-A3 to 180°+A3 through 180°
arc(FL-BASKET_X, FW/2, THREE_POINT_RADIUS, 180-THREE_ARC_STOP_ANGLE, 180+THREE_ARC_STOP_ANGLE)
ax.plot(FL-BASKET_X, FW/2, 'o', color='orange', ms=8, zorder=4)

# ── Keypoints ─────────────────────────────────────────────────────────────────
stroke = [withStroke(linewidth=2.5, foreground='black')]
for i, (x, y, _) in enumerate(pts):
    if i < 91:
        c = COLORS[i // 13]
        size = 480
    else:
        c = '#ff9800'
        size = 560
    ax.scatter(x, y, s=size, color=c, zorder=5, edgecolors='black', linewidths=1.0)
    ax.text(x, y, str(i), fontsize=7.5, ha='center', va='center',
            color='white', fontweight='bold', zorder=6,
            path_effects=stroke)

# ── Axis labels / ticks ───────────────────────────────────────────────────────
ax.set_xlim(-150, FL + 150)
# Y 轴翻转：y=FW 在底部（pt 0 在左下角），y=0 在顶部
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
ax.set_ylabel(f'Y  (cm,  pt0={FW:.0f} at bottom, pt78=0 at top)', color='#aaa', fontsize=9, labelpad=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Basketball Court Keypoints — Top-Down View  |  99 pts = 91 grid + 8 paint corners',
             color='white', fontsize=14, pad=12)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keypoints_topdown.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.close(fig)
