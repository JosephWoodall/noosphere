"""
Generate graphical_abstract.png for the AC-SSM JBHI submission.
Run: python graphical_abstract_gen.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
C_CSP    = "#2166AC"   # strong blue  – classical
C_NEURAL = "#4DAC26"   # muted green  – neural (all ~equiv.)
C_ACSSM  = "#D6604D"   # warm red     – this work
C_CHANCE = "#AAAAAA"   # grey         – chance
C_BG     = "#F7F7F7"
C_TITLE  = "#1A1A2E"

fig = plt.figure(figsize=(12, 6), facecolor="white")
fig.patch.set_facecolor("white")

# ── layout: 3 panels ──────────────────────────────────────────────────────────
# Panel A: static classification accuracy
ax1 = fig.add_axes([0.04, 0.14, 0.30, 0.68])
# Panel B: architecture schematic
ax2 = fig.add_axes([0.40, 0.08, 0.24, 0.76])
ax2.set_axis_off()
# Panel C: closed-loop convergence
ax3 = fig.add_axes([0.70, 0.14, 0.28, 0.68])

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A – Static LOSO Balanced Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
labels  = ["CSP+LDA", "EEGConformer", "EEGNet", "JEPA Enc", "AC-SSM\n(this work)", "Chance"]
means   = [60.1,       41.1,           38.4,     36.5,        34.7,                  33.3]
lo      = [60.1-54.7,  41.1-37.9,      38.4-35.6, 36.5-34.1,  34.7-32.4,            0]
hi      = [65.6-60.1,  44.2-41.1,      41.2-38.4, 39.2-36.5,  37.2-34.7,            0]
colors  = [C_CSP, C_NEURAL, C_NEURAL, C_NEURAL, C_ACSSM, C_CHANCE]

y_pos = np.arange(len(labels))[::-1]

bars = ax1.barh(y_pos, means, xerr=[lo, hi], align="center",
                color=colors, alpha=0.88, height=0.55,
                error_kw=dict(ecolor="#444", capsize=3, lw=1.2))

# 22 pp gap annotation
ax1.annotate("", xy=(60.1, 5.45), xytext=(37.3, 5.45),
             arrowprops=dict(arrowstyle="<->", color=C_TITLE, lw=1.4))
ax1.text(48.7, 5.62, "≈22 pp\ndata-limited gap",
         ha="center", va="bottom", fontsize=7.5, color=C_TITLE, fontstyle="italic")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=8.5)
ax1.set_xlabel("Balanced Accuracy (%)", fontsize=8.5)
ax1.set_xlim(20, 72)
ax1.axvline(33.3, color=C_CHANCE, lw=1.2, ls="--", alpha=0.7)
ax1.text(33.6, -0.6, "chance", fontsize=7, color=C_CHANCE, va="top")
ax1.set_facecolor(C_BG)
ax1.tick_params(axis="both", labelsize=8)
ax1.spines[["top","right"]].set_visible(False)
ax1.set_title("(A) Static Classification\n(LOSO, 20 subjects, 3-class MI)",
              fontsize=9, fontweight="bold", pad=6, color=C_TITLE)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B – Architecture Schematic
# ═══════════════════════════════════════════════════════════════════════════════
def box(ax, xy, w, h, label, sublabel="", fc="#DDEEFF", ec="#2166AC", fs=8, sfs=7):
    x, y = xy
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                          fc=fc, ec=ec, lw=1.4, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.03 if sublabel else 0), label,
            ha="center", va="center", fontsize=fs, fontweight="bold",
            color=C_TITLE, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.04, sublabel,
                ha="center", va="center", fontsize=sfs, color="#444", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#444", lw=1.3, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=10), zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.04, my, label, fontsize=7, color=color, va="center")

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("auto")

# EEG input
box(ax2, (0.05, 0.80), 0.90, 0.13, "EEG Window", "C=21 ch, T=1024 samp",
    fc="#E8F4FD", ec=C_CSP)

# ZOH-SSM encoder
box(ax2, (0.05, 0.55), 0.90, 0.18, "ZOH-SSM Encoder",
    "bilinear recurrence  ·  L=4 layers  ·  d=128",
    fc="#FFF3CD", ec="#F0A500")

# action-conditioning injection arrow from bottom
ax2.annotate("", xy=(0.50, 0.555), xytext=(0.50, 0.38),
             arrowprops=dict(arrowstyle="-|>", color=C_ACSSM, lw=2.0,
                             mutation_scale=12), zorder=5)
ax2.text(0.52, 0.465, "$\\mathbf{a}_{t-1}$\n(prev. cmd)",
         fontsize=8, color=C_ACSSM, va="center", fontweight="bold")

# motor command box
box(ax2, (0.05, 0.26), 0.90, 0.12, "Action-Conditioned Latent  $\\mathbf{z}_t$",
    "SiLU-gated motor context injection",
    fc="#FFE4E1", ec=C_ACSSM)

# decoder
box(ax2, (0.05, 0.08), 0.90, 0.12, "IntentDecoder  +  CEM-MPC Planner",
    "6-DOF velocity  ·  latency-minimising control",
    fc="#E9F5E9", ec=C_NEURAL)

# vertical flow arrows
arrow(ax2, 0.50, 0.93, 0.50, 0.73)
arrow(ax2, 0.50, 0.55, 0.50, 0.38)
arrow(ax2, 0.50, 0.26, 0.50, 0.20)

# JEPA pretrain annotation
ax2.text(0.50, 1.01, "JEPA pre-training  →  fine-tuned with recon + alignment",
         ha="center", va="bottom", fontsize=7.5, color="#555", fontstyle="italic")

ax2.set_title("(B) AC-SSM Architecture",
              fontsize=9, fontweight="bold", y=1.04, color=C_TITLE)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C – Closed-Loop Convergence
# ═══════════════════════════════════════════════════════════════════════════════
cl_labels = ["Oracle\n(perfect)", "AC-SSM\n(this work)", "CSP+LDA", "JEPA\n(no action\ncond.)"]
cl_vals   = [100.0, 33.8, 26.3, 0.0]
cl_colors = ["#777777", C_ACSSM, C_CSP, C_NEURAL]

x_pos = np.arange(len(cl_labels))
bars3 = ax3.bar(x_pos, cl_vals, color=cl_colors, alpha=0.88, width=0.6, zorder=3)

for bar, val in zip(bars3, cl_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=C_TITLE)

# significance brackets
def sig_bracket(ax, x1, x2, y, label, dy=3):
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=1.2, color=C_TITLE)
    ax.text((x1+x2)/2, y+dy+0.5, label, ha="center", va="bottom",
            fontsize=8, color=C_TITLE)

sig_bracket(ax3, 0.7, 1.3, 37.0, "p = 0.21 (n.s.)", dy=4)   # AC-SSM vs CSP
sig_bracket(ax3, 1.7, 3.3, 48.0, "p < 0.0001", dy=4)         # CSP or AC vs JEPA

ax3.set_xticks(x_pos)
ax3.set_xticklabels(cl_labels, fontsize=8.5)
ax3.set_ylabel("Closed-Loop Convergence Rate (%)", fontsize=8.5)
ax3.set_ylim(0, 60)
ax3.set_facecolor(C_BG)
ax3.spines[["top","right"]].set_visible(False)
ax3.tick_params(axis="both", labelsize=8)
ax3.set_title("(C) Closed-Loop Virtual Arm\n(133 trials, 10 subjects, 6-DOF)",
              fontsize=9, fontweight="bold", pad=6, color=C_TITLE)

# ═══════════════════════════════════════════════════════════════════════════════
# Global title
# ═══════════════════════════════════════════════════════════════════════════════
fig.text(0.50, 0.98,
         "Action-Conditioned ZOH-SSM World Model for Closed-Loop BCI",
         ha="center", va="top", fontsize=11, fontweight="bold", color=C_TITLE)
fig.text(0.50, 0.94,
         "Static accuracy is data-limited, not architecture-limited.  "
         "Action conditioning converts 0% → 33.8% closed-loop convergence.",
         ha="center", va="top", fontsize=8.5, color="#333", fontstyle="italic")

out = "graphical_abstract.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
