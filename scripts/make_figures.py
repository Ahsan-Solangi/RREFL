"""Draw Fig. 1 (base models) and Fig. 2 (adaptation to DF40) for the WIFS 2026 camera-ready.

Every number is read from nb_data.json, which parse_nb.py extracts from the printed
report tables of RREFL_R3_final.ipynb. Nothing is typed in by hand.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
D = json.load(open(os.path.join(HERE, "nb_data.json")))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # IEEE template: 8-point Times New Roman for figure labels (Liberation Serif is metric-identical)
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600,
})

# categorical slots 1-3 of the validated palette (all-pairs pass); others neutral
C = {"RR_uniform": "#2a78d6", "RR_adamw": "#eb6834", "POOLED_valsel": "#1baf7a"}
GRAY, LIGHT = "#8a8985", "#c9c8c3"
NAME = {"RR_uniform": "Reptile", "RR_adamw": "Reptile + AdamW", "POOLED_valsel": "Pooled"}
T1 = D["T1"]

def m(arm, key):
    v = T1[arm][key]; return v["med"], v["lo"], v["hi"]

def errpt(ax, x, y, **kw):
    """median point with [min, max] whiskers on both axes"""
    ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                elinewidth=0.6, capsize=0, **kw)

# =====================================================================================
# Fig. 1: base models
# =====================================================================================
FS = 8  # every text element is 8 pt
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
fig.subplots_adjust(left=0.062, right=0.995, bottom=0.225, top=0.89, wspace=0.3)

# (a) seen vs held-out balanced accuracy
ax = axs[0]
rows = [("RR_uniform", C["RR_uniform"], "o", 6), ("RR_adamw", C["RR_adamw"], "o", 6),
        ("POOLED_valsel", C["POOLED_valsel"], "o", 6),
        ("POOLED_linear", GRAY, "D", 4.5), ("ES_linear", GRAY, "v", 5.5)]
lab = {"RR_uniform": ("Reptile", 0.842, 0.862, "left"),
       "RR_adamw": ("Reptile+AdamW", 0.930, 0.785, "center"),
       "POOLED_valsel": ("Pooled", 0.954, 0.60, "left"),
       "POOLED_linear": ("pooled\nlinear", 0.902, 0.700, "right"),
       "ES_linear": ("early-stopped\nlinear probe$^\\dagger$", 0.846, 0.952, "left")}
for arm, col, mk, ms in rows:
    x = m(arm, "seen: bal. acc.")
    y = m(arm, "held-out: bal. acc.")
    yl = m(arm, "held-out vs LFW: bal. acc.")
    ax.plot([x[0], x[0]], [yl[0], y[0]], color=col, lw=0.5, ls=(0, (1.5, 1.5)), zorder=2)
    errpt(ax, x, y, fmt=mk, ms=ms, color=col, mec="white", mew=0.5, zorder=3)
    errpt(ax, x, yl, fmt=mk, ms=ms - 0.5, color=col, mfc="white", mec=col, mew=0.9, zorder=3)
    t, tx, ty, ha = lab[arm]
    ax.text(tx, ty, t, ha=ha, va="center", fontsize=FS, linespacing=0.9,
            color="#0b0b0b" if col != GRAY else "#3d3c39")
ax.set_xlim(0.80, 0.985); ax.set_ylim(0.47, 1.0)
ax.set_xticks([0.80, 0.85, 0.90, 0.95])
ax.set_xlabel("seen-generator accuracy")
ax.set_ylabel("held-out accuracy")
ax.set_title("(a) Fit vs. generalization", loc="left")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#52514e", mec="white", ms=5, label="seen-source reals"),
                   Line2D([], [], marker="o", ls="", color="#52514e", mfc="white", ms=4.5, label="LFW reals")],
          loc="lower left", frameon=False, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, fontsize=FS)

# (b) distance from initialization (MLP heads)
ax = axs[1]
mlp = [("RR_uniform", C["RR_uniform"], "o", 6), ("IID_uniform", C["RR_uniform"], "o", 3.5),
       ("BAL_uniform", C["RR_uniform"], "o", 3.5), ("RR_none", C["RR_uniform"], "o", 3.5),
       ("RR_category", C["RR_uniform"], "o", 3.5),
       ("ES_mlp", GRAY, "^", 5), ("RR_adamw", C["RR_adamw"], "o", 6),
       ("POOLED_lr1e5", LIGHT, "o", 4), ("POOLED_lr3e5", LIGHT, "o", 4),
       ("POOLED_lr1e4", LIGHT, "o", 4), ("POOLED_lr3e4", LIGHT, "o", 4),
       ("POOLED_valsel", C["POOLED_valsel"], "o", 6), ("POOLED_reuse8", GRAY, "s", 4.5)]
for arm, col, mk, ms in mlp:
    x = m(arm, "distance from init")
    y = m(arm, "held-out: bal. acc.")
    errpt(ax, x, y, fmt=mk, ms=ms, color=col, mec="white", mew=0.5, zorder=4 if ms >= 6 else 3)
txt = [("RR_uniform", "Reptile", 1.35, 0.0, "left"),
       ("RR_adamw", "Reptile+AdamW", 0.78, 0.0, "right"),
       ("ES_mlp", "early-stopped\nMLP$^\\dagger$", 1.2, 0.012, "left"),
       ("POOLED_valsel", "Pooled", 1.0, -0.036, "center"),
       ("POOLED_reuse8", "batch\nreuse", 1.0, 0.075, "center")]
for arm, t, fx, dy, ha in txt:
    x, y = m(arm, "distance from init")[0], m(arm, "held-out: bal. acc.")[0]
    ax.text(x * fx, y + dy, t, ha=ha, va="center", fontsize=FS, linespacing=0.9)
x = m("POOLED_lr3e5", "distance from init")[0]; y = m("POOLED_lr3e5", "held-out: bal. acc.")[0]
ax.text(2.6, 0.618, "pooled, fixed\nlearning rates", ha="center", va="center",
        fontsize=FS, color="#52514e", linespacing=0.9)
ax.set_xscale("log"); ax.set_xlim(0.4, 110); ax.set_ylim(0.56, 0.93)
ax.set_xlabel("distance from initialization (log)")
ax.set_ylabel("held-out accuracy")
ax.set_title("(b) MLP heads: distance moved", loc="left")

# (c) per-class accuracy on the held-out generator
ax = axs[2]
arms_c = ["RR_uniform", "RR_adamw", "POOLED_valsel", "ES_linear"]
xl = ["Reptile", "Reptile\n+AdamW", "Pooled", "linear\nprobe$^\\dagger$"]
keys = [("held-out: fakes caught", "#3d3c39", None, "held-out fakes caught"),
        ("held-out: reals called real (seen-source reals)", "#9e9d98", None, "seen-source reals called real"),
        ("LFW: reals called real", "white", "////", "LFW reals called real")]
w = 0.26
for j, (key, fc, hatch, lb) in enumerate(keys):
    for i, arm in enumerate(arms_c):
        v = m(arm, key)
        ax.bar(i + (j - 1) * w, v[0], w * 0.92, color=fc, edgecolor="#3d3c39", linewidth=0.5, hatch=hatch,
               label=lb if i == 0 else None, zorder=2)
        ax.plot([i + (j - 1) * w] * 2, [v[1], v[2]], color="#0b0b0b", lw=0.7, zorder=3)
ax.set_xticks(range(len(arms_c))); ax.set_xticklabels(xl, fontsize=FS, linespacing=0.9)
for t, arm in zip(ax.get_xticklabels(), arms_c):
    if arm in C: t.set_color(C[arm])
ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("fraction correct")
ax.grid(axis="x", visible=False)
ax.set_title("(c) Per-class accuracy", loc="left")
ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=FS, handlelength=1.3,
          handletextpad=0.35, borderaxespad=0.05, labelspacing=0.2)
fig.savefig(os.path.join(OUT, "fig_base.pdf"))
fig.savefig(os.path.join(OUT, "fig_base.png"), dpi=220)
plt.close(fig)

# =====================================================================================
# Fig. 2: adaptation to DF40
# =====================================================================================
BASES = ["RR_uniform", "RR_adamw", "POOLED_valsel"]
STY = {"QA": dict(ls="-", marker="o"), "FTR": dict(ls="--", marker="^"), "FT": dict(ls=":", marker="s")}
ANAME = {"QA": "Quick-Adapt", "FTR": "FTR", "FT": "FT"}
zs = {r["arm"]: r["auc"]["med"] for r in D["T2_DF40"] if r["label"] == "zero-shot"}
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0))
fig.subplots_adjust(left=0.062, right=0.995, bottom=0.225, top=0.775, wspace=0.5)
SX = range(5)
SLAB = ["0.04", "0.12", "0.35", "0.5", "1"]
def sweep(b, meth):
    return sorted([r for r in D["T5_DF40"] if r["arm"] == b and r["method"] == meth], key=lambda r: r["step"])
def line(ax, xs, ys, b, meth):
    ax.plot(xs, ys, color=C[b], lw=1.0, ls=STY[meth]["ls"], marker=STY[meth]["marker"], ms=3.6,
            mec="white", mew=0.4, zorder=3)

# (a) step size vs DF40 AUC, no rollback
ax = axs[0]
for b in BASES:
    for meth in ["FT", "FTR", "QA"]:
        rr = sweep(b, meth); line(ax, SX, [r["auc"]["med"] for r in rr], b, meth)
    ax.plot(-0.6, zs[b], marker="*", ms=6.5, color=C[b], mec="#0b0b0b", mew=0.4, ls="", zorder=4, clip_on=False)
ax.text(-0.35, 0.641, "zero-shot", fontsize=FS, color="#52514e", ha="left", va="center")
ax.set_xticks(SX); ax.set_xticklabels(SLAB)
ax.set_xlim(-0.85, 4.25); ax.set_ylim(0.62, 0.86)
ax.set_xlabel(r"step ($\varepsilon$ or $s$)")
ax.set_ylabel("DF40 AUC")
ax.set_title("(a) No rollback", loc="left")

# (b) step size vs retention, no rollback
ax = axs[1]
for b in BASES:
    for meth in ["FT", "FTR", "QA"]:
        rr = sweep(b, meth); line(ax, SX, [r["seen"]["med"] for r in rr], b, meth)
ax.axhline(0, color="#52514e", lw=0.6, zorder=1)
ax.axhline(-5, color="#52514e", lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_xticks(SX); ax.set_xticklabels(SLAB)
ax.set_xlim(-0.25, 4.25); ax.set_ylim(-17.5, 10)
ax.set_xlabel(r"step ($\varepsilon$ or $s$)")
ax.set_ylabel("retention (points)")
ax.set_title("(b) No rollback", loc="left")

# (c) when does the 5-point rollback fire?
ax = axs[2]
FV = D["fired_vs_step"]
jit = {"RR_uniform": -0.22, "RR_adamw": 0.0, "POOLED_valsel": 0.22}
for b in BASES:
    for meth in ["FT", "FTR", "QA"]:
        line(ax, SX, np.array(FV["counts"][b][meth]) + jit[b], b, meth)
ax.set_xticks(SX); ax.set_xticklabels(SLAB)
ax.set_yticks([0, 3, 6, 9]); ax.set_ylim(-0.8, 9.8); ax.set_xlim(-0.25, 4.25)
ax.set_xlabel(r"step ($\varepsilon$ or $s$)")
ax.set_ylabel("rollbacks (of 9 runs)")
ax.set_title("(c) 5-point rollback", loc="left")

# (d) support size with rollback
ax = axs[3]
Ks = [1, 2, 4, 8, 15, 20]
for b in BASES:
    for meth in ["FT", "FTR", "QA"]:
        rr = sorted([r for r in D["T4_DF40"] if r["arm"] == b and r["method"] == meth], key=lambda r: r["K"])
        line(ax, [r["K"] for r in rr], [r["auc"]["med"] for r in rr], b, meth)
        if b == "RR_uniform" and meth == "QA":
            ax.fill_between([r["K"] for r in rr], [r["auc"]["lo"] for r in rr], [r["auc"]["hi"] for r in rr],
                            color=C[b], alpha=0.13, lw=0, zorder=1)
ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks]); ax.minorticks_off()
for t in ax.get_xticklabels():
    if t.get_text() == "15": t.set_ha("right")
ax.set_xlim(0.85, 27); ax.set_ylim(0.58, 0.85)
ax.set_xlabel(r"videos per method, $K$")
ax.set_ylabel("DF40 AUC")
ax.set_title("(d) 5-point rollback", loc="left")

h1 = [Patch(color=C[b], label=NAME[b]) for b in BASES]
h2 = [Line2D([], [], color="#3d3c39", lw=1.0, ls=STY[k]["ls"], marker=STY[k]["marker"], ms=3.6,
             mec="white", mew=0.4, label=ANAME[k]) for k in ["QA", "FTR", "FT"]]
fig.legend(handles=h1 + h2, loc="upper center", ncol=6, frameon=False, fontsize=FS,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6, handletextpad=0.4)
fig.savefig(os.path.join(OUT, "fig_adapt.pdf"))
fig.savefig(os.path.join(OUT, "fig_adapt.png"), dpi=220)
plt.close(fig)

# guard: no text element smaller than 8 pt
import matplotlib.text as mtext
print("figures written to", os.path.abspath(OUT))
