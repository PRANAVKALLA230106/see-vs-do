import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import os; HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "../results/main_qwen"); OUT = os.path.join(HERE, "../figures"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, RED, GREY, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984", "#0b0b0b", "#52514e", "#b5b4ae"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight"})

# ---- steering artifact (run 1b, layer 16, after-fact positions, alpha 4, raw shifts)
x = pd.read_csv(f"{R}/X_sweep_layer16.csv")
x = x[(x.alpha == 4) & (x.position == "after") & x.kind.isin(["raw", "raw_unrelated"])]
x["dfill"] = x[["df0", "df1", "df2"]].mean(axis=1)
m = x.groupby("kind")[["dcause", "deffect", "dmed", "dfill"]].mean()
labels = ["cause", "effect", "the fact", "unrelated\nfillers"]
fig, ax = plt.subplots(figsize=(3.6, 2.4))
w = 0.36; idx = np.arange(4)
for i, (k, c, lab) in enumerate([("raw", BLUE, "direction of this scenario's fact"), ("raw_unrelated", GREY, "direction from an unrelated scenario")]):
    v = m.loc[k, ["dcause", "deffect", "dmed", "dfill"]].values
    ax.bar(idx + (i - .5) * w, v, w * 0.92, color=c, label=lab)
    for j, val in enumerate(v):
        ax.text(idx[j] + (i - .5) * w, val + 0.05, f"{val:.1f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(idx, labels); ax.set_ylabel("raw shift in log-odds of \"Yes\"")
ax.set_ylim(0, 3.7); ax.axhline(0, color=LINE, lw=.8)
ax.set_title("Steering moves every answer towards Yes", loc="left"); ax.legend(fontsize=6.8, loc="upper left")
fig.savefig(f"{OUT}/fig_steering.png"); fig.savefig(f"{OUT}/fig_steering.pdf"); plt.close(fig)

# ---- main Qwen: exploration vs confirmation, word + sentence
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, s, title in zip(axes, ["exploration", "confirmation"], ["a  Exploration set (15 of 20 chains)", "b  Confirmation set (15 of 25 new chains)"]):
    d = pd.read_csv(f"{R}/P_{s}_D_summary.csv")
    for mode, c, off in [("word", BLUE, -0.25), ("sentence", ORANGE, 0.25)]:
        q = d[(d["mode"] == mode) & d.passes & (d.n_scored >= 5)]
        ax.errorbar(q.layer + off, q.D_median, yerr=[q.D_median - q.ci_lo, q.ci_hi - q.D_median], fmt="-o", ms=3, capsize=2, color=c, lw=1.4,
                    label=f"{mode} patch")
    ax.axvspan(16.5, 34, color="#f1f0ec", zorder=0)
    ax.text(25, 0.6, "patches fail\nthe pass rule", ha="center", color=MUTED, fontsize=7)
    ax.axhline(0, color=LINE, lw=.8); ax.axhline(1, color=LINE, lw=.8)
    ax.set_xlim(-1, 34); ax.set_ylim(-0.3, 1.1); ax.set_title(title, loc="left"); ax.set_xlabel("patched layer (of 36)")
axes[0].set_ylabel("do-likeness D (median, 95% CI)"); axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.85))

fig.savefig(f"{OUT}/fig_main.png"); fig.savefig(f"{OUT}/fig_main.pdf"); plt.close(fig)

# ---- colliders
h = pd.read_csv(f"{R}/H3_summary.csv"); h = h[h.passes]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), gridspec_kw=dict(wspace=0.35))
ax = axes[0]
ax.errorbar(h.layer, h.D_median, yerr=[h.D_median - h.D_lo, h.D_hi - h.D_median], fmt="-o", ms=3, capsize=2, color=BLUE)
ax.axhline(0, color=LINE, lw=.8); ax.axhline(1, color=LINE, lw=.8); ax.set_ylim(-0.3, 1.1)
ax.set_title("a  Do-likeness of a patched C", loc="left"); ax.set_xlabel("patched layer"); ax.set_ylabel("D (median, 95% CI)")
ax = axes[1]
ax.errorbar(h.layer, h.EA_patch_median, yerr=[h.EA_patch_median - h.EA_lo, h.EA_hi - h.EA_patch_median], fmt="-o", ms=3, capsize=2, color=BLUE, label="C patched")
ax.axhline(h.EA_text_median.iloc[0], color=ORANGE, ls="--", lw=1.4, label="C read in the text")
ax.axhline(0, color=LINE, lw=.8, ls=":"); ax.text(15.3, 0.05, "intervention: 0", ha="right", fontsize=6.8, color=MUTED)
ax.set_ylim(-0.1, 2.0); ax.set_title("b  Explaining away after C", loc="left"); ax.set_xlabel("patched layer")
ax.set_ylabel("interaction (log-odds)"); ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.06))
fig.savefig(f"{OUT}/fig_colliders.png"); fig.savefig(f"{OUT}/fig_colliders.pdf"); plt.close(fig)
print("ok")
