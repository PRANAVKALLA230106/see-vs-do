import pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import os; HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "../results"); OUT = os.path.join(HERE, "../figures"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, RED, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#b5b4ae"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight"})
M = [("Qwen3-4B", f"{R}/ds2_qwen/DS2_qwen_summary.csv", [14, 15, 16, 17]), ("Llama-3.2-3B", f"{R}/ds2_llama/DS2_llama_summary.csv", [7, 8, 9])]
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), gridspec_kw=dict(hspace=0.55, wspace=0.28))
for row, (name, f, late) in enumerate(M):
    s = pd.read_csv(f)
    ax = axes[row, 0]
    ax.axvspan(min(late) - 0.5, max(late) + 0.5, color="#f1f0ec", zorder=0)
    for mode, c, off in [("single", RED, -0.12), ("column", ORANGE, 0.12)]:
        x = s[(s.test == "forward") & (s["mode"] == mode)]
        ax.errorbar(x.layer + off, x["diff"], yerr=[x["diff"] - x.diff_lo, x.diff_hi - x["diff"]], fmt="-o", ms=3, capsize=2, color=c,
                    lw=1.4, label=f"{mode}-layer patch" if mode == "single" else "column patch (layers 0..L)")
    ax.axhline(0, color=LINE, lw=.8); ax.set_ylim(-0.15, 0.75)
    ax.set_title(f"{name} · forward: do − see source", loc="left", fontsize=8)
    ax.set_xlabel("patched layer"); ax.set_ylabel("paired difference in D")
    ax = axes[row, 1]
    ax.axvspan(min(late) - 0.5, max(late) + 0.5, color="#f1f0ec", zorder=0)
    for mode, c, off in [("single", BLUE, -0.12), ("column", "#1baf7a", 0.12)]:
        x = s[(s.test == "reverse") & (s["mode"] == mode)]
        ax.errorbar(x.layer + off, x.D_rev, yerr=[x.D_rev - x.rev_lo, x.rev_hi - x.D_rev], fmt="-o", ms=3, capsize=2, color=c, lw=1.4,
                    label=f"{mode}-layer" if mode == "single" else "column")
    ax.axhline(1, color=LINE, lw=.8); ax.axhline(0, color=LINE, lw=.8, ls=":"); ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"{name} · reverse: see-word in do story", loc="left", fontsize=8)
    ax.set_xlabel("patched layer"); ax.set_ylabel("D of the patched do story")
axes[0, 0].legend(fontsize=6.8, loc="upper left"); axes[0, 1].legend(fontsize=6.8, loc="lower left")
fig.savefig(f"{OUT}/fig_ds2.png"); fig.savefig(f"{OUT}/fig_ds2.pdf"); plt.close(fig); print("ok")
