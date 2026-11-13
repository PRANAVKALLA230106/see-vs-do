import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BLUE, RED, ORANGE, INK, MUTED, LINE = "#2a78d6", "#e34948", "#eb6834", "#0b0b0b", "#52514e", "#b5b4ae"
import os; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
cols = [
    ("SEE  (read the fact)", "…The road was icy.", None, "↑", "↑", BLUE, "evidence: model infers the cause"),
    ("DO  (fact imposed)", "…A pipe burst and the water\nfroze, so the road was icy.", None, "–", "↑", RED, "intervention: cause unchanged"),
    ("PATCH  (edit the state)", "…The road was long.", "copy state of 'icy'\nat one layer", "?", "↑", ORANGE, "which pattern does the model follow?"),
]
for ax, (title, story, patch, dc, de, c, foot) in zip(axes, cols):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.2, 9.6, title, fontsize=8.5, fontweight="bold", color=c, va="top")
    ax.add_patch(FancyBboxPatch((0.2, 6.7), 9.6, 2.1, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#f6f5f2", ec=LINE, lw=0.8))
    ax.text(0.5, 7.75, story, fontsize=6.9, va="center", color=INK, style="italic")
    if patch:
        ax.text(0.4, 5.55, "copy the hidden state\nof 'icy' from the SEE\nrun, at one layer", fontsize=6.4, va="center", color=ORANGE)
        ax.add_patch(FancyArrowPatch((7.9, 5.0), (7.1, 7.2), arrowstyle="-|>", mutation_scale=8, color=ORANGE, lw=1.1))
    # mini causal graph: snow -> icy -> bus late
    xs = {"snow": 1.45, "icy": 5.0, "late": 8.55}
    for k, x in xs.items():
        ax.add_patch(FancyBboxPatch((x - 1.35, 2.6), 2.7, 1.2, boxstyle="round,pad=0.05,rounding_size=0.25",
                                    fc="white", ec=(c if k == "icy" else LINE), lw=1.2 if k == "icy" else 0.8))
        ax.text(x, 3.2, {"snow": "snowed?", "icy": "road icy", "late": "bus late?"}[k], ha="center", va="center", fontsize=6.4)
    for a, b in [("snow", "icy"), ("icy", "late")]:
        ax.add_patch(FancyArrowPatch((xs[a] + 1.4, 3.2), (xs[b] - 1.4, 3.2), arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.9))
    ax.text(xs["snow"], 1.75, dc, ha="center", fontsize=12, color=c, fontweight="bold")
    ax.text(xs["late"], 1.75, de, ha="center", fontsize=12, color=c, fontweight="bold")
    ax.text(5.0, 0.4, foot, ha="center", fontsize=6.8, color=MUTED)
fig.savefig(OUT + "/fig_schematic.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + "/fig_schematic.pdf", bbox_inches="tight")
