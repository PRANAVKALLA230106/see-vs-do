"""Paper figures from the result zips (unzipped under /tmp/claude-0/rep/*). Recomputes pooled medians and bootstrap CIs
from per-scenario files so every panel uses the same rules as the notebooks."""
import glob, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

import os; HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "../results")
OUT = os.path.join(HERE, "../figures"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, RED, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984", "#0b0b0b", "#52514e"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
rng = np.random.default_rng(0)


def boot(x, n=4000):
    x = pd.Series(x).dropna().values
    if len(x) < 3:
        return np.nan, np.nan
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def passes(means, mode, kind="true"):
    """Fixed pass rule on pooled scenario means: {layer: bool}."""
    out = {}
    for L, g in means[means["mode"] == mode].groupby("layer"):
        t = g[g.kind == kind].set_index("scenario_id"); u = g[g.kind == "unrelated"].set_index("scenario_id")
        f = g[g.kind == "false"].set_index("scenario_id")
        d = (t.dmedC - u.dmedC).dropna()
        out[L] = bool(t.dmedC.mean() >= 1 and d.mean() > .5 and (d > 0).mean() >= .7 and abs(t.dfill.mean()) < .5 and f.dmedC.mean() < 0)
    return out


def summarise(scores, means, mode="word", kind="true", min_n=5):
    p = passes(means, mode, kind)
    rows = []
    for L, g in scores[(scores["mode"] == mode) & (scores.kind == kind)].groupby("layer"):
        lo, hi = boot(g.D)
        mm = means[(means["mode"] == mode) & (means.kind == kind) & (means.layer == L)]
        rows.append(dict(layer=L, D=g.D.median(), lo=lo, hi=hi, n=int(g.D.notna().sum()), passes=p.get(L, False),
                         dmed=mm.dmedC.mean(), dcause=mm.dcauseC.mean(), deffect=mm.deffectC.mean()))
    s = pd.DataFrame(rows)
    s["show"] = s.passes & (s.n >= min_n)
    return s


def load(pattern):
    fs = sorted(glob.glob(pattern))
    assert fs, pattern
    d = pd.concat(pd.read_csv(f) for f in fs)
    if "kind" in d:
        d["kind"] = d["kind"].astype(str).str.lower()      # pandas reads a column of "true" as booleans
    return d


# ---------------- data
MODELS = {
    "Qwen3-4B": dict(n_layers=36, color=BLUE, scores=load(f"{R}/ds_qwen/DS_scenario_scores.csv"), means=load(f"{R}/ds_qwen/DS_scenario_means.csv"),
                     note="30 scen."),
    "Llama-3.2-3B": dict(n_layers=28, color=ORANGE, scores=load(f"{R}/llama/M_*_scenario_scores.csv"),
                         means=load(f"{R}/llama/M_*_scenario_means.csv"), note="23 scen."),
    "Gemma-2-2B": dict(n_layers=26, color=AQUA, scores=load(f"{R}/gemma/M_*_scenario_scores.csv"),
                       means=load(f"{R}/gemma/M_*_scenario_means.csv"), note="12 scen."),
}
for m, d in MODELS.items():
    if "kind" not in d["scores"]:
        d["scores"]["kind"] = "true"
    d["word"] = summarise(d["scores"], d["means"], "word")
    d["note"] = f"{d['scores'].scenario_id.nunique()} scen."
    d["word"].to_csv(f"{OUT}/data_models_{m}.csv", index=False)

# ---------------- Figure: three models
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.3))
ax = axes[0]
for m, d in MODELS.items():
    s = d["word"]; x = s.layer / d["n_layers"]
    ax.plot(x, s.dmed, "-o", color=d["color"], ms=3, label=f"{m} ({d['note']})")
    sh = s[s.passes]
    ax.plot(sh.layer / d["n_layers"], sh.dmed, "o", color=d["color"], ms=5, mfc="white", mew=1.4)
ax.axhline(0, color="#b5b4ae", lw=.8)
ax.axhline(1, color="#b5b4ae", lw=.8, ls=":")
ax.text(1.0, 1.05, "pass threshold", color=MUTED, ha="right", va="bottom", fontsize=7)
ax.set_xlabel("patched layer (fraction of depth)"); ax.set_ylabel("Δ belief in patched fact (log-odds)")
ax.plot([], [], "o", color=MUTED, mfc="white", mew=1.4, ms=5, label="open marker: patch passes the rule")
ax.set_title("a  Patch strength by depth", loc="left")
ax.legend(fontsize=6.8, loc="upper right")
ax = axes[1]
for i, (m, d) in enumerate(MODELS.items()):
    s = d["word"][d["word"].show]; x = s.layer / d["n_layers"] + (i - 1) * 0.006
    ax.errorbar(x, s.D, yerr=[s.D - s.lo, s.hi - s.D], fmt="-o", color=d["color"], ms=3, capsize=2, lw=1.4, label=m)
ax.axhline(0, color="#b5b4ae", lw=.8); ax.axhline(1, color="#b5b4ae", lw=.8)
ax.set_ylim(-0.6, 1.15); ax.set_xlim(-0.02, 0.52)
ax.text(0.0, -0.06, "seeing (D = 0)", color=MUTED, ha="left", va="top", fontsize=7)
ax.text(0.51, 1.02, "doing (D = 1)", color=MUTED, ha="right", va="bottom", fontsize=7)
ax.set_xlabel("patched layer (fraction of depth)"); ax.set_ylabel("do-likeness D (median, 95% CI)")
ax.set_title("b  Do-likeness where patches work", loc="left")
fig.savefig(f"{OUT}/fig_models.png"); fig.savefig(f"{OUT}/fig_models.pdf"); plt.close(fig)

# ---------------- Figure: do-source patch (chains + forks, Qwen + Llama)
PANELS = [("Qwen3-4B · chains", f"{R}/ds_qwen/DS", 36), ("Qwen3-4B · forks", f"{R}/fork_qwen/FK", 36),
          ("Llama-3.2-3B · chains", f"{R}/ds_llama/DS_llama", 28), ("Llama-3.2-3B · forks", f"{R}/fork_llama/FK_llama", 28)]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.5))
axes = axes.ravel()
rows = []
for ax, (title, pre, nl) in zip(axes, PANELS):
    sc, me = load(f"{pre}_scenario_scores.csv"), load(f"{pre}_scenario_means.csv")
    st = summarise(sc, me, "word", "true"); sd = summarise(sc, me, "word", "do")
    last = st[st.passes].layer.max()
    ax.axvspan(-1, last + 0.5, color="#f1f0ec", zorder=0)
    for s, c, lab, off in [(st, BLUE, "fact copied from 'see' story", -0.15), (sd, RED, "fact copied from 'do' story", 0.15)]:
        s = s[s.show]
        ax.errorbar(s.layer + off, s.D, yerr=[s.D - s.lo, s.hi - s.D], fmt="-o", color=c, ms=3, capsize=2, lw=1.4, label=lab)
    # paired difference
    w = sc[sc["mode"] == "word"].pivot_table(index=["scenario_id", "layer"], columns="kind", values="D").reset_index()
    for L, g in w.groupby("layer"):
        d = (g["do"] - g["true"]).dropna(); lo, hi = boot(d)
        rows.append(dict(panel=title, layer=L, diff=d.median(), lo=lo, hi=hi, n=len(d)))
        if lo > 0 and d.median() >= 0.05 and L <= last:
            ax.text(L + 0.15, 1.08, "*", ha="center", va="top", fontsize=11, color=RED)
    ax.axhline(0, color="#b5b4ae", lw=.8); ax.axhline(1, color="#b5b4ae", lw=.8)
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel("patched layer")
    ax.set_xlim(-1, last + 1.5); ax.set_ylim(-0.5, 1.15)
axes[0].set_ylabel("do-likeness D (median, 95% CI)"); axes[2].set_ylabel("do-likeness D (median, 95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
fig.savefig(f"{OUT}/fig_dosource.png"); fig.savefig(f"{OUT}/fig_dosource.pdf"); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/data_dosource_paired_diff.csv", index=False)

# ---------------- Figure: answer-format robustness (Qwen, Yes/No vs True/False, same items)
tf_s, tf_m = load(f"{R}/tf_qwen/TF_scenario_scores.csv"), load(f"{R}/tf_qwen/TF_scenario_means.csv")
yn = MODELS["Qwen3-4B"]["word"]; tf = summarise(tf_s, tf_m, "word")
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for s, c, lab, off in [(yn[yn.layer.isin(tf.layer)], BLUE, "Yes / No answers", -0.2), (tf, ORANGE, "True / False answers", 0.2)]:
    s = s[s.show]
    ax.errorbar(s.layer + off, s.D, yerr=[s.D - s.lo, s.hi - s.D], fmt="-o", color=c, ms=3, capsize=2, lw=1.4, label=lab)
ax.axhline(0, color="#b5b4ae", lw=.8); ax.axhline(1, color="#b5b4ae", lw=.8)
ax.set_ylim(-0.3, 1.1); ax.set_xlabel("patched layer"); ax.set_ylabel("do-likeness D")
ax.set_title("Answer format does not matter (Qwen3-4B)", loc="left"); ax.legend(fontsize=7, loc="center right")
fig.savefig(f"{OUT}/fig_tf.png"); fig.savefig(f"{OUT}/fig_tf.pdf"); plt.close(fig)
tf.to_csv(f"{OUT}/data_tf.csv", index=False)
print("done")
