"""Regenerate Figures 1-5 of the final manuscript from saved analysis outputs.

Usage (from the repository root):
    python make_paper_figures.py --analysis analysis/main_results --out analysis/paper_figures

Inputs are read only from the frozen analysis folder; no score is recomputed or changed.
Outputs: figN_*.png (600 dpi) and figN_*.pdf (vector) plus figure_sources.json with the
exact values drawn in each figure.
"""
import argparse, json, hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

STRATS = ["fixed", "overlap", "semantic"]
LABEL = {"fixed": "Fixed-size", "overlap": "Overlap", "semantic": "Semantic"}
MARK = {"fixed": "o", "overlap": "s", "semantic": "^"}
SHADE = {"fixed": "#9a9a9a", "overlap": "#5a5a5a", "semantic": "#000000"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "savefig.dpi": 600, "pdf.fonttype": 42})

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(fig, out, name, sources, values):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sources[name] = values

def fig_pipeline(out, sources):
    fig, ax = plt.subplots(figsize=(6.8, 2.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    W = 18.2
    def box(x, title, lines, bold=False):
        ax.add_patch(FancyBboxPatch((x, 5.6), W, 22.4, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc="#efefef" if bold else "white", ec="black", lw=1.4 if bold else 0.8))
        ax.text(x + W / 2, 25.0, title, ha="center", va="center", fontsize=7.2, weight="bold")
        for i, l in enumerate(lines):
            ax.text(x + W / 2, 20.9 - 3.25 * i, l, ha="center", va="center", fontsize=6.2)
    xs = [0.6, 20.8, 41.0, 61.2, 81.4]
    box(xs[0], "Data", ["SQuAD 1.1 dev", "20 articles, seed 42", "2,000 retrieval Qs", "1,000 generation Qs"])
    box(xs[1], "Chunking", ["Fixed: 100 words", "Overlap: 100 words", "+ 20-word overlap", "Semantic: sentences,", "break at 90th pct"], bold=True)
    box(xs[2], "Retrieval", ["MPNet embeddings", "cosine similarity", "over all fragments", "top 5 saved"])
    box(xs[3], "Generation", ["FLAN-T5-Large", "top 3 fragments", "512-token prompt", "greedy decoding"])
    box(xs[4], "Scoring", ["Hit@1/3/5", "by answer offsets", "Exact Match", "token F1"])
    for a in xs[:-1]:
        ax.annotate("", xy=(a + W + 2.0, 17), xytext=(a + W + 0.2, 17), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    ax.text(50, 2.2, "Questions, model revisions, prompt and decoding are identical in all conditions; only the chunking step (shaded) differs.",
            ha="center", fontsize=7, style="italic")
    save(fig, out, "fig1_pipeline", sources, {"type": "diagram", "values": "configuration from experiment_config.json"})

def fig_retrieval(a, out, sources):
    t = pd.read_csv(a / "retrieval_table.csv").set_index("strategy")
    fig, ax = plt.subplots(figsize=(6.6, 2.6)); vals = {}
    ks = ["hit_at_1", "hit_at_3", "hit_at_5"]; names = ["Hit@1", "Hit@3 (primary)", "Hit@5"]
    for gi, k in enumerate(ks):
        for si, s in enumerate(STRATS):
            y = gi * 4 + (2 - si)
            m, lo, hi = (t.loc[s, k] * 100, t.loc[s, k + "_ci_low"] * 100, t.loc[s, k + "_ci_high"] * 100)
            ax.plot([lo, hi], [y, y], color=SHADE[s], lw=1.4)
            ax.plot(m, y, MARK[s], color=SHADE[s], ms=6, mfc="white" if s == "fixed" else SHADE[s], mec=SHADE[s], mew=1.2,
                    label=LABEL[s] if gi == 0 else None)
            ax.text(hi + 0.6, y, f"{m:.2f}", va="center", fontsize=7.6)
            vals[f"{s}_{k}"] = [round(m, 2), round(lo, 2), round(hi, 2)]
        ax.text(48.2, gi * 4 + 1, names[gi], ha="right", va="center", fontsize=8.6, weight="bold" if k == "hit_at_3" else "normal")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(48.5, 95)
    ax.set_xlabel("Questions with a complete annotated answer span in the top k fragments (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.6, 1.16), fontsize=8)
    ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    save(fig, out, "fig2_retrieval_hitk", sources, {"source": "retrieval_table.csv", "values_pct_mean_lo_hi": vals})

def fig_generation(a, out, sources):
    t = pd.read_csv(a / "generation_table.csv").set_index("strategy")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.9), sharey=True); vals = {}
    for ax, (k, title) in zip(axes, [("exact_match", "Exact Match (%)"), ("f1", "Token F1 (%)")]):
        for si, s in enumerate(STRATS):
            y = 2 - si; m, lo, hi = t.loc[s, k] * 100, t.loc[s, k + "_ci_low"] * 100, t.loc[s, k + "_ci_high"] * 100
            ax.plot([lo, hi], [y, y], color=SHADE[s], lw=1.4)
            ax.plot(m, y, MARK[s], color=SHADE[s], ms=6, mfc="white" if s == "fixed" else SHADE[s], mec=SHADE[s], mew=1.2)
            ax.text(hi + 0.5, y, f"{m:.2f}", va="center", fontsize=7.6)
            vals[f"{s}_{k}"] = [round(m, 2), round(lo, 2), round(hi, 2)]
        ax.set_title(title, fontsize=9); ax.set_xlim(60, 90); ax.set_ylim(-0.5, 2.5); ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_yticks([2, 1, 0]); axes[0].set_yticklabels([LABEL[s] for s in STRATS])
    axes[1].tick_params(left=False)
    save(fig, out, "fig3_generation_em_f1", sources, {"source": "generation_table.csv", "values_pct_mean_lo_hi": vals})

def fig_accounting(a, out, sources):
    t = pd.read_csv(a / "retrieval_table.csv").set_index("strategy")
    eff = json.loads((a / "analysis.json").read_text())["efficiency"]
    fig, ax = plt.subplots(figsize=(6.6, 1.75)); vals = {}
    for si, s in enumerate(STRATS):
        y = 2 - si; hit = int(t.loc[s, "hit_at_3_sum"]); rank = int(eff[s]["ranking_miss_with_span_available_count"])
        miss = int(eff[s]["corpus_span_missing_count"]); assert hit + rank + miss == 2000
        ax.barh(y, hit, color="#3a3a3a", height=0.6); ax.barh(y, rank, left=hit, color="#bdbdbd", height=0.6, hatch="////", edgecolor="white", lw=0)
        ax.barh(y, miss, left=hit + rank, color="black", height=0.6)
        ax.text((1300 + hit) / 2, y, f"top-3 hit {hit}", ha="center", va="center", color="white", fontsize=7.8)
        ax.text(hit + rank / 2, y, f"ranking miss {rank}", ha="center", va="center", fontsize=7.8)
        ax.text(2008, y, f"index miss {miss}", va="center", fontsize=7.8)
        vals[s] = {"top3_hit": hit, "ranking_miss": rank, "index_miss": miss}
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([LABEL[s] for s in STRATS]); ax.set_xlim(1300, 2000)
    ax.set_xlabel("Retrieval questions (n = 2,000 per strategy; axis starts at 1,300)")
    save(fig, out, "fig4_evidence_accounting", sources, {"source": "retrieval_table.csv + analysis.json:efficiency", "counts": vals})

def fig_articles(a, out, sources):
    r = pd.read_csv(a / "retrieval_article_metrics.csv").pivot(index="article", columns="strategy", values="hit_at_3") * 100
    r["d_sem"] = r.semantic - r.fixed; r["d_ovl"] = r.overlap - r.fixed; r = r.sort_values("d_sem")
    fig, ax = plt.subplots(figsize=(6.6, 4.3)); ys = list(range(len(r)))
    ax.axvline(0, color="black", lw=0.8)
    ax.plot(r.d_ovl, ys, "s", ms=5, mfc="white", mec="#5a5a5a", mew=1.1, label="Overlap − fixed")
    ax.plot(r.d_sem, ys, "^", ms=5.5, color="black", label="Semantic − fixed")
    ax.set_yticks(list(ys)); ax.set_yticklabels([x.replace("_", " ") for x in r.index], fontsize=7.6)
    ax.set_xlabel("Article-level Hit@3 difference vs. fixed-size (percentage points; 100 questions per article)")
    ax.grid(axis="x", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, out, "fig5_article_differences", sources,
         {"source": "retrieval_article_metrics.csv", "semantic_minus_fixed_pp": r.d_sem.round(2).to_dict(),
          "overlap_minus_fixed_pp": r.d_ovl.round(2).to_dict(),
          "semantic_better_articles": int((r.d_sem > 0).sum()), "overlap_better_articles": int((r.d_ovl > 0).sum()),
          "mean_semantic_minus_fixed_pp": round(r.d_sem.mean(), 2), "mean_overlap_minus_fixed_pp": round(r.d_ovl.mean(), 2)})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, default=Path("analysis/main_results"))
    p.add_argument("--out", type=Path, default=Path("analysis/paper_figures"))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); sources = {}
    fig_pipeline(a.out, sources); fig_retrieval(a.analysis, a.out, sources); fig_generation(a.analysis, a.out, sources)
    fig_accounting(a.analysis, a.out, sources); fig_articles(a.analysis, a.out, sources)
    inputs = {f: sha(a.analysis / f) for f in ["retrieval_table.csv", "generation_table.csv", "retrieval_article_metrics.csv", "analysis.json"]}
    json.dump({"input_sha256": inputs, "figures": sources}, open(a.out / "figure_sources.json", "w"), indent=2)
    print("Wrote", sorted(x.name for x in a.out.iterdir()))

if __name__ == "__main__":
    main()
