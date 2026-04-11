import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    import sys
    import pathlib
    from collections import defaultdict

    # Make the project root importable
    project_root = pathlib.Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Real library functions — this notebook tests the actual implementation
    from bookdb.vector_db.reranking import reciprocal_rank_fusion, hybrid_fusion

    return (
        defaultdict,
        hybrid_fusion,
        mo,
        mpatches,
        np,
        plt,
        reciprocal_rank_fusion,
    )


@app.cell
def _(mo):
    mo.md("""
    # Reranking: RRF vs Hybrid Fusion

    Comparison of the two fusion strategies used in `/discovery/recommendations`.

    Before: Reciprocal Rank Fusion (RRF) -> treats both BPR and vector as rank-only lists.
    After: Hybrid Fusion -> uses BPR's actual prediction scores, RRF only for the vector side.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Simulated data
    """)
    return


@app.cell
def _(np):
    # Simulate a realistic BPR output for one user.
    # Items 1-10: model is confident (tight cluster around 0.90-0.98)
    # Items 11-20: model is less sure (0.60-0.75)
    # Items 21-30: model barely recommends them (0.20-0.40)
    # This "cliff" between the groups is exactly what RRF ignores.

    np.random.seed(42)
    bpr_ids    = list(range(1, 31))
    bpr_scores = (
        list(np.linspace(0.98, 0.90, 10)) + # confident zone
        list(np.linspace(0.75, 0.60, 10)) + # uncertain zone
        list(np.linspace(0.40, 0.20, 10))  # weak zone
    )
    bpr_scored = list(zip(bpr_ids, bpr_scores))

    # Vector search returns 20 books via round-robin cluster merging.
    # Some overlap with BPR (items 5, 15, 25), rest are unique vector picks.
    vector_ids = [101, 5, 102, 103, 15, 104, 105, 25, 106, 107,
                  108, 109, 110, 111, 112, 113, 114, 115, 116, 117]

    print("BPR top-5:", bpr_scored[:5])
    print("BPR bottom-5:", bpr_scored[-5:])
    print("Vector top-5:", vector_ids[:5])
    return bpr_ids, bpr_scored, bpr_scores, vector_ids


@app.cell
def _(mo):
    mo.md("""
    ## The cliff problem
    """)
    return


@app.cell
def _(bpr_ids, bpr_scores, mpatches, plt):
    fig_cliff, ax = plt.subplots(figsize=(12, 4))

    colors = (
        ["#2563eb"] * 10 + # confident
        ["#f59e0b"] * 10 +# uncertain
        ["#ef4444"] * 10  # weak
    )
    ax.bar(bpr_ids, bpr_scores, color=colors, width=0.7)
    ax.axhline(0.75, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.40, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Item rank (BPR order)")
    ax.set_ylabel("BPR prediction score")
    ax.set_title("BPR prediction scores")
    ax.set_ylim(0, 1.05)

    legend_patches = [
        mpatches.Patch(color="#2563eb", label="Confident zone (0.90-0.98)"),
        mpatches.Patch(color="#f59e0b", label="Uncertain zone (0.60-0.75)"),
        mpatches.Patch(color="#ef4444", label="Weak zone (0.20-0.40)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    RRF sees 30 ranks and assigns `1/(60+rank+1)` to each. The jump from rank 10 to rank 11 (score drops from 0.90 to 0.75) looks the same as rank 11 to rank 12 (0.75 to 0.73).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Implementing both approaches
    """)
    return


@app.cell
def _(defaultdict, hybrid_fusion, reciprocal_rank_fusion):
    # The ranking always comes from the library; the score dict is computed
    # here only so the visualisation cells can show numeric values.

    def rrf_old(bpr_ids, vector_ids, k=60):
        """Old approach: wraps reciprocal_rank_fusion with both sources as rank-only."""
        ranking = reciprocal_rank_fusion([bpr_ids, vector_ids], k=k)
        scores = defaultdict(float)
        for rank, item_id in enumerate(bpr_ids):
            scores[item_id] += 1.0 / (k + rank + 1)
        for rank, item_id in enumerate(vector_ids):
            scores[item_id] += 1.0 / (k + rank + 1)
        return ranking, dict(scores)

    def hybrid_fusion_scored(bpr_scored, vector_ids, bpr_weight=0.6, k=60):
        """New approach: wraps hybrid_fusion and also returns combined scores for display."""
        ranking = hybrid_fusion(bpr_scored, vector_ids, bpr_weight=bpr_weight, rrf_k=k)
        # Recompute combined scores for display (same math as bookdb.vector_db.reranking)
        combined = defaultdict(float)
        vector_weight = 1.0 - bpr_weight
        lo_b = min(s for _, s in bpr_scored)
        hi_b = max(s for _, s in bpr_scored)
        span_b = hi_b - lo_b if hi_b > lo_b else 1.0
        for item_id, score in bpr_scored:
            combined[item_id] += bpr_weight * (score - lo_b) / span_b
        rrf_raw = {item_id: 1.0 / (k + rank + 1) for rank, item_id in enumerate(vector_ids)}
        lo_v = min(rrf_raw.values())
        hi_v = max(rrf_raw.values())
        span_v = hi_v - lo_v if hi_v > lo_v else 1.0
        for item_id, rrf_score in rrf_raw.items():
            combined[item_id] += vector_weight * (rrf_score - lo_v) / span_v
        return ranking, dict(combined)

    return hybrid_fusion_scored, rrf_old


@app.cell
def _(bpr_ids, bpr_scored, hybrid_fusion_scored, rrf_old, vector_ids):
    old_ranking, old_scores = rrf_old(bpr_ids, vector_ids)
    new_ranking, new_scores = hybrid_fusion_scored(bpr_scored, vector_ids, bpr_weight=0.6)

    print("Old top-15:", old_ranking[:15])
    print("New top-15:", new_ranking[:15])
    return new_ranking, new_scores, old_ranking, old_scores


@app.cell
def _(mo):
    mo.md("""
    ## Side-by-side ranking comparison (top 20)
    """)
    return


@app.cell
def _(new_ranking, new_scores, old_ranking, old_scores):
    def label(item_id):
        if item_id <= 30:
            return f"BPR-{item_id}"
        return f"vec-{item_id}"

    def score_zone(item_id, scores_dict):
        s = scores_dict.get(item_id, 0)
        return f"{s:.4f}"

    rows = []
    for pos in range(20):
        old_item = old_ranking[pos] if pos < len(old_ranking) else "-"
        new_item = new_ranking[pos] if pos < len(new_ranking) else "-"
        rows.append({
            "Rank": pos + 1,
            "RRF (old)": label(old_item),
            "RRF score": score_zone(old_item, old_scores),
            "Hybrid (new)": label(new_item),
            "Hybrid score": score_zone(new_item, new_scores),
            "Same?": "=" if old_item == new_item else "/",
        })
    return label, rows


@app.cell
def _(mo, rows):
    mo.ui.table(rows)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Where the cliff items land
    """)
    return


@app.cell
def _(new_ranking, np, old_ranking, plt):
    cliff_items = list(range(21, 31))  # weak zone BPR items
    vector_only = [101, 102, 103, 104, 105, 106, 107]

    def rank_of(item, ranking):
        try:
            return ranking.index(item) + 1
        except ValueError:
            return len(ranking) + 1

    old_cliff_ranks  = [rank_of(i, old_ranking)  for i in cliff_items]
    new_cliff_ranks  = [rank_of(i, new_ranking)  for i in cliff_items]
    old_vector_ranks = [rank_of(i, old_ranking)  for i in vector_only]
    new_vector_ranks = [rank_of(i, new_ranking)  for i in vector_only]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    x = np.arange(len(cliff_items))
    ax1.bar(x - 0.2, old_cliff_ranks,  0.4, label="RRF (old)",    color="#ef4444", alpha=0.8)
    ax1.bar(x + 0.2, new_cliff_ranks,  0.4, label="Hybrid (new)", color="#2563eb", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"BPR-{i}" for i in cliff_items], rotation=30)
    ax1.set_ylabel("Position in final ranking (lower = better)")
    ax1.set_title("Weak-zone BPR items: where do they end up?")
    ax1.legend()
    ax1.invert_yaxis()

    x2 = np.arange(len(vector_only))
    ax2.bar(x2 - 0.2, old_vector_ranks,  0.4, label="RRF (old)",    color="#ef4444", alpha=0.8)
    ax2.bar(x2 + 0.2, new_vector_ranks,  0.4, label="Hybrid (new)", color="#2563eb", alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"vec-{i}" for i in vector_only], rotation=30)
    ax2.set_ylabel("Position in final ranking (lower = better)")
    ax2.set_title("Vector-only items: do they rise when BPR is weak?")
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## How does bpr_weight shift the ranking?
    """)
    return


@app.cell
def _(mo):
    weight_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6,
        label="bpr_weight (0 = full vector, 1 = full BPR)",
    )
    return (weight_slider,)


@app.cell
def _(weight_slider):
    weight_slider


@app.cell
def _(
    bpr_scored,
    hybrid_fusion_scored,
    label,
    mpatches,
    plt,
    vector_ids,
    weight_slider,
):
    w = weight_slider.value
    ranked, scores_w = hybrid_fusion_scored(bpr_scored, vector_ids, bpr_weight=w)
    top15 = ranked[:15]

    colors_w = []
    for item in top15:
        if item in [5, 15, 25]:
            colors_w.append("#8b5cf6")   # overlap (both sources)
        elif item <= 30:
            if item <= 10:
                colors_w.append("#2563eb")   # confident BPR
            elif item <= 20:
                colors_w.append("#f59e0b")   # uncertain BPR
            else:
                colors_w.append("#ef4444")   # weak BPR
        else:
            colors_w.append("#10b981")       # vector-only

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.barh([label(i) for i in reversed(top15)],
             [scores_w[i] for i in reversed(top15)],
             color=list(reversed(colors_w)))
    ax3.set_xlabel("Combined score")
    ax3.set_title(f"Top-15 ranking at bpr_weight={w:.2f} / vector_weight={1-w:.2f}")

    legend_w = [
        mpatches.Patch(color="#2563eb", label="BPR confident (rank 1-10)"),
        mpatches.Patch(color="#f59e0b", label="BPR uncertain (rank 11-20)"),
        mpatches.Patch(color="#ef4444", label="BPR weak (rank 21-30)"),
        mpatches.Patch(color="#10b981", label="Vector-only"),
        mpatches.Patch(color="#8b5cf6", label="In both sources"),
    ]
    ax3.legend(handles=legend_w, loc="lower right")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
