"""
STEP 6: Visualizations
========================
Generates all figures needed for your report/presentation:

  1. Bar chart: F1-score comparison across strategies per density
  2. ROC-AUC heatmap: models × strategies
  3. Hybrid bucket distribution pie chart
  4. Performance improvement delta chart (hybrid vs baselines)
  5. Gradient Boosting performance across densities (reproduces base paper Fig 5)
  6. Correlation matrices for features (reproduces base paper Fig 3-4)
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})
PALETTE = {"shortest": "#E07B54", "longest": "#5B8DB8", "hybrid": "#2CA02C"}
DENSITY_ORDER = ["sparse", "moderate", "dense"]
STRATEGY_ORDER = ["shortest", "longest", "hybrid"]


# ════════════════════════════════════════════════════════════════════════════════
# FIG 1 — F1-Score comparison bar chart  (grouped by density, coloured by strategy)
# ════════════════════════════════════════════════════════════════════════════════

def plot_f1_comparison(results_df: pd.DataFrame, dataset_name: str, model: str = "GB"):
    """
    For a given model (default Gradient Boosting), plot F1-score
    across densities for all three sampling strategies.
    """
    sub = results_df[results_df["model"] == model].copy()
    sub["density"] = pd.Categorical(sub["density"], categories=DENSITY_ORDER, ordered=True)
    sub = sub.sort_values("density")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DENSITY_ORDER))
    width = 0.25

    for i, strat in enumerate(STRATEGY_ORDER):
        vals = [
            sub[(sub["density"] == d) & (sub["strategy"] == strat)]["f1_score"].values
            for d in DENSITY_ORDER
        ]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=strat.capitalize(), color=PALETTE[strat], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Graph Density")
    ax.set_ylabel("F1-Score (Weighted)")
    ax.set_title(f"F1-Score Comparison by Sampling Strategy\n"
                 f"Model: {model} | Dataset: {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DENSITY_ORDER])
    ax.set_ylim(0, 1.08)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_f1_comparison_{model}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 2 — ROC-AUC heatmap  (all models × strategies for each density)
# ════════════════════════════════════════════════════════════════════════════════

def plot_roc_auc_heatmap(results_df: pd.DataFrame, dataset_name: str,
                          density: str = "dense"):
    sub = results_df[results_df["density"] == density].copy()
    pivot = sub.pivot_table(index="model", columns="strategy", values="roc_auc")
    pivot = pivot.reindex(columns=STRATEGY_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0,
                annot_kws={"size": 9})
    ax.set_title(f"ROC-AUC Heatmap — {density.capitalize()} Graph\n"
                 f"Dataset: {dataset_name}")
    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Model")
    # Highlight hybrid column
    ax.add_patch(mpatches.FancyBboxPatch(
        (STRATEGY_ORDER.index("hybrid"), 0),
        1, len(pivot),
        boxstyle="square,pad=0", fill=False,
        edgecolor="#2CA02C", linewidth=2.5
    ))
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_roc_auc_heatmap_{density}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 3 — Hybrid bucket distribution pie chart
# ════════════════════════════════════════════════════════════════════════════════

def plot_hybrid_bucket_distribution(all_datasets: dict, dataset_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, density in zip(axes, DENSITY_ORDER):
        if density not in all_datasets:
            ax.axis("off")
            continue
        h_stats = all_datasets[density].get("hybrid_stats")
        if not h_stats:
            ax.axis("off")
            continue

        counts = h_stats["counts"]
        labels = [f"Short\n(d=2-3)\n{counts['short']}", 
                  f"Medium\n(d=4-5)\n{counts['medium']}",
                  f"Long\n(d>5)\n{counts['long']}"]
        sizes = [counts["short"], counts["medium"], counts["long"]]
        colors = ["#E07B54", "#FFD700", "#5B8DB8"]

        if sum(sizes) == 0:
            ax.axis("off")
            continue

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.75, textprops={"fontsize": 8}
        )
        ax.set_title(f"{density.capitalize()}\nGraph", fontsize=10)

    fig.suptitle(f"Hybrid Negative Sampling — Bucket Distribution\nDataset: {dataset_name}",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_hybrid_buckets.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 4 — Delta chart: hybrid improvement over baselines
# ════════════════════════════════════════════════════════════════════════════════

def plot_improvement_delta(comparison_df: pd.DataFrame, dataset_name: str):
    """
    Plots F1 delta (hybrid − baseline) as horizontal bar chart.
    Green = hybrid better, Red = hybrid worse.
    """
    if "delta_vs_longest_f1_score" not in comparison_df.columns:
        print("  ⚠  No delta data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, title, baseline in zip(
        axes,
        ["delta_vs_shortest_f1_score", "delta_vs_longest_f1_score"],
        ["Hybrid vs Shortest-Path", "Hybrid vs Longest-Path"],
        ["Shortest", "Longest"]
    ):
        if col not in comparison_df.columns:
            ax.axis("off")
            continue

        plot_data = comparison_df[["density", "model", col]].dropna().copy()
        plot_data["label"] = plot_data["model"] + " (" + plot_data["density"] + ")"
        plot_data = plot_data.sort_values(col, ascending=True)

        colors = ["#2CA02C" if v >= 0 else "#D62728" for v in plot_data[col]]
        ax.barh(plot_data["label"], plot_data[col], color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"F1 Delta: {title}")
        ax.set_xlabel("ΔF1-Score (Hybrid − Baseline)")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Hybrid Sampling Improvement Analysis\nDataset: {dataset_name}",
                 fontsize=12)
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_hybrid_delta.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 5 — Performance across densities (reproduces base paper style)
# ════════════════════════════════════════════════════════════════════════════════

def plot_density_progression(results_df: pd.DataFrame, dataset_name: str,
                              model: str = "GB"):
    sub = results_df[
        (results_df["model"] == model)
    ].copy()
    sub["density"] = pd.Categorical(sub["density"], categories=DENSITY_ORDER, ordered=True)
    sub = sub.sort_values(["strategy", "density"])

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    linestyles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, strat in zip(axes, STRATEGY_ORDER):
        s = sub[sub["strategy"] == strat]
        for metric, ls in zip(metrics, linestyles):
            vals = [s[s["density"] == d][metric].values for d in DENSITY_ORDER]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax.plot(DENSITY_ORDER, vals, marker="o", linestyle=ls, label=metric.upper())
            for x, y in zip(DENSITY_ORDER, vals):
                if not np.isnan(y):
                    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=6.5)
        ax.set_title(f"Strategy: {strat.capitalize()}", fontsize=11)
        ax.set_ylabel("Score")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{model} Performance Across Graph Densities\nDataset: {dataset_name}",
                 fontsize=12)
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_density_progression_{model}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# FIG 6 — Feature correlation matrix
# ════════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(feature_store: dict, dataset_name: str):
    """
    Plot correlation matrices for features (sparse & dense, shortest & longest).
    Reproduces base paper Figures 3-4.
    """
    FEAT_COLS = ["CN","PA","JCI","Salton","Sorenson","HPI","HDI","LHN","AA","RA"]

    combos = [("sparse", "shortest"), ("sparse", "longest"),
              ("dense",  "shortest"), ("dense",  "longest"),
              ("dense",  "hybrid")]

    valid_combos = [(d, s) for d, s in combos
                    if d in feature_store and s in feature_store[d]]

    n = len(valid_combos)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (density, strat) in zip(axes, valid_combos):
        df = feature_store[density][strat]
        corr = df[FEAT_COLS].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.3, ax=ax, vmin=-1, vmax=1,
                    annot_kws={"size": 6}, cbar=False)
        ax.set_title(f"{density.capitalize()} / {strat.capitalize()}", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    fig.suptitle(f"Feature Correlation Matrices — {dataset_name}", fontsize=12)
    fig.tight_layout()

    path = f"results/figures/{dataset_name}_correlation_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    for dataset_name in ["facebook_4k", "facebook_1k"]:
        results_path = f"results/tables/{dataset_name}_results.csv"
        comp_path    = f"results/tables/{dataset_name}_hybrid_comparison.csv"
        feat_path    = f"data/processed/{dataset_name}_features.pkl"
        ds_path      = f"data/processed/{dataset_name}_datasets.pkl"

        if not os.path.exists(results_path):
            print(f"⚠  Skipping {dataset_name} — run step5 first")
            continue

        print(f"\n{'═'*50}")
        print(f"Generating figures: {dataset_name}")
        print(f"{'═'*50}")

        results_df    = pd.read_csv(results_path)
        comparison_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()

        # F1 comparison for key models
        for model in ["GB", "RF", "LogR"]:
            plot_f1_comparison(results_df, dataset_name, model)

        # ROC-AUC heatmaps
        for density in ["sparse", "moderate", "dense"]:
            plot_roc_auc_heatmap(results_df, dataset_name, density)

        # Delta analysis
        if not comparison_df.empty:
            plot_improvement_delta(comparison_df, dataset_name)

        # Density progression
        for model in ["GB", "RF"]:
            plot_density_progression(results_df, dataset_name, model)

        # Feature correlation
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                feature_store = pickle.load(f)
            plot_correlation_matrix(feature_store, dataset_name)

        # Hybrid bucket pie charts
        if os.path.exists(ds_path):
            with open(ds_path, "rb") as f:
                all_datasets = pickle.load(f)
            plot_hybrid_bucket_distribution(all_datasets, dataset_name)

    print("\n✓ Step 6 complete. All figures saved to results/figures/")
