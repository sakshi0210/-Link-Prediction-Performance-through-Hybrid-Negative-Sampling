"""
STEP 3: Negative Sample Generation
====================================
Implements THREE strategies for negative (non-edge) selection:

  1. SHORTEST-PATH  (base paper baseline)   — select non-edges with minimum d(u,v)
  2. LONGEST-PATH   (base paper baseline)   — select non-edges with maximum d(u,v)
  3. HYBRID         (YOUR NOVELTY)          — proportional mix: 30% short (2-3 hops)
                                                                  40% medium (4-5 hops)
                                                                  30% long  (>5 hops)

This file also assembles the final balanced datasets (pos + neg).
"""

import os
import pickle
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════════
# 1. COMPUTE SHORTEST PATHS FOR ALL CANDIDATE NON-EDGES
# ════════════════════════════════════════════════════════════════════════════════

def compute_non_edge_paths(G_present: nx.Graph,
                            removed_edges: list,
                            sample_limit: int = None) -> dict:
    """
    For each non-edge (u,v) in G_present, compute shortest path length d(u,v).

    To avoid O(V^2) BFS on large graphs, we sample candidate non-edges
    up to `sample_limit` (use None for exact, or e.g. 500_000 for speed).

    Returns:
        path_dict: {(u,v): path_length}  — sorted by path length
    """
    n = G_present.number_of_nodes()
    e = G_present.number_of_edges()
    nodes = list(G_present.nodes())
    edge_set = set(G_present.edges())
    # add reverse direction for undirected
    edge_set |= {(v, u) for u, v in edge_set}

    needed = len(removed_edges)  # we need at least this many negatives

    # Build candidate non-edge list
    print(f"  Building candidate non-edges (need {needed * 4} minimum candidates)...")

    # For large graphs, sample node pairs instead of enumerating all
    MAX_CANDIDATES = max(needed * 10, 200_000)

    path_dict = {}
    rng = random.Random(42)

    if sample_limit or n > 2000:
        # Sampling approach for large graphs
        attempts = 0
        max_attempts = MAX_CANDIDATES * 5
        with tqdm(total=MAX_CANDIDATES, desc="  Sampling non-edges") as pbar:
            while len(path_dict) < MAX_CANDIDATES and attempts < max_attempts:
                attempts += 1
                u, v = rng.sample(nodes, 2)
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key in path_dict:
                    continue
                if (u, v) in edge_set:
                    continue
                # Compute path length via BFS
                try:
                    d = nx.shortest_path_length(G_present, u, v)
                    path_dict[key] = d
                    pbar.update(1)
                except nx.NetworkXNoPath:
                    pass
    else:
        # Exact enumeration for small graphs
        for i, u in enumerate(tqdm(nodes, desc="  Computing paths")):
            paths = nx.single_source_shortest_path_length(G_present, u)
            for v, d in paths.items():
                if v <= u:
                    continue
                key = (min(u, v), max(u, v))
                if key in path_dict:
                    continue
                if (u, v) not in edge_set:
                    path_dict[key] = d

    print(f"  Found {len(path_dict)} candidate non-edges")
    return path_dict


# ════════════════════════════════════════════════════════════════════════════════
# 2. BASELINE STRATEGIES  (Raghuwanshi et al. 2025)
# ════════════════════════════════════════════════════════════════════════════════

def sample_shortest_path(path_dict: dict, n_needed: int) -> list:
    """Select n_needed non-edges with MINIMUM path length (ascending sort)."""
    sorted_pairs = sorted(path_dict.items(), key=lambda x: x[1])
    selected = [pair for pair, _ in sorted_pairs[:n_needed]]
    print(f"  [Shortest-path] Selected {len(selected)} non-edges "
          f"(d range: {sorted_pairs[0][1]} – {sorted_pairs[min(n_needed-1, len(sorted_pairs)-1)][1]})")
    return selected


def sample_longest_path(path_dict: dict, n_needed: int) -> list:
    """Select n_needed non-edges with MAXIMUM path length (descending sort)."""
    sorted_pairs = sorted(path_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [pair for pair, _ in sorted_pairs[:n_needed]]
    print(f"  [Longest-path] Selected {len(selected)} non-edges "
          f"(d range: {sorted_pairs[0][1]} – {sorted_pairs[min(n_needed-1, len(sorted_pairs)-1)][1]})")
    return selected


# ════════════════════════════════════════════════════════════════════════════════
# 3. HYBRID NEGATIVE SAMPLING  ← YOUR NOVEL CONTRIBUTION
# ════════════════════════════════════════════════════════════════════════════════

# Distribution: 30% short (2-3), 40% medium (4-5), 30% long (>5)
HYBRID_CONFIG = {
    "short":  {"range": (2, 3),   "proportion": 0.30, "rationale": "challenging near-positives"},
    "medium": {"range": (4, 5),   "proportion": 0.40, "rationale": "transitional relationships"},
    "long":   {"range": (6, 999), "proportion": 0.30, "rationale": "true negatives"},
}


def sample_hybrid(path_dict: dict, n_needed: int,
                   config: dict = HYBRID_CONFIG,
                   random_seed: int = 42) -> list:
    """
    HYBRID NEGATIVE SAMPLING STRATEGY (Novel contribution).

    Proportionally selects non-edges from three path-distance buckets:
      Short  (d=2-3) : 30%  — near-positive pairs, adds classification difficulty
      Medium (d=4-5) : 40%  — transitional, improves decision boundary learning
      Long   (d>5)   : 30%  — distant pairs, true negatives, stable classification

    Args:
        path_dict  : {(u,v): path_length}
        n_needed   : total number of negatives required
        config     : bucket configuration (proportion + d-range)

    Returns:
        List of selected non-edge pairs
    """
    rng = random.Random(random_seed)

    # Partition non-edges into buckets
    buckets = {cat: [] for cat in config}
    for (u, v), d in path_dict.items():
        for cat, cfg in config.items():
            lo, hi = cfg["range"]
            if lo <= d <= hi:
                buckets[cat].append((u, v))
                break

    print(f"\n  [HYBRID] Bucket sizes:")
    for cat, pairs in buckets.items():
        print(f"    {cat:8s} (d={config[cat]['range']}): {len(pairs)} candidates")

    # Sample from each bucket
    selected = []
    for cat, cfg in config.items():
        quota = int(cfg["proportion"] * n_needed)
        pool  = buckets[cat]

        if len(pool) < quota:
            print(f"  ⚠  {cat}: only {len(pool)} available, need {quota}. Using all.")
            sampled = pool
        else:
            sampled = rng.sample(pool, quota)

        selected.extend(sampled)
        print(f"    {cat:8s}: sampled {len(sampled)}/{quota} ({cfg['proportion']*100:.0f}%)")

    # Fill remainder if any shortfall (due to rounding or small buckets)
    remaining = n_needed - len(selected)
    if remaining > 0:
        used = set(selected)
        all_candidates = [(p, d) for p, d in path_dict.items() if p not in used]
        extra = rng.sample(all_candidates, min(remaining, len(all_candidates)))
        selected.extend([p for p, _ in extra])
        print(f"    filled {len(extra)} remaining slots from unused candidates")

    print(f"  [HYBRID] Total selected: {len(selected)} non-edges")
    return selected[:n_needed]


def get_hybrid_stats(selected: list, path_dict: dict) -> dict:
    """Return per-bucket counts and proportions for the selected hybrid set."""
    stats = {"short": 0, "medium": 0, "long": 0, "other": 0}
    for pair in selected:
        d = path_dict.get(pair, path_dict.get((pair[1], pair[0]), -1))
        if 2 <= d <= 3:
            stats["short"] += 1
        elif 4 <= d <= 5:
            stats["medium"] += 1
        elif d >= 6:
            stats["long"] += 1
        else:
            stats["other"] += 1
    total = len(selected)
    proportions = {k: round(v / total, 3) for k, v in stats.items()}
    return {"counts": stats, "proportions": proportions, "total": total}


# ════════════════════════════════════════════════════════════════════════════════
# 4. ASSEMBLE BALANCED DATASET
# ════════════════════════════════════════════════════════════════════════════════

def build_dataset(removed_edges: list, negative_pairs: list) -> dict:
    """
    Create balanced dataset dict:
      label=1  → positive (removed edges, actual connections)
      label=0  → negative (sampled non-edges)
    """
    n = min(len(removed_edges), len(negative_pairs))  # ensure balance
    positives = [(u, v, 1) for u, v in removed_edges[:n]]
    negatives = [(u, v, 0) for u, v in negative_pairs[:n]]
    data = positives + negatives
    random.shuffle(data)
    print(f"  Dataset: {n} positives + {n} negatives = {2*n} total samples")
    return {"samples": data, "n_pos": n, "n_neg": n}


# ════════════════════════════════════════════════════════════════════════════════
# 5. MAIN — generate all datasets
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    datasets_to_process = ["facebook_4k", "facebook_1k"]

    for dataset_name in datasets_to_process:
        past_graph_path = f"data/processed/{dataset_name}_past_graphs.pkl"
        present_path    = f"data/processed/{dataset_name}_present.pkl"

        if not os.path.exists(past_graph_path):
            print(f"⚠  Skipping {dataset_name} — run step2 first")
            continue

        print(f"\n{'═'*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'═'*60}")

        with open(past_graph_path, "rb") as f:
            past_graphs = pickle.load(f)
        with open(present_path, "rb") as f:
            G_present = pickle.load(f)

        all_datasets = {}

        for density_level, info in past_graphs.items():
            print(f"\n── {density_level.upper()} GRAPH ──────────────────────────")
            G_past       = info["G_past"]
            removed_edges = info["removed_edges"]
            n_needed      = len(removed_edges)

            print(f"  Need {n_needed} negative samples")

            # Compute path lengths on the PAST graph (per base paper methodology)
            print("  Computing path lengths on past graph...")
            path_dict = compute_non_edge_paths(G_past, removed_edges)

            # ── Strategy 1: Shortest path (baseline) ────────────────────────
            print("\n  Strategy: SHORTEST PATH (baseline)")
            neg_shortest = sample_shortest_path(path_dict, n_needed)
            ds_shortest  = build_dataset(removed_edges, neg_shortest)

            # ── Strategy 2: Longest path (baseline) ─────────────────────────
            print("\n  Strategy: LONGEST PATH (baseline)")
            neg_longest = sample_longest_path(path_dict, n_needed)
            ds_longest  = build_dataset(removed_edges, neg_longest)

            # ── Strategy 3: Hybrid (NOVEL) ───────────────────────────────────
            print("\n  Strategy: HYBRID (novel contribution)")
            neg_hybrid = sample_hybrid(path_dict, n_needed)
            ds_hybrid  = build_dataset(removed_edges, neg_hybrid)

            # Print hybrid stats
            h_stats = get_hybrid_stats(neg_hybrid, path_dict)
            print(f"\n  Hybrid bucket distribution:")
            for cat, prop in h_stats["proportions"].items():
                print(f"    {cat:8s}: {h_stats['counts'][cat]:6d} ({prop*100:.1f}%)")

            all_datasets[density_level] = {
                "G_past":         G_past,
                "removed_edges":  removed_edges,
                "path_dict":      path_dict,
                "ds_shortest":    ds_shortest,
                "ds_longest":     ds_longest,
                "ds_hybrid":      ds_hybrid,
                "hybrid_stats":   h_stats,
            }

        # Save
        out_path = f"data/processed/{dataset_name}_datasets.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(all_datasets, f)
        print(f"\n✓ All datasets saved to {out_path}")

    print("\n✓ Step 3 complete. Proceed to step4_feature_extraction.py")
