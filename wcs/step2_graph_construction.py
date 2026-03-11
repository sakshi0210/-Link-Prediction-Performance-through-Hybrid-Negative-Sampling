"""
STEP 2: Past Graph Construction
================================
Builds three density variants of the past graph from the present graph:
  - Sparse:   |E'| ≈ |V| - 1  (spanning tree)
  - Moderate: |E'| ≈ 0.50 * |E|
  - Dense:    |E'| ≈ 0.90 * |E|

Methodology follows Raghuwanshi et al. (2025):
  - Edges are removed while preserving graph connectivity
  - Removed edges → positive samples for ML training
"""

import os
import random
import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════════
# 1. LOAD GRAPH
# ════════════════════════════════════════════════════════════════════════════════

def load_graph(path: str) -> nx.Graph:
    """Load edge-list file into undirected NetworkX graph."""
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)                    # ensure simple graph
    G.remove_edges_from(nx.selfloop_edges(G))
    # Keep only largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ════════════════════════════════════════════════════════════════════════════════
# 2. EDGE REMOVAL — connectivity-preserving
# ════════════════════════════════════════════════════════════════════════════════

def get_bridge_edges(G: nx.Graph) -> set:
    """Return set of bridge edges (removal disconnects graph)."""
    return set(nx.bridges(G))


def remove_edges_safely(G: nx.Graph, target_edge_count: int,
                         random_seed: int = 42) -> tuple:
    """
    Remove edges from G until |E'| ≈ target_edge_count.
    Only removes non-bridge edges to preserve connectivity.

    Returns:
        G_past      : past graph with reduced edges
        removed_edges: list of removed edges (positive samples)
    """
    G_past = G.copy()
    removed_edges = []
    rng = random.Random(random_seed)

    edges_to_remove = G.number_of_edges() - target_edge_count
    if edges_to_remove <= 0:
        return G_past, removed_edges

    print(f"  Removing {edges_to_remove} edges (target={target_edge_count})...")
    attempts = 0
    max_attempts = edges_to_remove * 20  # safety cap

    with tqdm(total=edges_to_remove, desc="  Edge removal") as pbar:
        while len(removed_edges) < edges_to_remove and attempts < max_attempts:
            attempts += 1
            # Sample a random edge
            edge_list = list(G_past.edges())
            u, v = rng.choice(edge_list)

            # Skip bridge edges
            G_past.remove_edge(u, v)
            if nx.is_connected(G_past):
                removed_edges.append((u, v))
                pbar.update(1)
            else:
                G_past.add_edge(u, v)  # restore

    print(f"  Removed {len(removed_edges)} edges. Past graph: {G_past.number_of_edges()} edges")
    return G_past, removed_edges


# ════════════════════════════════════════════════════════════════════════════════
# 3. BUILD ALL THREE DENSITY LEVELS
# ════════════════════════════════════════════════════════════════════════════════

def build_past_graphs(G: nx.Graph, dataset_name: str) -> dict:
    """
    Construct sparse, moderate, and dense past graphs.

    Returns dict with keys: 'sparse', 'moderate', 'dense'
    Each value: {'G_past': nx.Graph, 'removed_edges': list, 'density_label': str}
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()

    targets = {
        "sparse":   n - 1,          # spanning tree
        "moderate": int(0.50 * e),  # 50% edges retained
        "dense":    int(0.90 * e),  # 90% edges retained
    }

    results = {}
    for level, target in targets.items():
        print(f"\n[{level.upper()}] Target edges in past graph: {target}")
        G_past, removed = remove_edges_safely(G, target_edge_count=target)
        sparsity = 1 - (2 * G_past.number_of_edges()) / (n * (n - 1))
        results[level] = {
            "G_past":        G_past,
            "removed_edges": removed,
            "density_label": level,
            "sparsity":      round(sparsity, 6),
            "n_edges_past":  G_past.number_of_edges(),
            "n_removed":     len(removed),
        }
        print(f"  Sparsity S = {sparsity:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = f"data/processed/{dataset_name}_past_graphs.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✓ Past graphs saved to {save_path}")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# 4. STATISTICS SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

def print_summary(G: nx.Graph, past_graphs: dict, dataset_name: str):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    ne = n * (n - 1) // 2 - e  # total non-edges in present graph

    print(f"\n{'═'*60}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'═'*60}")
    print(f"  Present graph:  {n} nodes, {e} edges, {ne} non-edges")
    print(f"{'─'*60}")
    print(f"  {'Level':<10} {'Past|E|':<10} {'Removed':<10} {'Dataset size':<14} {'Sparsity'}")
    print(f"{'─'*60}")
    for level, info in past_graphs.items():
        dataset_size = 2 * info["n_removed"]  # balanced: equal pos/neg
        print(f"  {level:<10} {info['n_edges_past']:<10} {info['n_removed']:<10} "
              f"{dataset_size:<14} {info['sparsity']:.6f}")
    print(f"{'═'*60}\n")


# ════════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    datasets = {
        "facebook_4k": "data/raw/facebook_4k.txt",
        "facebook_1k": "data/raw/facebook_1k.txt",
    }

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"⚠  Skipping {name} — file not found at {path}")
            continue

        print(f"\n{'═'*60}")
        print(f"Processing: {name}")
        print(f"{'═'*60}")

        G = load_graph(path)

        # Save present graph
        with open(f"data/processed/{name}_present.pkl", "wb") as f:
            pickle.dump(G, f)

        past_graphs = build_past_graphs(G, name)
        print_summary(G, past_graphs, name)

    print("\n✓ Step 2 complete. Proceed to step3_negative_sampling.py")
