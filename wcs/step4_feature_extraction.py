"""
STEP 4: Feature Extraction
============================
Computes 10 similarity-based features from past graph G' for every (u,v) pair.

Features (same as Raghuwanshi et al. 2025):
  1.  Common Neighbors (CN)
  2.  Preferential Attachment (PA)
  3.  Jaccard Coefficient (JCI)
  4.  Salton Index (SI)
  5.  Sorenson Index
  6.  Hub Promoted Index (HPI)
  7.  Hub Depressed Index (HDI)
  8.  Leicht-Holme-Newman Index (LHN)
  9.  Adamic/Adar Index (AA)
  10. Resource Allocation Index (RA)
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import math

# ════════════════════════════════════════════════════════════════════════════════
# 1. SIMILARITY METRICS
# ════════════════════════════════════════════════════════════════════════════════

def common_neighbors(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))

def preferential_attachment(G, u, v):
    return G.degree(u) * G.degree(v)

def jaccard_coefficient(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    union = nu | nv
    if len(union) == 0:
        return 0.0
    return len(nu & nv) / len(union)

def salton_index(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    denom = math.sqrt(len(nu) * len(nv))
    if denom == 0:
        return 0.0
    return len(nu & nv) / denom

def sorenson_index(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    denom = len(nu) + len(nv)
    if denom == 0:
        return 0.0
    return 2 * len(nu & nv) / denom

def hub_promoted_index(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    denom = max(len(nu), len(nv))
    if denom == 0:
        return 0.0
    return len(nu & nv) / denom

def hub_depressed_index(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    denom = min(len(nu), len(nv))
    if denom == 0:
        return 0.0
    return len(nu & nv) / denom

def leicht_holme_newman(G, u, v):
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    denom = len(nu) * len(nv)
    if denom == 0:
        return 0.0
    return len(nu & nv) / denom

def adamic_adar(G, u, v):
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg_w = G.degree(w)
        if deg_w > 1:
            score += 1.0 / math.log(deg_w)
    return score

def resource_allocation(G, u, v):
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        deg_w = G.degree(w)
        if deg_w > 0:
            score += 1.0 / deg_w
    return score


FEATURE_NAMES = [
    "CN", "PA", "JCI", "Salton", "Sorenson",
    "HPI", "HDI", "LHN", "AA", "RA"
]

FEATURE_FUNCS = [
    common_neighbors, preferential_attachment, jaccard_coefficient,
    salton_index, sorenson_index, hub_promoted_index, hub_depressed_index,
    leicht_holme_newman, adamic_adar, resource_allocation
]


# ════════════════════════════════════════════════════════════════════════════════
# 2. BATCH FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_features(G_past: nx.Graph, samples: list) -> pd.DataFrame:
    """
    Extract 10 similarity features for each (u, v, label) sample.

    Args:
        G_past  : past graph used for feature computation
        samples : list of (u, v, label) tuples

    Returns:
        DataFrame with columns [CN, PA, ..., RA, label]
    """
    rows = []
    for u, v, label in tqdm(samples, desc="  Extracting features"):
        if not G_past.has_node(u) or not G_past.has_node(v):
            # nodes missing from past graph — use zeros
            feats = [0.0] * len(FEATURE_FUNCS)
        else:
            feats = [fn(G_past, u, v) for fn in FEATURE_FUNCS]
        rows.append(feats + [label])

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    return df


# ════════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    datasets_to_process = ["facebook_4k", "facebook_1k"]

    for dataset_name in datasets_to_process:
        dataset_path = f"data/processed/{dataset_name}_datasets.pkl"

        if not os.path.exists(dataset_path):
            print(f"⚠  Skipping {dataset_name} — run step3 first")
            continue

        print(f"\n{'═'*60}")
        print(f"Feature extraction: {dataset_name}")
        print(f"{'═'*60}")

        with open(dataset_path, "rb") as f:
            all_datasets = pickle.load(f)

        feature_store = {}

        STRATEGIES = ["ds_shortest", "ds_longest", "ds_hybrid"]
        STRATEGY_LABELS = {"ds_shortest": "shortest", "ds_longest": "longest", "ds_hybrid": "hybrid"}

        for density_level, info in all_datasets.items():
            G_past = info["G_past"]
            feature_store[density_level] = {}

            for strat_key in STRATEGIES:
                strat_label = STRATEGY_LABELS[strat_key]
                samples = info[strat_key]["samples"]

                print(f"\n  [{density_level.upper()}] [{strat_label.upper()}] "
                      f"Extracting from {len(samples)} samples...")

                df = extract_features(G_past, samples)
                feature_store[density_level][strat_label] = df

                # Quick sanity check
                print(f"  Shape: {df.shape} | label dist: "
                      f"{df['label'].value_counts().to_dict()}")

        # Save feature DataFrames
        out_path = f"data/processed/{dataset_name}_features.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(feature_store, f)
        print(f"\n✓ Features saved to {out_path}")

        # Also save as CSV for inspection
        for density_level, strats in feature_store.items():
            for strat_label, df in strats.items():
                csv_path = (f"data/processed/{dataset_name}_"
                            f"{density_level}_{strat_label}.csv")
                df.to_csv(csv_path, index=False)

    print("\n✓ Step 4 complete. Proceed to step5_ml_training.py")
