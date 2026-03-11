"""
STEP 1: Environment Setup & Dataset Download
============================================
Platform: Google Colab (recommended) or Jupyter Notebook
Run this first to install dependencies and download Facebook datasets.

Datasets:
- Facebook 1K: https://networkrepository.com/socfb-Reed98.php
- Facebook 4K: https://snap.stanford.edu/data/ego-Facebook.html
"""

# ── Install dependencies (run in Colab/terminal) ──────────────────────────────
# !pip install networkx scikit-learn pandas matplotlib seaborn tqdm

import os
import urllib.request
import zipfile
import gzip
import shutil

# ── Directory structure ────────────────────────────────────────────────────────
DIRS = ["data/raw", "data/processed", "results/tables", "results/figures", "models"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)
print("✓ Directories created")

# ── Dataset download instructions ─────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║           DATASET DOWNLOAD INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  FACEBOOK 4K (ego-Facebook from SNAP):                       ║
║  URL: https://snap.stanford.edu/data/facebook_combined.txt.gz║
║  → Save as: data/raw/facebook_4k.txt                         ║
║                                                              ║
║  FACEBOOK 1K:                                                ║
║  URL: https://networkrepository.com (socfb-Reed98)           ║
║  → Save as: data/raw/facebook_1k.txt                         ║
║                                                              ║
║  FORMAT: Each line = "node1 node2" (edge list)               ║
╚══════════════════════════════════════════════════════════════╝
""")

# ── Auto-download Facebook 4K from SNAP ───────────────────────────────────────
def download_facebook_4k():
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    gz_path = "data/raw/facebook_combined.txt.gz"
    out_path = "data/raw/facebook_4k.txt"

    if os.path.exists(out_path):
        print(f"✓ Facebook 4K already exists at {out_path}")
        return

    print("Downloading Facebook 4K dataset...")
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"✓ Facebook 4K saved to {out_path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("  Please download manually from: https://snap.stanford.edu/data/ego-Facebook.html")

download_facebook_4k()

# ── Verify datasets ────────────────────────────────────────────────────────────
import networkx as nx

def verify_dataset(path, name):
    if not os.path.exists(path):
        print(f"✗ {name} not found at {path}")
        return False
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)  # remove multi-edges
    print(f"✓ {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return True

verify_dataset("data/raw/facebook_4k.txt", "Facebook 4K")
if os.path.exists("data/raw/facebook_1k.txt"):
    verify_dataset("data/raw/facebook_1k.txt", "Facebook 1K")
else:
    print("ℹ  Facebook 1K not found — will use 4K only (sufficient for experiments)")

print("\n✓ Setup complete. Proceed to step2_graph_construction.py")
