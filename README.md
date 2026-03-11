# -Link-Prediction-Performance-through-Hybrid-Negative-Sampling
Enhancing Link Prediction Performance through Hybrid Negative Sampling
# Enhancing Link Prediction Performance through Hybrid Negative Sampling

>
---

## Overview

This project proposes and implements **Hybrid Negative Sampling**, a novel extension to the empirical link prediction framework by Raghuwanshi et al. (2025). Instead of selecting negative samples from only shortest or longest path lengths (as in the base paper), we proportionally sample from three path-distance buckets:

| Bucket | Distance Range | Proportion | Rationale |
|--------|---------------|------------|-----------|
| Short | d = 2–3 hops | 30% | Challenging near-positive pairs |
| Medium | d = 4–5 hops | 40% | Transitional relationships |
| Long | d > 5 hops | 30% | True negatives |

---

## Base Paper

> Raghuwanshi, M.M., Shobhane, P.D., Balvir, S.U., Goswami, M.M.,
> *"Empirical Study on Impact of Non-Edge Selection and Graph Density on Link Prediction Performance"*,
> Social Network Analysis and Mining, 2025.
> https://doi.org/10.1007/s13278-025-01478-z

---

## Key Results (Facebook 4K Dataset)

### Best F1-Score per Strategy (Gradient Boosting)

| Graph Density | Shortest Path | **Hybrid (Ours)** | Longest Path | Base Paper |
|---------------|--------------|-------------------|--------------|------------|
| Sparse | 0.603 | **0.608** | 0.628 | 0.636 |
| Moderate | 0.865 | **0.952** | 0.979 | 0.981 |
| Dense | 0.923 | **0.980** | 0.998 | 0.999 |

<img width="364" height="200" alt="Image" src="https://github.com/user-attachments/assets/204f734e-f0d8-4ddf-8a60-68f23d852218" />

<img width="1141" height="688" alt="Image" src="https://github.com/user-attachments/assets/87ae94b7-65b5-437b-a261-54545b67f3f2" />

### Key Findings
- Hybrid consistently outperforms shortest-path across ALL density levels
- Moderate graph improvement: **+8.7%** over shortest-path baseline
- Dense graph improvement: **+5.7%** over shortest-path baseline
- Successfully reproduces base paper results within 0.001–0.008 margin

---

## Project Structure

```
hybrid-link-prediction/
│
├── scripts/
│   ├── step1_setup_and_download.py     # Dataset download and setup
│   ├── step2_graph_construction.py     # Build sparse/moderate/dense past graphs
│   ├── step3_negative_sampling.py      # Hybrid + baseline negative sampling
│   ├── step4_feature_extraction.py     # 10 similarity features extraction
│   ├── step5_ml_training.py            # 7 ML models training (reference)
│   └── step6_visualizations.py        # All charts and figures
│
├── results/
│   ├── figures/
│   │   ├── final_comparison_chart.png          # Main result figure
│   │   ├── facebook_4k_hybrid_buckets.png      # Hybrid bucket distribution
│   │   ├── facebook_4k_roc_auc_heatmap_dense.png
│   │   ├── facebook_4k_f1_comparison_GB.png
│   │   ├── facebook_4k_density_progression_GB.png
│   │   └── facebook_4k_correlation_matrices.png
│   └── tables/
│       ├── facebook_4k_results.csv             # All 54 experiment results
│       └── comparison_with_base_paper.csv      # Our results vs base paper
│
├── notebooks/
│   └── hybrid_link_prediction_kaggle.ipynb    # Full Kaggle notebook
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Facebook 4K (SNAP Stanford)**
- Nodes: 4,039 | Edges: 88,234
- Download: https://snap.stanford.edu/data/ego-Facebook.html
- Auto-downloaded by `step1_setup_and_download.py`

| Graph Type | Edges Retained | Dataset Size |
|------------|---------------|--------------|
| Sparse | 4,038 (spanning tree) | 168,392 samples |
| Moderate | 44,117 (50%) | 88,234 samples |
| Dense | 79,410 (90%) | 17,648 samples |

---

## Methodology

### Pipeline

```
Facebook 4K Dataset
       ↓
Past Graph Construction (3 density levels)
       ↓
Negative Sampling (Shortest / Longest / Hybrid)
       ↓
Feature Extraction (10 similarity metrics)
       ↓
ML Training (6 models × 3 strategies × 3 densities = 54 experiments)
       ↓
Evaluation (F1, ROC-AUC, Accuracy, Precision, Recall)
```

### 10 Similarity Features

| Feature | Description |
|---------|-------------|
| Common Neighbors (CN) | Shared neighbor count |
| Preferential Attachment (PA) | Degree product |
| Jaccard Coefficient (JCI) | Normalized neighborhood overlap |
| Salton Index (SI) | Cosine similarity of neighborhoods |
| Sorenson Index | Intersection emphasis |
| Hub Promoted Index (HPI) | Favors high-degree nodes |
| Hub Depressed Index (HDI) | Penalizes high-degree nodes |
| Leicht-Holme-Newman (LHN) | Normalized intersection |
| Adamic/Adar (AA) | Weights low-degree shared neighbors |
| Resource Allocation (RA) | Resource distribution similarity |

### ML Models Evaluated
- Logistic Regression (LogR)
- Random Forest (RF)
- Decision Tree (DT)
- K-Nearest Neighbours (KNN)
- Naive Bayes (NB)
- Gradient Boosting (GB)
- Support Vector Machine (SVM) — dense graph only

---

## How to Run

### Platform: Kaggle (Recommended)

1. Upload all scripts from `scripts/` as a Kaggle dataset
2. Create a new Kaggle notebook with GPU T4 enabled
3. Run cells in order:

```python
# Cell 1 — Setup
import os, shutil, urllib.request, gzip
os.chdir('/kaggle/working')
for d in ['data/raw','data/processed','results/tables','results/figures','models']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)
# copy scripts and download dataset...

# Cell 2 — Graph Construction (~45 min)
%run /kaggle/working/step2_graph_construction.py

# Cell 3 — Negative Sampling (~15 min)
%run /kaggle/working/step3_negative_sampling.py

# Cell 4 — Feature Extraction (~5 min)
%run /kaggle/working/step4_feature_extraction.py

# Cell 5 — ML Training (~15 min)
# Use the fast version in the notebook (SVM skipped for large datasets)

# Cell 6 — Visualizations (~2 min)
%run /kaggle/working/step6_visualizations.py
```

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/hybrid-link-prediction.git
cd hybrid-link-prediction
pip install -r requirements.txt
python scripts/step1_setup_and_download.py
python scripts/step2_graph_construction.py
# ... continue in order
```

---

## Requirements

```
networkx>=3.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
```

---

## Conclusion

Hybrid Negative Sampling provides a statistically balanced alternative to extreme path-length strategies. It significantly improves F1-score on sparse and moderate graphs — conditions that better represent real-world evolving networks — while remaining competitive on dense graphs. The 30-40-30 proportional distribution reduces sampling bias and creates more realistic training data that mirrors natural network topology.

---

## References

1. Raghuwanshi et al. (2025) — Base paper
2. Liben-Nowell & Kleinberg (2007) — Link prediction problem

 IT752: Web and Social Computing | NIT Karnataka, Surathkal  
> Group 4 | Samriddhi Sharma (252IT024) | Sakshi Jitendra Vispute (252IT032)  
> Guide: Dr. Shrutilipi Bhattacharjee | April 2026

3. Adamic & Adar (2003) — Adamic/Adar index
4. Zhou et al. (2009) — Resource allocation index
5. Facebook SNAP Dataset — McAuley & Leskovec (2012)
