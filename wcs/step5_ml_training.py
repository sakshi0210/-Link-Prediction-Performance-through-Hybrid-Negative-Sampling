"""
STEP 5: Machine Learning Training & Evaluation
================================================
Trains 7 classifiers on each combination of:
  - Density level : sparse / moderate / dense
  - Sampling strategy : shortest / longest / hybrid

Models (same as Raghuwanshi et al. 2025):
  1. Logistic Regression (LogR)
  2. Support Vector Machine (SVM)
  3. Random Forest (RF)
  4. Decision Tree (DT)
  5. K-Nearest Neighbours (KNN)
  6. Naive Bayes (NB)
  7. Gradient Boosting (GB)

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Split: 80/20 train-test
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════════════════════
# 1. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_models():
    return {
        "LogR": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM":  SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "RF":   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "DT":   DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN":  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "NB":   GaussianNB(),
        "GB":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


# ════════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 6),
    }


def run_experiment(df: pd.DataFrame, model_name: str, model,
                   scale: bool = False) -> dict:
    """
    Train + evaluate a single model on a feature DataFrame.
    scale=True for SVM, LogR (distance-based models need normalization).
    """
    X = df[["CN","PA","JCI","Salton","Sorenson","HPI","HDI","LHN","AA","RA"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    metrics["train_size"] = len(X_train)
    metrics["test_size"]  = len(X_test)
    return metrics


# ════════════════════════════════════════════════════════════════════════════════
# 3. RUN ALL EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════════

SCALE_MODELS = {"LogR", "SVM", "KNN"}  # models that benefit from scaling

def run_all_experiments(feature_store: dict) -> pd.DataFrame:
    """
    Run all 7 models × 3 strategies × 3 densities = 63 experiments.
    Returns a results DataFrame.
    """
    rows = []
    total = len(feature_store) * 3 * 7  # densities × strategies × models

    with tqdm(total=total, desc="Training models") as pbar:
        for density_level, strategies in feature_store.items():
            for strat_label, df in strategies.items():
                models = get_models()
                for model_name, model in models.items():
                    scale = model_name in SCALE_MODELS
                    try:
                        metrics = run_experiment(df, model_name, model, scale=scale)
                        row = {
                            "density":   density_level,
                            "strategy":  strat_label,
                            "model":     model_name,
                            **metrics
                        }
                        rows.append(row)
                    except Exception as exc:
                        print(f"  ✗ {density_level}/{strat_label}/{model_name}: {exc}")
                    pbar.update(1)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
# 4. PRETTY PRINT RESULTS
# ════════════════════════════════════════════════════════════════════════════════

def print_results_table(results_df: pd.DataFrame, density: str):
    """Print a formatted table for one density level (matches base paper format)."""
    subset = results_df[results_df["density"] == density].copy()
    print(f"\n{'═'*90}")
    print(f"  DENSITY: {density.upper()}")
    print(f"{'─'*90}")
    print(f"  {'Model':<6} {'Strategy':<12} {'Accuracy':<10} {'Precision':<10} "
          f"{'Recall':<10} {'F1-Score':<10} {'ROC-AUC'}")
    print(f"{'─'*90}")
    for _, row in subset.sort_values(["strategy","model"]).iterrows():
        print(f"  {row['model']:<6} {row['strategy']:<12} {row['accuracy']:<10.6f} "
              f"{row['precision']:<10.6f} {row['recall']:<10.6f} "
              f"{row['f1_score']:<10.6f} {row['roc_auc']:.6f}")
    print(f"{'═'*90}")


# ════════════════════════════════════════════════════════════════════════════════
# 5. COMPARE HYBRID vs BASELINES
# ════════════════════════════════════════════════════════════════════════════════

def compare_hybrid_vs_baselines(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta metrics: hybrid - shortest_path and hybrid - longest_path.
    Highlights where hybrid improves over baselines.
    """
    pivot = results_df.pivot_table(
        index=["density", "model"],
        columns="strategy",
        values=["f1_score", "roc_auc", "accuracy"]
    ).reset_index()

    comparison_rows = []
    for density in results_df["density"].unique():
        for model in results_df["model"].unique():
            sub = results_df[
                (results_df["density"] == density) &
                (results_df["model"] == model)
            ].set_index("strategy")

            if "hybrid" not in sub.index:
                continue

            row = {"density": density, "model": model}
            for metric in ["f1_score", "roc_auc", "accuracy"]:
                hybrid_val = sub.loc["hybrid", metric]
                row[f"hybrid_{metric}"] = hybrid_val

                if "shortest" in sub.index:
                    delta = hybrid_val - sub.loc["shortest", metric]
                    row[f"delta_vs_shortest_{metric}"] = round(delta, 6)

                if "longest" in sub.index:
                    delta = hybrid_val - sub.loc["longest", metric]
                    row[f"delta_vs_longest_{metric}"] = round(delta, 6)

            comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


# ════════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    datasets_to_process = ["facebook_4k", "facebook_1k"]

    for dataset_name in datasets_to_process:
        feat_path = f"data/processed/{dataset_name}_features.pkl"

        if not os.path.exists(feat_path):
            print(f"⚠  Skipping {dataset_name} — run step4 first")
            continue

        print(f"\n{'═'*60}")
        print(f"Running experiments: {dataset_name}")
        print(f"{'═'*60}")

        with open(feat_path, "rb") as f:
            feature_store = pickle.load(f)

        # Run all experiments
        results_df = run_all_experiments(feature_store)

        # Save raw results
        results_path = f"results/tables/{dataset_name}_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")

        # Print tables per density
        for density in ["sparse", "moderate", "dense"]:
            if density in results_df["density"].values:
                print_results_table(results_df, density)

        # Hybrid comparison
        comparison_df = compare_hybrid_vs_baselines(results_df)
        comp_path = f"results/tables/{dataset_name}_hybrid_comparison.csv"
        comparison_df.to_csv(comp_path, index=False)
        print(f"\n✓ Hybrid comparison saved to {comp_path}")

        # Print top hybrid improvements
        print("\n  TOP HYBRID IMPROVEMENTS (F1 delta vs longest-path):")
        if "delta_vs_longest_f1_score" in comparison_df.columns:
            top = comparison_df.nlargest(5, "delta_vs_longest_f1_score")[
                ["density","model","hybrid_f1_score","delta_vs_longest_f1_score"]
            ]
            print(top.to_string(index=False))

    print("\n✓ Step 5 complete. Proceed to step6_visualizations.py")
