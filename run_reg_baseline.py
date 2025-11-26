"""
run_logreg_only.py

Minimal, engineering-faithful replication of the logistic regression baseline
from SAE-Probes, with visualization.

- Uses utils_data.get_xy_traintest / get_layers / get_datasets
- Reimplements utils_training.find_best_reg (logistic regression sweep)
- Writes CSVs and makes plots similar to the original repo style.

Place this file in the SAE-Probes repo root and run:

    python run_logreg_only.py --model_name gemma-2-9b --mode normal

Requirements: same env as README (sklearn, numpy, pandas, matplotlib, etc.)
"""

import os
import argparse
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeavePOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Import from the repo – same entry points as run_baselines.py
from utils_data import (
    get_xy_traintest,
    get_dataset_sizes,
    get_datasets,
    get_layers,
)


# ----------------------------------------------------------------------
# 1. Cross-validation utilities (copied / cleaned up from utils_training.py)
# ----------------------------------------------------------------------


def get_cv(X_train: np.ndarray):
    """
    Cross-validation strategy used in utils_training.find_best_reg.

    - <= 12 samples: LeavePOut(2)
    - < 128 samples: StratifiedKFold(6)
    - otherwise: single holdout, 20% (max 100) samples as validation
    """
    n_samples = X_train.shape[0]
    if n_samples <= 12:
        cv = LeavePOut(2)
    elif n_samples < 128:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    else:
        val_size = min(int(0.2 * n_samples), 100)  # 20% or max 100
        train_size = n_samples - val_size
        cv = [
            (list(range(train_size)), list(range(train_size, n_samples)))
        ]  # single split
    return cv


def get_splits(cv, X_train: np.ndarray, y_train: np.ndarray):
    """
    Filter CV splits to ensure validation sets contain both classes
    (mirrors utils_training.get_splits).
    """
    # If this is a scikit-learn CV object with split()
    if hasattr(cv, "split"):
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            if len(np.unique(y_train[val_idx])) == 2:
                splits.append((train_idx, val_idx))
    else:
        # Predefined list of (train, val) splits
        splits = cv
    return splits


# ----------------------------------------------------------------------
# 2. Logistic regression sweep (engineering-faithful find_best_reg)
# ----------------------------------------------------------------------


def find_best_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plot: bool = False,
    n_jobs: int = -1,
    parallel: bool = False,
    penalty: str = "l2",
    seed: int = 1,
    return_classifier: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], LogisticRegression]:
    """
    Reimplementation of utils_training.find_best_reg for logistic regression.

    - Sweeps C over logspace(5, -5, 10)
    - Uses CV defined by get_cv / get_splits
    - Returns metrics dict with:
        - 'val_auc', 'test_auc', 'test_f1', 'test_acc'
    """
    metrics: Dict[str, float] = {}

    # Very small datasets: skip CV and just fit default parameters
    if X_train.shape[0] <= 3:
        if penalty == "l1":
            final_model = LogisticRegression(
                penalty="l1", solver="saga", random_state=seed, max_iter=1000
            )
        else:
            final_model = LogisticRegression(
                random_state=seed, max_iter=1000
            )

        # Shuffle to match original seed behavior
        rng = np.random.RandomState(seed)
        shuffle_idx = rng.permutation(len(X_train))
        X_train_shuf = X_train[shuffle_idx]
        y_train_shuf = y_train[shuffle_idx]

        final_model.fit(X_train_shuf, y_train_shuf)
        y_test_pred = final_model.predict(X_test)
        y_test_proba = final_model.predict_proba(X_test)[:, 1]

        metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
        metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
        metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
        metrics["val_auc"] = roc_auc_score(
            y_train_shuf, final_model.predict_proba(X_train_shuf)[:, 1]
        )

        if return_classifier:
            return metrics, final_model
        return metrics

    # Otherwise: full CV-based C sweep
    cv = get_cv(X_train)
    Cs = np.logspace(5, -5, 10)
    avg_scores: List[float] = []

    def evaluate_fold(C_val, train_index, val_index):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        if penalty == "l1":
            model = LogisticRegression(
                C=C_val,
                penalty="l1",
                solver="saga",
                random_state=seed,
                max_iter=1000,
            )
        else:
            model = LogisticRegression(
                C=C_val, random_state=seed, max_iter=1000
            )

        model.fit(X_fold_train, y_fold_train)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        return roc_auc_score(y_fold_val, y_pred_proba)

    for C_val in Cs:
        splits = get_splits(cv, X_train, y_train)
        if parallel:
            fold_scores = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(C_val, tr, va) for tr, va in splits
            )
        else:
            fold_scores = [
                evaluate_fold(C_val, tr, va) for tr, va in splits
            ]
        avg_scores.append(float(np.mean(fold_scores)))

    # Pick best C
    best_idx = int(np.argmax(avg_scores))
    best_C = float(Cs[best_idx])

    # Train final model
    if penalty == "l1":
        final_model = LogisticRegression(
            C=best_C,
            penalty="l1",
            solver="saga",
            random_state=seed,
            max_iter=1000,
        )
    else:
        final_model = LogisticRegression(
            C=best_C, random_state=seed, max_iter=1000
        )

    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(X_train))
    X_train_shuf = X_train[shuffle_idx]
    y_train_shuf = y_train[shuffle_idx]

    final_model.fit(X_train_shuf, y_train_shuf)
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    metrics["test_f1"] = f1_score(y_test, y_test_pred, average="weighted")
    metrics["test_acc"] = accuracy_score(y_test, y_test_pred)
    metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
    metrics["val_auc"] = float(np.max(avg_scores))

    if plot:
        plt.figure(figsize=(4, 3))
        plt.semilogx(Cs, avg_scores, marker="o")
        plt.xlabel("Inverse Regularization Strength C")
        plt.ylabel("AUC on validation")
        plt.title(
            f"Logistic Regression vs C\n"
            f"Best C={best_C:.3g}, val AUC={metrics['val_auc']:.3f}"
        )
        plt.tight_layout()
        plt.show()

    if return_classifier:
        return metrics, final_model
    return metrics


# ----------------------------------------------------------------------
# 3. Running logreg baseline over datasets / layers (normal regime)
# ----------------------------------------------------------------------


def run_logreg_dataset_layer(
    layer: int,
    numbered_dataset: str,
    model_name: str = "gemma-2-9b",
    out_root: str = "data/logreg_only_results",
) -> Dict[str, Any] | None:
    """
    Equivalent of run_baseline_dataset_layer, but only for 'logreg':
    - chooses num_train the same way
    - calls get_xy_traintest from utils_data
    - calls find_best_reg
    - writes a small per-(layer,dataset) CSV
    """
    dataset_sizes = get_dataset_sizes()
    size = dataset_sizes[numbered_dataset]
    num_train = min(size - 100, 1024)

    savepath = os.path.join(
        out_root,
        model_name,
        "normal",
        "allruns",
        f"layer{layer}_{numbered_dataset}_logreg.csv",
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    if os.path.exists(savepath):
        # Already done
        return None

    X_train, y_train, X_test, y_test = get_xy_traintest(
        num_train, numbered_dataset, layer, model_name=model_name
    )

    metrics = find_best_reg(X_train, y_train, X_test, y_test)

    row = {"dataset": numbered_dataset, "method": "logreg"}
    row.update(metrics)
    df = pd.DataFrame([row])
    df.to_csv(savepath, index=False)

    result_with_meta = dict(row)
    result_with_meta["layer"] = layer
    result_with_meta["num_train"] = num_train
    return result_with_meta


def run_logreg_all_normal(
    model_name: str = "gemma-2-9b",
    out_root: str = "data/logreg_only_results",
) -> pd.DataFrame:
    """
    Sweep:
        - all layers for the model
        - all datasets from get_datasets()
    Returns a combined DataFrame of all runs.
    """
    all_results: List[Dict[str, Any]] = []
    layers = list(get_layers(model_name))
    datasets = list(get_datasets())

    for layer in layers:
        for ds in datasets:
            res = run_logreg_dataset_layer(
                layer, ds, model_name=model_name, out_root=out_root
            )
            if res is not None:
                all_results.append(res)

    if not all_results:
        # Maybe everything already on disk; reload
        all_results = []
        for layer in layers:
            for ds in datasets:
                savepath = os.path.join(
                    out_root,
                    model_name,
                    "normal",
                    "allruns",
                    f"layer{layer}_{ds}_logreg.csv",
                )
                if os.path.exists(savepath):
                    df = pd.read_csv(savepath)
                    df["layer"] = layer
                    all_results.extend(df.to_dict(orient="records"))

    combined = pd.DataFrame(all_results)
    # Also write a per-layer combined file like run_baselines.coalesce_all_baseline_normal
    for layer in combined["layer"].unique():
        layer_df = combined[combined["layer"] == layer].copy()
        layer_savepath = os.path.join(
            "results",
            f"logreg_only_{model_name}",
            "normal_settings",
            f"layer{layer}_results.csv",
        )
        os.makedirs(os.path.dirname(layer_savepath), exist_ok=True)
        layer_df.to_csv(layer_savepath, index=False)

    # Global summary
    summary_path = os.path.join(
        "results",
        f"logreg_only_{model_name}",
        "normal_settings",
        "all_layers_logreg_results.csv",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    combined.to_csv(summary_path, index=False)
    return combined


# ----------------------------------------------------------------------
# 4. Visualization helpers (baseline-style plots)
# ----------------------------------------------------------------------


def plot_logreg_layer_curve(
    combined_df: pd.DataFrame,
    model_name: str,
    save: bool = True,
):
    """
    Plot mean test AUC vs layer (aggregated over datasets).
    Similar in spirit to the layer-wise plots in the notebooks.
    """
    grouped = (
        combined_df.groupby("layer")["test_auc"].agg(["mean", "std"]).reset_index()
    )
    plt.figure(figsize=(6.75, 3))
    plt.errorbar(
        grouped["layer"],
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        linestyle="-",
        capsize=3,
    )
    plt.xlabel("Layer")
    plt.ylabel("Test AUC (mean ± std over datasets)")
    plt.title(f"Logistic Regression Baseline vs Layer ({model_name})")
    plt.ylim(0.5, 1.01)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        plot_dir = os.path.join(
            "plots",
            f"logreg_only_{model_name}",
        )
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, "logreg_layer_curve.png")
        plt.savefig(path, dpi=300)
    plt.show()


def plot_logreg_dataset_bar(
    combined_df: pd.DataFrame,
    model_name: str,
    layer: int = 20,
    save: bool = True,
):
    """
    For a fixed layer (default 20, like in the paper), make a bar plot of
    test AUC for each dataset.

    This echoes the per-dataset view used in some of the repo's plots.
    """
    df_layer = combined_df[combined_df["layer"] == layer].copy()
    df_layer = df_layer.sort_values("test_auc", ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(df_layer)), df_layer["test_auc"])
    plt.xticks(
        np.arange(len(df_layer)),
        df_layer["dataset"],
        rotation=90,
        fontsize=6,
    )
    plt.ylabel("Test AUC")
    plt.title(f"LogReg Test AUC by Dataset (Layer {layer}, {model_name})")
    plt.ylim(0.5, 1.01)
    plt.tight_layout()

    if save:
        plot_dir = os.path.join(
            "plots",
            f"logreg_only_{model_name}",
        )
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, f"logreg_dataset_bar_layer{layer}.png")
        plt.savefig(path, dpi=300)
    plt.show()


# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run ONLY logistic regression baselines from SAE-Probes."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma-2-9b",
        help="Model name as used in utils_data (default: gemma-2-9b)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal"],
        help="Currently only 'normal' regime is implemented here.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, do not generate plots.",
    )
    args = parser.parse_args()

    if args.mode != "normal":
        raise NotImplementedError(
            "This script currently only replicates the 'normal' regime."
        )

    combined = run_logreg_all_normal(model_name=args.model_name)

    summary = {
        "n_rows": int(len(combined)),
        "n_layers": int(combined["layer"].nunique()),
        "n_datasets": int(combined["dataset"].nunique()),
        "metrics": list(
            sorted(
                set(
                    col
                    for col in combined.columns
                    if col
                    not in ["dataset", "method", "layer", "num_train"]
                )
            )
        ),
    }
    print(json.dumps(summary, indent=2))

    if not args.no_plots:
        plot_logreg_layer_curve(combined, model_name=args.model_name)
        # Layer 20 is highlighted in the paper as the main baseline layer
        if 20 in combined["layer"].unique():
            plot_logreg_dataset_bar(
                combined, model_name=args.model_name, layer=20
            )


if __name__ == "__main__":
    main()
