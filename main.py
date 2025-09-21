"""Training and evaluation script for parallel sentence mining classifiers."""
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from evaluation import evaluate
from plot_curve import plot_metrics_from_list
from sentence_embedding import (to_labse_sentence_embeddings,
                                to_xlmr_sentence_embeddings)

RANDOM_STATE = 42


def build_features(
    hsb_vecs: np.ndarray,
    de_vecs: np.ndarray,
    similarities: np.ndarray,
    *,
    fit_pca: bool,
    pca_hsb: Optional[PCA] = None,
    pca_de: Optional[PCA] = None,
    n_components: float = 0.95,
) -> Tuple[np.ndarray, PCA, PCA]:
    """Project embeddings with PCA and stack similarity column."""
    diff = hsb_vecs - de_vecs
    prod = hsb_vecs * de_vecs
    _ = diff, prod  # keep placeholders for potential future features

    if fit_pca:
        pca_hsb = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(hsb_vecs)
        pca_de = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(de_vecs)

    if pca_hsb is None or pca_de is None:
        raise ValueError("PCA models must be provided when fit_pca is False.")

    cs_pca = pca_hsb.transform(hsb_vecs)
    de_pca = pca_de.transform(de_vecs)

    sim_col = similarities.reshape(-1, 1)
    features = np.hstack([cs_pca, de_pca, sim_col])
    return features, pca_hsb, pca_de


def train_eval(X_train, y_train, X_val, y_val, model, name):
    """Fit model, evaluate on validation split, and return metrics."""
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_val)
    else:
        y_score = model.predict(X_val)

    metrics = evaluate(y_val, y_score)
    metrics["model"] = name
    return metrics


def evaluate_similarity_only(similarities, labels, name="CosineSim"):
    """Baseline using cosine similarity thresholding only."""
    precision, recall, thresholds = precision_recall_curve(labels, similarities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_thr = thresholds[best_idx] if thresholds.size else 0.5
    print(f"{best_thr:.4f}")

    y_pred = (similarities >= best_thr).astype(int)

    metrics = {
        "model": name,
        "best_threshold": best_thr,
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred, zero_division=0),
        "recall": recall_score(labels, y_pred, zero_division=0),
        "f1": f1_score(labels, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(labels, similarities),
        "pr_auc": average_precision_score(labels, similarities),
        "y_true": labels,
        "y_score": similarities,
    }

    print(f"\n=== {name} baseline (cosine similarity only) ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:15}: {v:.4f}")
    return metrics


def final_test(model, X_test, y_test, threshold, *, sim=None, name=""):
    """Evaluate model on the held-out test set using stored threshold."""
    if model is None:
        if sim is None:
            raise ValueError("Similarity scores must be provided for the baseline model.")
        score = sim
    elif hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
    else:
        score = model.predict(X_test)

    y_pred = (score >= threshold).astype(int)
    metrics = {
        "model": name,
        "best_threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score),
        "pr_auc": average_precision_score(y_test, score),
        "y_true": y_test,
        "y_score": score,
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel sentence mining experiment runner")
    parser.add_argument("--input-file", required=True, help="Path to TSV file with columns cs, de, label")
    parser.add_argument("--embedding-model", default="xlmr", choices=("xlmr", "glot500", "labse"))
    parser.add_argument("--pca-components", type=float, default=0.95, help="PCA components or variance to retain")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_embeddings(df, embedding_model):
    if embedding_model == "labse":
        return to_labse_sentence_embeddings(df)
    return to_xlmr_sentence_embeddings(df, embedding_model)


def main():
    args = parse_args()

    df = pd.read_csv(args.input_file, sep="\t")
    if not {"cs", "de", "label"}.issubset(df.columns):
        raise ValueError("Input file must contain columns: cs, de, label")

    label = df["label"].to_numpy()
    df_train, df_temp, y_train, y_temp = train_test_split(
        df,
        label,
        test_size=0.8,
        random_state=args.seed,
        stratify=label,
    )
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_temp,
    )

    cs_train, de_train, sim_train = load_embeddings(df_train, args.embedding_model)
    cs_val, de_val, sim_val = load_embeddings(df_val, args.embedding_model)
    cs_test, de_test, sim_test = load_embeddings(df_test, args.embedding_model)

    X_tr, pca_h, pca_d = build_features(cs_train, de_train, sim_train, fit_pca=True, n_components=args.pca_components)
    X_val, _, _ = build_features(cs_val, de_val, sim_val, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d)
    X_test, _, _ = build_features(cs_test, de_test, sim_test, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d)

    print("HSB PCA components to retain 95% variance:", pca_h.n_components_)
    print("DE  PCA components to retain 95% variance:", pca_d.n_components_)

    models = {
        "Baseline": None,
        "LogReg": LogisticRegression(max_iter=1000, random_state=args.seed),
        "RF": RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1),
        "LGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=args.seed),
        "SVM": make_pipeline(StandardScaler(), LinearSVC(max_iter=1000, dual=True, random_state=args.seed)),
        "XGB": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=args.seed,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=args.seed),
        ),
    }

    fitted_models: Dict[str, Optional[object]] = {}
    metrics_list: List[Dict[str, float]] = []
    best_thresholds: Dict[str, float] = {}

    for name, mdl in models.items():
        print(f"\n===== {name} =====")
        if mdl is None:
            metrics = evaluate_similarity_only(sim_val, y_val, name="Baseline")
            fitted_models[name] = None
        else:
            metrics = train_eval(X_tr, y_train, X_val, y_val, mdl, name)
            fitted_models[name] = mdl
        best_thresholds[name] = metrics["best_threshold"]
        metrics_list.append(metrics)

    metrics_df = plot_metrics_from_list(metrics_list)

    cleaned = []
    for m in metrics_list:
        filtered = {
            k: v
            for k, v in m.items()
            if k in {"model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "best_threshold"}
        }
        cleaned.append(filtered)

    df_summary = pd.DataFrame(cleaned).set_index("model")
    print(df_summary.round(4))

    list_metric = []
    for name, mdl in fitted_models.items():
        thr = best_thresholds[name]
        if models[name] is None:
            metric = final_test(None, X_test, y_test, thr, sim=sim_test, name=name)
        else:
            metric = final_test(mdl, X_test, y_test, thr, name=name)
        list_metric.append(metric)

    df_m = pd.DataFrame(list_metric)
    df_m = df_m[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    print(df_m)


if __name__ == "__main__":
    main()
