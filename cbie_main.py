from sentence_embedding import to_xlmr_sentence_embeddings
from sentence_embedding import to_labse_sentence_embeddings
from plot_curve import plot_metrics_from_list
from evaluation import evaluate
import utils as utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def build_features(hsb_vecs, de_vecs, sim, fit_pca=True, pca_hsb=None, pca_de=None,
                   pca_diff=None,
                   n_components=0.95):
    diff = np.abs(hsb_vecs - de_vecs)
    prod = hsb_vecs * de_vecs
    l2_dist = np.linalg.norm(diff, axis=1, keepdims=True)
    if fit_pca:
        pca_hsb = PCA(n_components=n_components).fit(hsb_vecs)
        pca_de = PCA(n_components=n_components).fit(de_vecs)
        pca_diff = PCA(n_components=n_components).fit(diff)
    # PCA 
    hsb_pca = pca_hsb.transform(hsb_vecs)
    de_pca = pca_de.transform(de_vecs)
    diff_pca = pca_diff.transform(diff)
    sim_col = sim.diagonal().reshape(-1, 1) 
    features = np.hstack([hsb_pca, de_pca,diff_pca])
    return features,pca_hsb, pca_de,pca_diff

def train_eval(X_train, y_train, X_val, y_val, model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1] 
    elif hasattr(model, "decision_function"): 
        y_score = model.decision_function(X_val)
    else:
        y_score = y_pred  # fallback
        
    metrics = evaluate(y_val, y_score)
    metrics["model"] = name
    return metrics

def evaluate_similarity_only(similarities, labels, name="CosineSim"):
    def find_best_f1_threshold(y_true, y_score):
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        print(f"{thresholds[best_idx]:.4f}")
        return thresholds[best_idx]

    best_thr = find_best_f1_threshold(labels, similarities)
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
        "y_score": similarities
    }

    print(f"\n=== {name} baseline (cosine similarity only) ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:15}: {v:.4f}")
    return metrics

def final_test(model, X_test, y_test, threshold, name=""):
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
    else:
        score = model.predict(X_test)
        threshold = 0.5  # fallback

    y_pred = (score >= threshold).astype(int)
    metrics = {
       "best_threshold": threshold,
       "accuracy"      : accuracy_score(y_test, y_pred),
       "precision"     : precision_score(y_test, y_pred, zero_division=0),
       "recall"        : recall_score(y_test, y_pred, zero_division=0),
       "f1"            : f1_score(y_test, y_pred, zero_division=0),
       "roc_auc"       : roc_auc_score(y_test, y_pred),
       "pr_auc"        : average_precision_score(y_test, y_pred),
       "y_true": y_test,
       "y_score": y_pred,
    }
    metrics["model"] = name
    return metrics


def plot_similarity(similarities, y_true):
    similarities = np.array(similarities)
    y_true = np.array(y_true)
    min_idx = np.argmin(similarities)
    sim_pos = similarities[y_true == 1]
    sim_neg = similarities[y_true == 0]

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.hist(sim_pos, bins=50, alpha=0.6, label="Parallel (label=1)", density=True)
    plt.hist(sim_neg, bins=50, alpha=0.6, label="Non-Parallel (label=0)", density=True)


    plt.title("Similarity Score Distribution by Label")
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train(X_train, y_train, X_val, y_val, sim_val):
    models = {
    "Baseline": None,
    "LGBM"   : LGBMClassifier(n_estimators=400,learning_rate=0.1, random_state=42,reg_alpha=1.0, reg_lambda=1.0),
    "SVM":    make_pipeline(StandardScaler(), LinearSVC(max_iter=1000, dual=True, random_state=42)),
    "XGB":    XGBClassifier(n_estimators=400, learning_rate = 0.1, use_label_encoder=False, eval_metric="logloss", random_state=42,reg_alpha=1.0, reg_lambda=1.0),
    "MLP":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42))
    }
    fitted_models = []
    metrics_list = []
    best_thresholds = {}
    for name, mdl in models.items():
        print(f"\n===== {name} =====")
        if mdl is None:
            sim_val=np.diagonal(sim_val)  # 提取对角线元素作为相似度特征
            metrics = evaluate_similarity_only(sim_val, y_val, name="Baseline")
        else:
            metrics = train_eval(X_train, y_train, X_val, y_val, mdl, name)
        best_thresholds[name] = metrics['best_threshold']
        metrics_list.append(metrics)
        fitted_models.append((name, mdl))

    metrics_df = plot_metrics_from_list(metrics_list)

    cleaned = []
    for m in metrics_list:
        filtered = {
            k: v for k, v in m.items()
            if k in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "best_threshold"]
        }
        cleaned.append(filtered)

    df = pd.DataFrame(cleaned).set_index("model")
    print(df.round(4))
    return fitted_models, best_thresholds, metrics_df

def test(fitted_models, X_test, y_test, best_thresholds):
    final_metrics = []
    for name, model in fitted_models:
        if model is None:
            continue  
        threshold = best_thresholds[name]
        metrics = final_test(model, X_test, y_test, threshold, name)
        final_metrics.append(metrics)

    final_df = pd.DataFrame(final_metrics).set_index("model")
    final_df = final_df[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    print("\n=== Final Test Results ===")
    print(final_df.round(4))
    return final_df

def main():
    model = "glot500"  # or "labse" "glot500"
    base_path = f"./CBIE/cs-de"
    path= f"./CBIE/hsb-de"
    _, hsb_all = utils.load_vec_file(f"{base_path}/{model}.cs-de.train.cs.vec")
    _, de_all  = utils.load_vec_file(f"{base_path}/{model}.cs-de.train.de.vec")
    _, y_all   = utils.load_label_file(f"{base_path}/{model}.cs-de.train.labels")

    _, hsb_test = utils.load_vec_file(f"{path}/{model}.hsb-de.test.hsb.vec")
    _, de_test  = utils.load_vec_file(f"{path}/{model}.hsb-de.test.de.vec")
    _, y_test   = utils.load_label_file(f"{path}/{model}.hsb-de.test.labels")

    # 拆分 train/val
    hsb_train, hsb_val, de_train, de_val, y_train, y_val = train_test_split(hsb_all, de_all, y_all, test_size=0.2, random_state=42)
    
    sim_train = cosine_similarity(hsb_train, de_train)
    sim_val = cosine_similarity(hsb_val, de_val)     
    sim_test = cosine_similarity(hsb_test, de_test)

    X_tr, pca_h, pca_d, pca_diff = build_features(hsb_train, de_train, sim_train, fit_pca=True,n_components=0.95)
    X_val, _, _,_  = build_features(hsb_val, de_val, sim_val, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d,pca_diff=pca_diff)
    X_test, _, _,_ = build_features(hsb_test, de_test, sim_test, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d,pca_diff=pca_diff)

    fitted_models, best_thresholds, metrics_df = train(X_tr, y_train, X_val, y_val, sim_val)    
    final_df = test(fitted_models, X_test, y_test, best_thresholds)
    

if __name__ == '__main__':
    main() 