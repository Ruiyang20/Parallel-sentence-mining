from sentence_embedding import to_xlmr_sentence_embeddings
from sentence_embedding import to_labse_sentence_embeddings
from plot_curve import plot_metrics_from_list
from evaluate import evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


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

def build_features(hsb_vecs, de_vecs, similarities, fit_pca=True, pca_hsb=None, pca_de=None, n_components=20):
    if fit_pca:
        pca_hsb = PCA(n_components=n_components).fit(hsb_vecs)
        pca_de = PCA(n_components=n_components).fit(de_vecs)

    hsb_pca = pca_hsb.transform(hsb_vecs)
    de_pca = pca_de.transform(de_vecs)
    features = np.hstack([hsb_pca, de_pca, similarities.reshape(-1, 1)])
    return features, pca_hsb, pca_de

def train_eval(X_train, y_train, X_val, y_val, model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1] 
    elif hasattr(model, "decision_function"): 
        y_score = model.decision_function(X_val)
    else:
        y_score = y_pred  
        
    metrics = evaluate(y_val, y_score)
    metrics["model"] = name
    return metrics

def process(input_path,model_name):
    df = pd.read_csv(input_path, sep='\t')
    split = np.load("split_indices.npz")
    df_train = df.loc[split["train"]]
    df_val   = df.loc[split["val"]]
    df_test  = df.loc[split["test"]]
    y_train = df.loc[split["train"], "label"].values
    y_val   = df.loc[split["val"], "label"].values
    y_test  = df.loc[split["test"], "label"].values
    
    df = pd.read_csv("HSB-DE_dataset.tsv",sep='\t')
    labels = df['label'].tolist()
    if model_name in ['xlmr', 'glot500']:
        hsb_train, de_train, sim_train = to_xlmr_sentence_embeddings(df_train,model_name)
        hsb_val, de_val, sim_val = to_xlmr_sentence_embeddings(df_val,model_name)
        hsb_test, de_test, sim_test = to_xlmr_sentence_embeddings(df_test,model_name)
    elif model_name == 'labse':
        hsb_train, de_train, sim_train = to_labse_sentence_embeddings(df_train)
        hsb_val, de_val, sim_val = to_labse_sentence_embeddings(df_val)
        hsb_test, de_test, sim_test = to_labse_sentence_embeddings(df_test)
        
    X_tr, pca_h, pca_d = build_features(hsb_train, de_train, sim_train, fit_pca=True, n_components=20)
    X_val, _, _        = build_features(hsb_val, de_val, sim_val, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d)
    X_test, _, _ = build_features(hsb_test, de_test, sim_test, fit_pca=False, pca_hsb=pca_h, pca_de=pca_d)

    models = {
        "Baseline": None,
        "LogReg" : LogisticRegression(max_iter=1000, random_state=42),
        "RF"     : RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        "LGBM"   : LGBMClassifier(n_estimators=500,learning_rate=0.05, random_state=42),
        "SVM":    make_pipeline(StandardScaler(), LinearSVC(max_iter=1000, dual=True, random_state=42)),
        "XGB":    XGBClassifier(n_estimators=500, learning_rate = 0.05, use_label_encoder=False, eval_metric="logloss", random_state=42),
        "MLP":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=400, random_state=42))
    }
    
    fitted_models = []
    metrics_list = []
    best_thresholds = {}
    for name, mdl in models.items():
        print(f"\n===== {name} =====")
        if mdl is None:
            metrics = evaluate_similarity_only(sim_val, y_val, name="Baseline")
        else:
            metrics = train_eval(X_tr, y_train, X_val, y_val, mdl, name)
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

if __name__ == "__main__":
    process(
        input_path="HSB-DE_dataset.tsv",
        model_name='labse' # 'xlmr' or 'glot500'
        )