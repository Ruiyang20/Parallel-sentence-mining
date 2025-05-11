from sentence_embedding import to_xlmr_sentence_embeddings
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


model_name = 'xlmr' #or 'glot500m'
df = pd.read_csv("HSB-DE_dataset.tsv",sep='\t')
labels = df['label'].tolist()
hsb, de, sim = to_xlmr_sentence_embeddings(df, model_name)  

hsb_tr, hsb_val, de_tr, de_val, sim_tr, sim_val, y_tr, y_val = train_test_split( hsb, de, sim, labels, test_size=0.2, stratify=labels, random_state=42)

X_tr, pca_h, pca_d = build_features(hsb_tr, de_tr, sim_tr, n_components=20)
X_val, _, _        = build_features(hsb_val, de_val, sim_val, fit_pca=False,
                                    pca_hsb=pca_h, pca_de=pca_d)

models = {
    "LogReg" : LogisticRegression(max_iter=1000, random_state=42),
    "RF"     : RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "LGBM"   : LGBMClassifier(n_estimators=500,learning_rate=0.05, random_state=42),
    "SVM":    make_pipeline(StandardScaler(), LinearSVC(max_iter=1000, dual=True, random_state=42)),
    "XGB":    XGBClassifier(n_estimators=500, learning_rate = 0.05, use_label_encoder=False, eval_metric="logloss", random_state=42),
    "MLP":    make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=400, random_state=42))
}

fitted_models = []
metrics_list = []
for name, mdl in models.items():
    print(f"\n===== {name} =====")
    metrics = train_eval(X_tr, y_tr, X_val, y_val, mdl, name)
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