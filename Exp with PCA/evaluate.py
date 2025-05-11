import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_best_f1_threshold(y_true, y_score):
  precision, recall, thresholds = precision_recall_curve(y_true, y_score)
  f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
  best_idx = np.argmax(f1_scores)
  return thresholds[best_idx]


def evaluate(y_true, y_score):
  best_thr = find_best_f1_threshold(y_true, y_score)
  print(f"threshold = {best_thr:.4f}")
  y_pred = (y_score >= best_thr).astype(int)
  metrics = {
    "best_threshold": best_thr,
    "accuracy"      : accuracy_score(y_true, y_pred),
    "precision"     : precision_score(y_true, y_pred, zero_division=0),
    "recall"        : recall_score(y_true, y_pred, zero_division=0),
    "f1"            : f1_score(y_true, y_pred, zero_division=0),
    "roc_auc"       : roc_auc_score(y_true, y_score),
    "pr_auc"        : average_precision_score(y_true, y_score),
    "y_true": y_true,
    "y_score": y_score,
    }
  print("\n=== Validation metrics (best‑F1 threshold) ===")
  for k, v in metrics.items():
     if isinstance(v, (int, float)):
       print(f"{k:15}: {v:.4f}") 
       
  return metrics

