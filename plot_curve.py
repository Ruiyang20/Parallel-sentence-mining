import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve
)

def plot_metrics_from_list(metrics_list):
    df = pd.DataFrame(metrics_list).set_index("model")
    fig_cols = 3 
    fig = plt.figure(figsize=(6 * fig_cols, 5))

    # ① ROC
    ax1 = fig.add_subplot(1, fig_cols, 1)
    for m in metrics_list:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_score"])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{m['model']} (AUC={roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    # ② PR
    ax2 = fig.add_subplot(1, fig_cols, 2)
    for m in metrics_list:
        prec, rec, _ = precision_recall_curve(m["y_true"], m["y_score"])
        ax2.plot(rec, prec, label=f"{m['model']} (AP={m['pr_auc']:.2f})")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    # ③ 柱状对比
    ax3 = fig.add_subplot(1, fig_cols, fig_cols)
    df[["accuracy", "precision", "recall", "f1"]].plot.bar(ax=ax3, rot=0)
    ax3.set_title("Metrics Comparison")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right")
    ax3.grid(axis="y", alpha=0.3)

#    ax4 = fig.add_subplot(1, 4, 4, polar=True)
#    _plot_radar_ax(metrics_df, ax4)

    fig.tight_layout()
    plt.show()

    return df.round(4)

def _plot_radar_ax(df: pd.DataFrame, ax):
    labels   = df.columns.tolist()
    num_vars = len(labels)
    angles   = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("Radar (Acc/F1/AUC)")

    for name, row in df.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=name, lw=2)
        ax.fill(angles, values, alpha=0.15)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
