import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

os.makedirs("../figures", exist_ok=True)

ridge = np.load("data/ridge_probs.npy")
rf = np.load("data/random_forest_probs.npy")
log = np.load("data/logistic_probs.npy")
lasso = np.load("data/lasso_probs.npy")

prob_matrix = np.vstack([ridge, rf, log, lasso]).T

ensemble_probs = prob_matrix.mean(axis=1)

y_true = np.load("data/y_test.npy")

preds = (ensemble_probs > 0.5).astype(int)
print(f"AUC: {roc_auc_score(y_true, ensemble_probs):.3f}")
print(f"Accuracy: {accuracy_score(y_true, preds):.3f}")
print(f"F1: {f1_score(y_true, preds):.3f}")

fpr, tpr, _ = roc_curve(y_true, ensemble_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Soft Voting (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Soft Voting Ensemble")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/soft_voting_roc.png")
plt.show()

precision, recall, _ = precision_recall_curve(y_true, ensemble_probs)
ap = average_precision_score(y_true, ensemble_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Soft Voting Ensemble")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/soft_voting_pr.png")
plt.show()
