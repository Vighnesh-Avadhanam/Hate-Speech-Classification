import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.makedirs("figures", exist_ok=True)

model_paths = {
    "ridge": ("data/ridge_probs.npy", "data/y_test.npy"),
    "logistic": ("data/logistic_probs.npy", "data/y_test.npy"),
    "lasso": ("data/lasso_probs.npy", "data/y_test.npy"),
    "random_forest": ("data/random_forest_probs.npy", "data/random_forest_labels.npy"),
    "knn": ("data/knn_probs.npy", "data/knn_labels.npy"),
    "naive_bayes": ("data/naive_bayes_probs.npy", "data/naive_bayes_labels.npy"),
    "xgboost": ("data/xgboost_probs.npy", "data/xgboost_labels.npy"),
}

plt.figure(figsize=(10, 7))

for model, (prob_path, label_path) in model_paths.items():
    try:
        probs = np.load(prob_path)
        labels = np.load(label_path)

        if probs.dtype.kind in {"i", "u"}:
            print(f"skipping {model}: predicted probs are integer like")
            continue

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        print(f"{model} AUC: {roc_auc:.3f}")
        plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.3f})", linewidth=2)

    except Exception as e:
        print(f"skipping {model}: {e}")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves by Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/compare_rocs.png")
plt.show()