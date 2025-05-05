import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

model_files = {
    "ridge": ("ridge_probs.npy", "ridge_labels.npy"),
    "random_forest": ("random_forest_probs.npy", "random_forest_labels.npy"),
    "logistic": ("logistic_probs.npy", "logistic_labels.npy"),
    "lasso": ("lasso_probs.npy", "lasso_labels.npy"),
    "knn": ("knn_probs.npy", "knn_labels.npy"),
    "weighted_voting": ("weighted_voting_probs.npy", "y_test.npy"),
}

plt.figure(figsize=(10, 7))

for model, (probs_file, labels_file) in model_files.items():
    probs_path = os.path.join("data", probs_file)
    labels_path = os.path.join("data", labels_file)

    if not os.path.exists(probs_path) or not os.path.exists(labels_path):
        print(f"⚠️ skipping {model}: missing .npy files")
        continue

    probs = np.load(probs_path)
    labels = np.load(labels_path)

    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc:.3f})", linewidth=2)
    except ValueError as e:
        print(f"❌ error in {model}: {e}")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves by Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/roc_comparison.png")
plt.show()
