import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

data_dir = Path("data")

model_files = {
    "ridge": ("ridge_probs.npy", "ridge_labels.npy"),
    "logistic": ("logistic_probs.npy", "logistic_labels.npy"),
    "lasso": ("lasso_probs.npy", "lasso_labels.npy"),
    "random_forest": ("random_forest_probs.npy", "random_forest_labels.npy"),
    "knn": ("knn_probs.npy", "knn_labels.npy"),
    "naive_bayes": ("naive_bayes_probs.npy", "naive_bayes_labels.npy"),
    "xgboost": ("xgboost_probs.npy", "xgboost_labels.npy")
}

plt.figure(figsize=(10, 7))
for name, (probs_file, labels_file) in model_files.items():
    try:
        probs = np.load(data_dir / probs_file)
        labels = np.load(data_dir / labels_file)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    except Exception as e:
        print(f"⚠️  skipping {name}: {e}")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves by Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/compare_rocs.png")
plt.show()
