import matplotlib.pyplot as plt
from sklearn.metrics import auc

roc_data = {
    "ridge": {
        "fpr": [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0],
        "tpr": [0.0, 0.13, 0.25, 0.48, 0.68, 0.83, 0.93, 0.98, 1.0],
        "auc": 0.789
    },
    "random_forest": {
        "fpr": [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
        "tpr": [0.0, 0.10, 0.30, 0.50, 0.70, 0.82, 0.92, 1.0],
        "auc": 0.721
    },
    "logistic": {
        "fpr": [0.0, 0.01, 0.06, 0.13, 0.23, 0.42, 0.65, 0.87, 1.0],
        "tpr": [0.0, 0.12, 0.29, 0.51, 0.69, 0.84, 0.92, 0.98, 1.0],
        "auc": 0.787
    },
    "lasso": {
        "fpr": [0.0, 0.01, 0.05, 0.11, 0.24, 0.41, 0.64, 0.86, 1.0],
        "tpr": [0.0, 0.14, 0.28, 0.49, 0.67, 0.85, 0.93, 0.97, 1.0],
        "auc": 0.789
    },
    "knn": {
        "fpr": [0.0, 0.02, 0.03, 0.04, 0.09, 0.11, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0],
        "tpr": [0.0, 0.10, 0.20, 0.30, 0.50, 0.60, 0.68, 0.74, 0.84, 0.90, 0.96, 1.0],
        "auc": 0.702
    },
    "naive_bayes": {
        "fpr": [0.0, 0.02, 0.08, 0.15, 0.27, 0.39, 0.53, 0.70, 0.88, 1.0],
        "tpr": [0.0, 0.09, 0.26, 0.44, 0.60, 0.72, 0.84, 0.92, 0.97, 1.0],
        "auc": 0.684
    },
    "xgboost": {
        "fpr": [0.0, 0.01, 0.04, 0.09, 0.17, 0.35, 0.58, 0.80, 1.0],
        "tpr": [0.0, 0.11, 0.32, 0.54, 0.71, 0.84, 0.93, 0.98, 1.0],
        "auc": 0.744
    }
}

plt.figure(figsize=(10, 7))
for model_name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=f"{model_name} (AUC = {data['auc']:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Across Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_comparison.png")
plt.show()
