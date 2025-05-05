import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ridge = np.load("data/ridge_probs.npy")
logistic = np.load("data/logistic_probs.npy")
lasso = np.load("data/lasso_probs.npy")
rf = np.load("data/random_forest_probs.npy")

X_meta = np.vstack([ridge, logistic, lasso, rf]).T

y_true = np.load("data/y_test.npy")

meta_learner = LogisticRegression()
meta_learner.fit(X_meta, y_true)

meta_probs = meta_learner.predict_proba(X_meta)[:, 1]
meta_preds = (meta_probs > 0.5).astype(int)

print(f"AUC: {roc_auc_score(y_true, meta_probs):.3f}")
print(f"Accuracy: {accuracy_score(y_true, meta_preds):.3f}")
print(f"F1: {f1_score(y_true, meta_preds):.3f}")


fpr, tpr, _ = roc_curve(y_true, meta_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Stacked Model (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacked Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/stacked_model_roc.png")
plt.show()
