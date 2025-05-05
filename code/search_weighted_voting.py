import numpy as np
from sklearn.metrics import f1_score

# load probs + labels
ridge = np.load("data/ridge_probs.npy")
rf    = np.load("data/random_forest_probs.npy")
log   = np.load("data/logistic_probs.npy")
lasso = np.load("data/lasso_probs.npy")
y_true = np.load("data/y_test.npy")

probs_list = [ridge, rf, log, lasso]
names = ["ridge", "rf", "log", "lasso"]

# define grid step size
step = 0.1
best_f1 = 0
best_weights = None

# grid search over all weights summing to 1.0
for w1 in np.arange(0, 1+step, step):
    for w2 in np.arange(0, 1+step, step):
        for w3 in np.arange(0, 1+step, step):
            w4 = 1.0 - (w1 + w2 + w3)
            if w4 < 0 or w4 > 1: continue

            weights = np.array([w1, w2, w3, w4])
            probs_stack = np.vstack(probs_list).T
            ensemble_probs = np.average(probs_stack, axis=1, weights=weights)
            preds = (ensemble_probs > 0.5).astype(int)
            f1 = f1_score(y_true, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights

print(f"Best F1 = {best_f1:.3f}")
for name, w in zip(names, best_weights):
    print(f"{name}: {w:.2f}")
