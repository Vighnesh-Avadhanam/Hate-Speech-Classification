import os
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

data_dir = Path(__file__).resolve().parents[1] / "data"
print("cwd =", os.getcwd())
test_df = pd.read_csv(data_dir / "test_data_clean.csv")
test_df.columns = test_df.columns.str.strip()
print("loaded test_data_clean.csv with", len(test_df), "rows")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_test_texts = test_df["comment"].tolist()

X_test_path = data_dir / "X_test.npy"
if not X_test_path.exists():
    print("regenerating X_test.npy...")
    X_test_embeddings = embedder.encode(X_test_texts, show_progress_bar=True)
    np.save(X_test_path, X_test_embeddings)
else:
    print("found existing X_test.npy")

X_test = np.load(X_test_path)
y_test = test_df["isHate"].astype(int).to_numpy()

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

nb = GaussianNB()
nb.fit(X_test_scaled, y_test)
nb_probs = nb.predict_proba(X_test_scaled)[:, 1]
np.save(data_dir / "naive_bayes_probs.npy", nb_probs)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_test_scaled, y_test)
knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
np.save(data_dir / "knn_probs.npy", knn_probs)

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_test_scaled, y_test)
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
np.save(data_dir / "xgboost_probs.npy", xgb_probs)

np.save(data_dir / "y_test.npy", y_test)

print("saved all model probs and labels to /data")
