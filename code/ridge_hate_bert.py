import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

threshold = 0.25
use_hatebert = True
penalty = "l2"
solver = "liblinear"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

print(f"\ncwd = {os.getcwd()}")
print("loading data...")

train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv", sep=";")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].astype(int).to_numpy()
test_texts = test_df["comment"].tolist()
test_labels = test_df["isHate"].astype(int).to_numpy()

print("generating embeddings...")
if use_hatebert:
    word_model = models.Transformer("GroNLP/hateBERT")
    pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
    embedder = SentenceTransformer(modules=[word_model, pooling_model])
else:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

X_train = embedder.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)
X_test = embedder.encode(test_texts, show_progress_bar=True, convert_to_numpy=True)

print("applying PCA...")
pca = PCA(n_components=50)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print("training ridge classifier...")
clf = LogisticRegression(
    penalty=penalty,
    C=1.0,
    solver=solver,
    max_iter=1000,
    random_state=42
)
clf.fit(X_train_reduced, train_labels)

probs = clf.predict_proba(X_test_reduced)[:, 1]
preds = (probs >= threshold).astype(int)

print("\n" + classification_report(test_labels, preds))

cm = confusion_matrix(test_labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Hate", "Hate"],
            yticklabels=["Not Hate", "Hate"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ridge HateBERT")
plt.tight_layout()
plt.savefig(f"{output_dir}/ridge_hatebert_confusion_matrix.jpg")
plt.close()

fpr, tpr, _ = roc_curve(test_labels, probs)
auc = roc_auc_score(test_labels, probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ridge HateBERT")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/ridge_hatebert_roc_curve.jpg")
plt.close()
