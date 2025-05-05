import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve
)
from sentence_transformers import SentenceTransformer, models

# log cwd
print("cwd =", os.getcwd())

# load data with correct relative paths
train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv", sep=";")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["comment"].tolist()
test_labels = test_df["isHate"].tolist()
test_labels_bin = np.array(test_labels).astype(int)

# use HateBERT to embed
word_model = models.Transformer("GroNLP/hateBERT")
pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
embedder = SentenceTransformer(modules=[word_model, pooling_model])

print("generating embeddings...")
X_train = embedder.encode(train_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
X_test = embedder.encode(test_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

# PCA for compression
print("applying PCA...")
pca = PCA(n_components=50)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# fit RF
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train_scaled, train_labels)

# evaluate
probs = clf.predict_proba(X_test_scaled)[:, 1]
preds = (probs >= 0.5).astype(int)
print(classification_report(test_labels_bin, preds))

# save confusion matrix
os.makedirs("images", exist_ok=True)
cm = confusion_matrix(test_labels_bin, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Hate", "Hate"],
            yticklabels=["Not Hate", "Hate"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("images/rf_hate_bert_confusion_matrix.jpg")
plt.close()

# save roc curve
fpr, tpr, _ = roc_curve(test_labels_bin, probs)
auc = roc_auc_score(test_labels_bin, probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/rf_hate_bert_roc_curve.jpg")
plt.close()
