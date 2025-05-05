import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier 

class HateSpeechXGBClassifier:
    def __init__(
        self,
        threshold: float = 0.75,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        bert_model_name: str = "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        max_length: int = 128,
        batch_size: int = 32
    ):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.encoder = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric="logloss"
        )

    def embed(self, texts: pd.Series) -> np.ndarray:
        """
        Generate BERT embeddings for a series of texts using the [CLS] token representation or pooled output.
        """
        self.encoder.eval()
        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts.iloc[start : start + self.batch_size].tolist()
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.encoder(**encoded)
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def train(self, X: pd.Series, y: pd.Series) -> None:
        X_embed = self.embed(X)
        self.model.fit(X_embed, y)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        X_embed = self.embed(X)
        proba = self.model.predict_proba(X_embed)[:, 1]
        return pd.Series(proba, index=X.index)

    def predict(self, X: pd.Series) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> None:
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Hate", "Hate"],
            yticklabels=["Not Hate", "Hate"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        
class KNNHateSpeechClassifier(BaseEstimator, ClassifierMixin):
    """
    KNN-based hate speech classifier using either SentenceTransformer or HateBERT.

    Args:
        threshold (float): Classification threshold.
        model_name (str): Name of Sentence-BERT model.
        n_neighbors (int): K for KNN.
        precomputed (bool): If True, skips embedding.
        use_hatebert (bool): If True, use HateBERT instead of SentenceTransformer.
    """

    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2',
                 n_neighbors=5, precomputed=False, use_hatebert=False):
        self.threshold = threshold
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.precomputed = precomputed
        self.use_hatebert = use_hatebert

        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        if use_hatebert:
            word_model = models.Transformer("GroNLP/hateBERT")
            pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
            self.embedder = SentenceTransformer(modules=[word_model, pooling_model])
        else:
            self.embedder = SentenceTransformer(model_name)

    def fit(self, X, y):
        self.knn.set_params(n_neighbors=self.n_neighbors)
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        self.knn.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        return self.knn.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))
    
    def get_embeddings(self, texts, scale=True, batch_size=32):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return self.scaler.fit_transform(embeddings) if scale else embeddings
    

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Hate", "Hate"],
                    yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def show_misclassifications(self, X, y_true, y_pred):
        X = pd.Series(X)
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        mismatches = X[y_true != y_pred]
        print("\nFALSE POSITIVES:")
        print(mismatches[(y_true == 0) & (y_pred == 1)].head(5).to_string(index=False))
        print("\nFALSE NEGATIVES:")
        print(mismatches[(y_true == 1) & (y_pred == 0)].head(5).to_string(index=False))

    def plot_roc_curve(self, X, y_true):
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - KNN Hate Speech Classifier")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class LogisticHateBERT(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression hate speech classifier using SentenceTransformer or HateBERT.

    Args:
        threshold (float): Classification threshold.
        model_name (str): Name of Sentence-BERT model.
        precomputed (bool): If True, input X is already embedded.
        use_hatebert (bool): If True, use HateBERT instead of SentenceTransformer.
        penalty (str or None): LogisticRegression penalty ('l2', 'l1', 'elasticnet', or None).
    """

    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2',
                 precomputed=False, use_hatebert=False,
                 penalty=None, C=1.0, solver='lbfgs', random_state=42):
        self.threshold = threshold
        self.model_name = model_name
        self.precomputed = precomputed
        self.use_hatebert = use_hatebert
        self.penalty = None if penalty == 'none' else penalty
        self.C = C
        self.solver = solver
        self.random_state = random_state

        self.embedder = None  # Delayed instantiation
        self.scaler = StandardScaler()
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=1000,
            random_state=self.random_state
        )

    def _load_embedder(self):
        if self.embedder is None:
            if self.use_hatebert:
                word_model = models.Transformer("GroNLP/hateBERT")
                pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
                self.embedder = SentenceTransformer(modules=[word_model, pooling_model])
            else:
                self.embedder = SentenceTransformer(self.model_name)

    def fit(self, X, y):
        if not self.precomputed:
            self._load_embedder()
            X = self.get_embeddings(X)
        X_scaled = self.scaler.fit_transform(X)
        self.model_.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if not self.precomputed:
            self._load_embedder()
            X = self.get_embeddings(X)
        X_scaled = self.scaler.transform(X)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, batch_size=32):
        self._load_embedder()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        return self.embedder.encode(
            texts, batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True
        )

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Hate", "Hate"],
                    yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpg')
        plt.close()


    def plot_roc_curve(self, X, y_true, save_path=None):
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Logistic Hate Speech Classifier")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpg')
        plt.close()