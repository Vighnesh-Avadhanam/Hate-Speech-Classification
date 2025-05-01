import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class HateSpeechClassifier:
    """Pipeline-based logistic model to classify hate speech in text."""

    def __init__(self, threshold: float = 0.75):
        """
        Initializes the hate speech classifier.

        Args:
            threshold (float): Probability threshold to classify as hate speech.
        """
        self.threshold = threshold
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def train(self, X: pd.Series, y: pd.Series) -> None:
        """
        Train the logistic model using the input text and labels.

        Args:
            X (pd.Series): Input phrases/text.
            y (pd.Series): Labels (1 for hate speech, 0 otherwise).
        """
        self.pipeline.fit(X, y)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        """
        Predict probabilities of being hate speech.

        Args:
            X (pd.Series): Input phrases/text.

        Returns:
            pd.Series: Probabilities of being hate speech.
        """
        proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index)

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Predict hate speech using the custom probability threshold.

        Args:
            X (pd.Series): Input phrases/text.

        Returns:
            pd.Series: Predicted labels (1 = hate speech, 0 = not).
        """
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> None:
        """
        Print classification report for model evaluation.

        Args:
            X_test (pd.Series): Test text data.
            y_test (pd.Series): Ground truth labels.
        """
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))


def main():
    """Example execution flow."""
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    model = HateSpeechClassifier(threshold=0.75)

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text"],
        train_df["label"],
        test_size=0.2,
        random_state=42
    )

    model.train(X_train, y_train)
    model.evaluate(X_val, y_val)

    test_probs = model.predict_proba(test_df["text"])
    test_preds = model.predict(test_df["text"])

    test_df["probability"] = test_probs
    test_df["prediction"] = test_preds
    test_df.to_csv("hate_speech_predictions.csv", index=False)


if __name__ == "__main__":
    main()