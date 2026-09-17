import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\training_data.csv"
GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"


class IntentClassifier:

    def __init__(self):
        print("Loading training data...")

        train_df = pd.read_csv(TRAIN_PATH)

        self.X_train = train_df["customer_message"].astype(str)
        self.y_train = train_df["intent"].astype(str)

        print("Training examples:", len(self.X_train))

        # ---------------------------------------------------------
        # Word-level TF-IDF
        # ---------------------------------------------------------

        print("\nCreating word-level TF-IDF...")

        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=50000
        )

        self.X_train_word = self.word_vectorizer.fit_transform(
            self.X_train
        )

        # ---------------------------------------------------------
        # Character-level TF-IDF
        # ---------------------------------------------------------

        print("Creating character-level TF-IDF...")

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        self.X_train_char = self.char_vectorizer.fit_transform(
            self.X_train
        )

        # ---------------------------------------------------------
        # Combine features
        # ---------------------------------------------------------

        print("Combining features...")

        X_train_combined = hstack([
            self.X_train_word,
            self.X_train_char
        ])

        print(
            "Training feature shape:",
            X_train_combined.shape
        )

        # ---------------------------------------------------------
        # Logistic Regression
        # ---------------------------------------------------------

        print("\nTraining improved classifier...")

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.model.fit(
            X_train_combined,
            self.y_train
        )

        print("Classifier ready!")

    # =========================================================
    # PREDICT ONE CUSTOMER MESSAGE
    # =========================================================

    def predict(self, text):

        # Word features
        word_vector = self.word_vectorizer.transform([text])

        # Character features
        char_vector = self.char_vectorizer.transform([text])

        # Combine
        combined_vector = hstack([
            word_vector,
            char_vector
        ])

        # Prediction
        prediction = self.model.predict(
            combined_vector
        )[0]

        # Confidence
        probabilities = self.model.predict_proba(
            combined_vector
        )[0]

        confidence = float(
            probabilities.max()
        )

        return prediction, confidence


# =============================================================
# EVALUATION
# =============================================================

if __name__ == "__main__":

    print("\nLoading golden set...")

    golden_df = pd.read_csv(GOLDEN_PATH)

    X_test = golden_df[
        "customer_message"
    ].astype(str)

    y_test = golden_df[
        "intent"
    ].astype(str)

    print("Golden examples:", len(X_test))

    # Create classifier
    classifier = IntentClassifier()

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\nEvaluating on golden set...")

    y_pred = []

    for text in X_test:
        prediction, confidence = classifier.predict(text)
        y_pred.append(prediction)

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print("IMPROVED INTENT CLASSIFIER RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Confidence examples
    # ---------------------------------------------------------

    print("\nConfidence examples:")

    for i in range(min(10, len(X_test))):

        prediction, confidence = classifier.predict(
            X_test.iloc[i]
        )

        print("-" * 60)

        print(
            "Message:",
            X_test.iloc[i]
        )

        print(
            "Predicted:",
            prediction
        )

        print(
            "Confidence:",
            round(confidence, 4)
        )