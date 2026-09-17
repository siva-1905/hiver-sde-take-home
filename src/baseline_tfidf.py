import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\training_data.csv"
GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"


print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading golden set...")
golden_df = pd.read_csv(GOLDEN_PATH)


# ---------------------------------------------------------
# Prepare training data
# ---------------------------------------------------------

X_train = train_df["customer_message"].astype(str)
y_train = train_df["intent"].astype(str)

# Golden set is NEVER used for training
X_test = golden_df["customer_message"].astype(str)
y_test = golden_df["intent"].astype(str)


print("\nTraining examples:", len(X_train))
print("Golden examples:", len(X_test))


# ---------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining TF-IDF + Logistic Regression...")

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("\nEvaluating on golden set...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)