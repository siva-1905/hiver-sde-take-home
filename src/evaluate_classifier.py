import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from intent_classifier import IntentClassifier


GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"


print("=" * 70)
print("INTENT CLASSIFIER EVALUATION")
print("=" * 70)

golden_df = pd.read_csv(GOLDEN_PATH)

golden_df = golden_df[
    golden_df["intent"].notna()
].copy()

X_test = golden_df["customer_message"].astype(str)
y_test = golden_df["intent"].astype(str)

print("\nGolden examples:", len(golden_df))

classifier = IntentClassifier()

print("\nGenerating predictions...")

y_pred = []

for text in X_test:
    prediction, confidence = classifier.predict(text)
    y_pred.append(prediction)


accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)
weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nAccuracy:   {accuracy:.4f}")
print(f"Macro F1:   {macro_f1:.4f}")
print(f"Weighted F1:{weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)