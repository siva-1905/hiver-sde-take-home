import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"

print("Loading golden set...")

df = pd.read_csv(GOLDEN_PATH)

# The majority class in the golden set
majority_class = df["intent"].value_counts().idxmax()

print("\nMajority class:", majority_class)
print("Number of examples:", len(df))

# Predict the same class for every example
y_true = df["intent"]
y_pred = [majority_class] * len(df)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print("MAJORITY CLASSIFIER RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)