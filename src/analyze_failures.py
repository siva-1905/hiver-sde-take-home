import pandas as pd


GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"


print("=" * 70)
print("CLASSIFIER FAILURE ANALYSIS")
print("=" * 70)

df = pd.read_csv(GOLDEN_PATH)

# Remove invalid rows
df = df[df["intent"].notna()].copy()

print("\nGolden examples:", len(df))

# Train classifier
from intent_classifier import IntentClassifier

classifier = IntentClassifier()

predictions = []
confidences = []

print("\nGenerating predictions...")

for i, text in enumerate(
    df["customer_message"].astype(str)
):

    prediction, confidence = classifier.predict(text)

    predictions.append(prediction)
    confidences.append(confidence)

    if (i + 1) % 25 == 0:
        print(
            f"Processed {i + 1}/{len(df)}"
        )


df["predicted_intent"] = predictions
df["confidence"] = confidences

df["correct"] = (
    df["intent"] ==
    df["predicted_intent"]
)


# --------------------------------------------------
# WRONG PREDICTIONS
# --------------------------------------------------

wrong = df[
    df["correct"] == False
].copy()

print("\n" + "=" * 70)
print("WRONG PREDICTIONS")
print("=" * 70)

print(
    "\nTotal wrong:",
    len(wrong)
)

print(
    "Error rate:",
    round(len(wrong) / len(df), 4)
)


# Most common confusion pairs
print("\nMost common confusion pairs:")

confusions = (
    wrong
    .groupby(
        ["intent", "predicted_intent"]
    )
    .size()
    .sort_values(ascending=False)
)

print(confusions.head(15))


# --------------------------------------------------
# LOW CONFIDENCE
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOW-CONFIDENCE EXAMPLES")
print("=" * 70)

low_confidence = df.sort_values(
    "confidence"
).head(15)

for _, row in low_confidence.iterrows():

    print("\n" + "-" * 60)

    print(
        "Customer:",
        row["customer_message"]
    )

    print(
        "True intent:",
        row["intent"]
    )

    print(
        "Predicted:",
        row["predicted_intent"]
    )

    print(
        "Confidence:",
        round(row["confidence"], 4)
    )


# --------------------------------------------------
# WRONG EXAMPLES
# --------------------------------------------------

print("\n" + "=" * 70)
print("REPRESENTATIVE WRONG EXAMPLES")
print("=" * 70)

for _, row in wrong.head(20).iterrows():

    print("\n" + "-" * 60)

    print(
        "Customer:",
        row["customer_message"]
    )

    print(
        "True intent:",
        row["intent"]
    )

    print(
        "Predicted:",
        row["predicted_intent"]
    )

    print(
        "Confidence:",
        round(row["confidence"], 4)
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\failure_analysis.csv"

wrong.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved failure analysis to:")
print(OUTPUT_PATH)