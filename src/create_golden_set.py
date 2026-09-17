import pandas as pd

INPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\human_labeled.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"

print("Loading human-labeled data...")

df = pd.read_csv(INPUT_PATH)

# Keep only actual intent labels
valid_intents = [
    "fare_payment",
    "cancellation",
    "driver_issue",
    "lost_item",
    "account_access",
    "safety_fraud",
    "technical_issue",
    "delivery_issue",
    "general_support"
]

golden = df[df["intent"].isin(valid_intents)].copy()

# Remove duplicate customer messages
golden = golden.drop_duplicates(subset=["customer_message"])

# Save golden set
golden.to_csv(OUTPUT_PATH, index=False)

print("\nGolden set created successfully!")
print("Total examples:", len(golden))

print("\nIntent distribution:")
print(golden["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)