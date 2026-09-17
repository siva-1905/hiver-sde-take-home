import pandas as pd

INPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"

OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\intent_labeling.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

print("Unique customer messages:", len(df))

# Take a reproducible random sample
sample_size = min(2000, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
).copy()

# Add empty intent column
sample["intent"] = ""

# Keep only columns needed for labeling
labeling_df = sample[
    [
        "customer_tweet_id",
        "customer_message",
        "uber_response",
        "created_at",
        "intent"
    ]
]

labeling_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nLabeling dataset created:")
print(OUTPUT_PATH)

print("\nRows:", len(labeling_df))

print("\nIntent column is empty and ready for labeling.")