import pandas as pd

INPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_conversations.csv"

OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"

print("Loading Uber conversations...")

df = pd.read_csv(INPUT_PATH)

print("Original rows:", len(df))

# Remove rows with missing messages
df = df.dropna(subset=["customer_message", "uber_response"])

# Remove exact duplicate customer + response pairs
df = df.drop_duplicates(
    subset=["customer_message", "uber_response"]
)

# Clean text
df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["uber_response"] = (
    df["uber_response"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove empty messages
df = df[df["customer_message"].str.len() > 0]

print("Rows after cleaning:", len(df))

print("\nRemoved rows:", 56160 - len(df))

print("\nSample cleaned conversations:")

for _, row in df.sample(10, random_state=42).iterrows():

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nUBER:")
    print(row["uber_response"])

    print("\n" + "-" * 80)

# Save
df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved cleaned dataset to:")
print(OUTPUT_PATH)