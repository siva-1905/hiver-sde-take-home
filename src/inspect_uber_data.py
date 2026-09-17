import pandas as pd

DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_conversations.csv"
print("Loading Uber conversation data...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate customer messages:")
print(df["customer_message"].duplicated().sum())

print("\nResponse length statistics:")
print(df["customer_message"].str.len().describe())

print("\nUber response length statistics:")
print(df["uber_response"].str.len().describe())

print("\n" + "=" * 80)
print("RANDOM CUSTOMER MESSAGES")
print("=" * 80)

sample = df.sample(30, random_state=42)

for i, (_, row) in enumerate(sample.iterrows(), 1):

    print(f"\n{i}. CUSTOMER:")
    print(row["customer_message"])

    print("\n   UBER:")
    print(row["uber_response"])

    print("\n" + "-" * 80)