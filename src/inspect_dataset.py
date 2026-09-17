import pandas as pd

DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\raw\twcs.csv"

df = pd.read_csv(DATA_PATH)

brand = "Uber_Support"

brand_df = df[df["author_id"] == brand].copy()

print("Brand:", brand)
print("Total Uber tweets:", len(brand_df))

print("\nInbound / Outbound:")
print(brand_df["inbound"].value_counts())

print("\nMissing values:")
print(brand_df.isnull().sum())

print("\nSample Uber tweets:")
print(brand_df[["tweet_id", "inbound", "text"]].head(20).to_string(index=False))