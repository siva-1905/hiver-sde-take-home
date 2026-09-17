import pandas as pd

DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\raw\twcs.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_conversations.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Make tweet IDs consistent
df["tweet_id"] = df["tweet_id"].astype(str)

# Create lookup using tweet_id as index
tweet_lookup = df.set_index("tweet_id")

# Get all Uber Support tweets
uber = df[df["author_id"] == "Uber_Support"].copy()

conversations = []

for _, uber_tweet in uber.iterrows():

    # Uber tweet is responding to a customer tweet
    parent_id = uber_tweet["in_response_to_tweet_id"]

    if pd.isna(parent_id):
        continue

    parent_id = str(int(parent_id))

    if parent_id not in tweet_lookup.index:
        continue

    customer_tweet = tweet_lookup.loc[parent_id]

    # Make sure parent is actually a customer message
    if customer_tweet["inbound"] != True:
        continue

    conversations.append({
        "customer_tweet_id": parent_id,
        "customer_message": customer_tweet["text"],
        "uber_tweet_id": uber_tweet["tweet_id"],
        "uber_response": uber_tweet["text"],
        "created_at": customer_tweet["created_at"]
    })

conversations_df = pd.DataFrame(conversations)

print("\nUber customer-support pairs:", len(conversations_df))

print("\nSample conversations:\n")

for _, row in conversations_df.head(10).iterrows():

    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nUBER:")
    print(row["uber_response"])

    print("\n" + "-" * 80)

# Save the processed dataset
conversations_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:")
print(OUTPUT_PATH)