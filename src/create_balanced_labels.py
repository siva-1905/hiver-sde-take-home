import pandas as pd

INPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"

OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\balanced_labeling.csv"


print("Loading cleaned Uber data...")

df = pd.read_csv(INPUT_PATH)

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

# Convert text to lowercase
df["text_lower"] = df["customer_message"].astype(str).str.lower()


# Candidate keywords for each intent
intent_keywords = {

    "fare_payment": [
        "charge", "charged", "fare", "payment",
        "bill", "price", "cost", "receipt",
        "credit", "fee"
    ],

    "cancellation": [
        "cancel", "cancelled", "canceled",
        "cancellation"
    ],

    "driver_issue": [
        "driver", "rude", "behavior",
        "driving", "car", "driver did",
        "driver was"
    ],

    "lost_item": [
        "lost", "wallet", "phone",
        "left my", "forgot", "belongings",
        "lost item"
    ],

    "account_access": [
        "login", "log in", "logged in",
        "password", "verification code",
        "verify", "2-factor", "two factor",
        "account access"
    ],

    "safety_fraud": [
        "threat", "threatened", "unsafe",
        "safety", "fraud", "fraudulent",
        "scam", "stolen", "hacked",
        "using my account"
    ],

    "technical_issue": [
        "app", "website", "error",
        "not working", "doesn't work",
        "can't", "cannot", "unable"
    ],

    "delivery_issue": [
        "ubereats", "food", "order",
        "delivery", "restaurant",
        "meal", "eats", "deliver"
    ]
}


# Create candidate rows
candidates = []

for intent, keywords in intent_keywords.items():

    mask = df["text_lower"].apply(
        lambda text: any(keyword in text for keyword in keywords)
    )

    subset = df[mask].copy()

    print(
        f"{intent:20s}: {len(subset)} candidates"
    )

    # Take up to 250 candidates for manual verification
    n = min(250, len(subset))

    if n > 0:
        subset = subset.sample(
            n=n,
            random_state=42
        )

        subset["suggested_intent"] = intent

        candidates.append(subset)


# Combine candidates
balanced = pd.concat(
    candidates,
    ignore_index=True
)


# Remove duplicate customer messages
balanced = balanced.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)


# Create empty final label
balanced["intent"] = ""


# Keep useful columns
balanced = balanced[
    [
        "customer_tweet_id",
        "customer_message",
        "uber_response",
        "created_at",
        "suggested_intent",
        "intent"
    ]
]


balanced.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("BALANCED LABELING FILE CREATED")
print("=" * 70)

print("\nRows:", len(balanced))

print("\nSuggested intent distribution:")
print(
    balanced["suggested_intent"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)

print(
    "\nIMPORTANT: suggested_intent is only a candidate label."
)

print(
    "You must manually verify the intent column before training."
)