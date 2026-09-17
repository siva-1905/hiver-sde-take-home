import pandas as pd
from collections import Counter

DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"

print("Loading cleaned Uber data...")

df = pd.read_csv(DATA_PATH)

# Combine all customer messages into one text column
texts = df["customer_message"].astype(str).str.lower()

# Keywords that can help us inspect common support themes
keyword_groups = {
    "payment_or_fare": [
        "charge", "charged", "fare", "payment", "bill",
        "price", "cost", "receipt", "credit"
    ],

    "refund": [
        "refund", "money back", "credit my account",
        "reimburse", "reimbursement"
    ],

    "cancellation": [
        "cancel", "cancelled", "canceled",
        "cancellation"
    ],

    "driver": [
        "driver", "driving", "ride", "car"
    ],

    "lost_item": [
        "lost", "wallet", "phone", "left my",
        "forgot", "belongings"
    ],

    "account_login": [
        "login", "log in", "logged in", "password",
        "verification code", "verify", "2-factor",
        "two factor", "account"
    ],

    "safety_fraud": [
        "threat", "threatened", "unsafe", "safety",
        "fraud", "fraudulent", "stolen", "scam",
        "using my account"
    ],

    "technical_issue": [
        "app", "website", "error", "not working",
        "doesn't work", "cannot", "can't", "unable"
    ],

    "delivery_food": [
        "ubereats", "food", "order", "delivery",
        "restaurant", "meal", "eats"
    ],

    "support_followup": [
        "no response", "no reply", "still waiting",
        "haven't heard", "didn't respond",
        "no one", "follow up", "already did"
    ]
}

print("\nKeyword-based theme counts:")
print("=" * 50)

for category, keywords in keyword_groups.items():

    count = 0

    for text in texts:
        if any(keyword in text for keyword in keywords):
            count += 1

    percentage = (count / len(df)) * 100

    print(f"{category:20s}: {count:6d} ({percentage:.2f}%)")


print("\n" + "=" * 50)
print("MOST COMMON WORDS")
print("=" * 50)

all_words = []

for text in texts:
    words = text.split()
    all_words.extend(words)

word_counts = Counter(all_words)

# Remove very common Twitter/Uber words
stop_words = {
    "the", "and", "to", "a", "i", "of", "is", "in",
    "for", "on", "my", "it", "you", "this", "that",
    "with", "me", "uber", "please", "us", "we",
    "@uber_support"
}

filtered_words = [
    (word, count)
    for word, count in word_counts.items()
    if word not in stop_words and len(word) > 2
]

filtered_words = sorted(
    filtered_words,
    key=lambda x: x[1],
    reverse=True
)

for word, count in filtered_words[:50]:
    print(f"{word:25s} {count}")