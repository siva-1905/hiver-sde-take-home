import pandas as pd
import re

CLEAN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"
GOLDEN_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\golden_set.csv"
OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\training_data.csv"

print("Loading data...")

df = pd.read_csv(CLEAN_PATH)
golden = pd.read_csv(GOLDEN_PATH)

# ---------------------------------------------------------
# Remove golden-set messages from training data
# ---------------------------------------------------------

golden_messages = set(
    golden["customer_message"]
    .astype(str)
    .str.strip()
)

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

df = df[~df["customer_message"].isin(golden_messages)].copy()

print("Messages available for training:", len(df))


# ---------------------------------------------------------
# Intent keyword rules
# ---------------------------------------------------------

patterns = {

    "safety_fraud": [
        r"\bfraud\b",
        r"\bscam\b",
        r"\bscammed\b",
        r"\bunsafe\b",
        r"\bdanger\b",
        r"\bthreat\b",
        r"\bthreaten\b",
        r"\bstolen\b",
        r"\bunauthorized\b",
        r"\bcharge.*not.*mine\b",
        r"\bdidn't.*authorize\b",
        r"\bdid not.*authorize\b",
        r"\bhacked\b"
    ],

    "lost_item": [
        r"\blost\b",
        r"\bmissing\b.*\b(phone|wallet|bag|item|luggage|keys)\b",
        r"\bforgot\b",
        r"\bleft\b.*\b(phone|wallet|bag|item|luggage|keys)\b",
        r"\bfind.*lost\b"
    ],

    "account_access": [
        r"\blogin\b",
        r"\blog in\b",
        r"\bsign in\b",
        r"\bpassword\b",
        r"\bverification\b",
        r"\bverify\b",
        r"\bverification code\b",
        r"\b2fa\b",
        r"\btwo.factor\b",
        r"\bcan't access.*account\b",
        r"\bcannot access.*account\b",
        r"\baccount.*access\b"
    ],

    "cancellation": [
        r"\bcancel\b",
        r"\bcancelled\b",
        r"\bcanceled\b",
        r"\bcancellation\b",
        r"\bcancel.*fee\b",
        r"\bcancellation.*fee\b"
    ],

    "delivery_issue": [
        r"\buber eats\b",
        r"\bubereats\b",
        r"\bfood\b",
        r"\border\b",
        r"\bdelivery\b",
        r"\brestaurant\b",
        r"\bmeal\b",
        r"\btip.*driver\b",
        r"\bdelivery.*driver\b"
    ],

    "technical_issue": [
        r"\bapp\b.*\b(not working|doesn't work|does not work|error|crash)\b",
        r"\bapp.*crash\b",
        r"\bwebsite\b",
        r"\btechnical\b",
        r"\berror\b",
        r"\bbug\b",
        r"\bglitch\b",
        r"\bnot working\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b"
    ],

    "fare_payment": [
        r"\bcharge\b",
        r"\bcharged\b",
        r"\bpayment\b",
        r"\bpay\b",
        r"\bpaid\b",
        r"\bfare\b",
        r"\bprice\b",
        r"\bcost\b",
        r"\bfee\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bcard\b",
        r"\breceipt\b",
        r"\binvoice\b",
        r"\brefund\b",
        r"\bmoney\b",
        r"\bexpensive\b",
        r"\bovercharg"
    ],

    "driver_issue": [
        r"\bdriver\b",
        r"\bdriving\b",
        r"\brider\b",
        r"\bdriver.*behavior\b",
        r"\bdriver.*rating\b",
        r"\bdriver.*feedback\b",
        r"\bspeeding\b",
        r"\bdriving.*complaint\b"
    ]
}


# ---------------------------------------------------------
# Classification function
# ---------------------------------------------------------

def classify(text):

    text = text.lower()

    scores = {}

    for intent, keywords in patterns.items():

        score = 0

        for pattern in keywords:
            if re.search(pattern, text):
                score += 1

        scores[intent] = score

    # Priority order for overlapping intents
    priority = [
        "safety_fraud",
        "lost_item",
        "account_access",
        "cancellation",
        "delivery_issue",
        "technical_issue",
        "fare_payment",
        "driver_issue"
    ]

    max_score = max(scores.values())

    # No matching keywords
    if max_score == 0:
        return "general_support"

    # If multiple intents have the same score,
    # use our predefined priority.
    candidates = [
        intent for intent, score in scores.items()
        if score == max_score
    ]

    for intent in priority:
        if intent in candidates:
            return intent


# ---------------------------------------------------------
# Apply weak labels
# ---------------------------------------------------------

print("Creating weak labels...")

df["intent"] = df["customer_message"].apply(classify)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nTraining data created successfully!")
print("Total training examples:", len(df))

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)