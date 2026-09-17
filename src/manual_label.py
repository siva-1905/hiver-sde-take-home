import pandas as pd
import os

INPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\balanced_labeling.csv"

OUTPUT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\human_labeled.csv"

labels = {
    "1": "fare_payment",
    "2": "cancellation",
    "3": "driver_issue",
    "4": "lost_item",
    "5": "account_access",
    "6": "safety_fraud",
    "7": "technical_issue",
    "8": "delivery_issue",
    "9": "general_support",
}

df = pd.read_csv(INPUT_PATH, dtype=str)

# Make sure the intent column can store text labels
df["intent"] = df["intent"].fillna("").astype(str)

# Resume previous labeling if the file already exists
if os.path.exists(OUTPUT_PATH):
    df = pd.read_csv(OUTPUT_PATH)
    print("Resuming previous labeling session.")

print("\nUber Intent Labeling Tool")
print("=" * 60)

print("\nLabels:")
for number, label in labels.items():
    print(f"{number} = {label}")

print("\nCommands:")
print("s = skip")
print("q = quit and save")

for index in range(len(df)):

    # Skip already labeled rows
    if pd.notna(df.loc[index, "intent"]) and str(df.loc[index, "intent"]).strip():
        continue

    print("\n" + "=" * 80)

    print(f"Example {index + 1} / {len(df)}")

    print("\nCUSTOMER:")
    print(df.loc[index, "customer_message"])

    print("\nUBER HISTORICAL RESPONSE:")
    print(df.loc[index, "uber_response"])

    print("\nSuggested intent:")
    print(df.loc[index, "suggested_intent"])

    while True:

        choice = input(
            "\nEnter label (1-9), s=skip, q=quit: "
        ).strip().lower()

        if choice in labels:

            df.loc[index, "intent"] = labels[choice]

            print(
                f"Saved: {labels[choice]}"
            )

            break

        elif choice == "s":

            df.loc[index, "intent"] = "skip"

            print("Skipped.")

            break

        elif choice == "q":

            df.to_csv(
                OUTPUT_PATH,
                index=False
            )

            print("\nProgress saved.")
            print(OUTPUT_PATH)

            raise SystemExit

        else:
            print("Invalid choice. Enter 1-9, s, or q.")


# Save final results
df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("LABELING COMPLETE")
print("=" * 60)

print("\nLabel distribution:")
print(df["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)