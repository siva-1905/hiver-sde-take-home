import os
import sys
import json
import pandas as pd

# Allow imports from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.agent import UberSupportAgent


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GOLDEN_PATH = os.path.join(
    BASE_DIR, "data", "golden_set.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "agent_evaluation.csv"
)


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

print("Loading golden evaluation set...")

golden_df = pd.read_csv(GOLDEN_PATH)

print(f"Golden examples: {len(golden_df)}")


# ---------------------------------------------------------
# Initialize agent
# ---------------------------------------------------------

print("\nInitializing Uber Support Agent...")

agent = UberSupportAgent()


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

results = []

for i, row in golden_df.iterrows():

    customer_message = str(row["customer_message"])
    true_intent = str(row["intent"])

    # -----------------------------
    # Intent classification
    # -----------------------------

    predicted_intent, confidence = agent.classifier.predict(
        customer_message
    )

    # -----------------------------
    # Historical retrieval
    # -----------------------------

    historical_results = agent.retriever.retrieve(
        customer_message,
        top_k=3
    )

    if historical_results:
        retrieval_score = float(
            historical_results[0]["similarity"]
        )
    else:
        retrieval_score = 0.0

    # -----------------------------
    # Escalation decision
    # -----------------------------

    decision_result = agent.decide_escalation(
        intent=predicted_intent,
        confidence=confidence,
        retrieval_score=retrieval_score,
        customer_message=customer_message
    )

    # Extract values from returned dictionary
    decision = decision_result["decision"]
    reason = decision_result["reason"]

    # -----------------------------
    # Historical evidence
    # -----------------------------

    historical_evidence = json.dumps(
        historical_results,
        ensure_ascii=False
    )

    # -----------------------------
    # Correct / incorrect
    # -----------------------------

    correct = int(
        predicted_intent == true_intent
    )

    results.append({
        "customer_message": customer_message,
        "true_intent": true_intent,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "retrieval_score": retrieval_score,
        "decision": decision,
        "reason": reason,
        "historical_evidence": historical_evidence,
        "correct": correct
    })

    # Progress
    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(golden_df)}")


# ---------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------

evaluation_df = pd.DataFrame(results)

evaluation_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = evaluation_df["correct"].mean()

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"Total examples: {len(evaluation_df)}")
print(f"Intent accuracy: {accuracy:.4f}")

# ---------------------------------------------------------
# Decision distribution
# ---------------------------------------------------------

print("\nDecision distribution:")

print(
    evaluation_df["decision"]
    .value_counts()
)

# ---------------------------------------------------------
# Decision by intent
# ---------------------------------------------------------

print("\nDecision by intent:")

print(
    pd.crosstab(
        evaluation_df["true_intent"],
        evaluation_df["decision"]
    )
)

# ---------------------------------------------------------
# Average retrieval score by decision
# ---------------------------------------------------------

print("\nAverage retrieval score by decision:")

print(
    evaluation_df
    .groupby("decision")["retrieval_score"]
    .mean()
    .round(4)
)

# ---------------------------------------------------------
# Average confidence by decision
# ---------------------------------------------------------

print("\nAverage confidence by decision:")

print(
    evaluation_df
    .groupby("decision")["confidence"]
    .mean()
    .round(4)
)

# ---------------------------------------------------------
# Retrieval statistics
# ---------------------------------------------------------

print("\nRetrieval score statistics:")

print(
    evaluation_df["retrieval_score"]
    .describe()
    .round(4)
)

# ---------------------------------------------------------
# Confidence statistics
# ---------------------------------------------------------

print("\nConfidence statistics:")

print(
    evaluation_df["confidence"]
    .describe()
    .round(4)
)

# ---------------------------------------------------------
# Save path
# ---------------------------------------------------------

print("\nSaved to:")
print(OUTPUT_PATH)