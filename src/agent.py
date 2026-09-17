from intent_classifier import IntentClassifier
from retrieval import HistoricalRetriever
from reply_generator import ReplyGenerator


class UberSupportAgent:

    def __init__(self):
        print("Initializing Uber Support Agent...")

        # --------------------------------------------------
        # Load intent classifier
        # --------------------------------------------------

        self.classifier = IntentClassifier()

        # --------------------------------------------------
        # Load historical conversation retriever
        # --------------------------------------------------

        self.retriever = HistoricalRetriever()
        self.reply_generator = ReplyGenerator()
        print("Agent ready!")

    # ======================================================
    # REPLY GENERATION
    # ======================================================

    def generate_reply(
        self,
        customer_message,
        intent,
        historical_results
    ):
        if not historical_results:
            return (
                "Thanks for reaching out. "
                "Please provide more details so our "
                "support team can help."
            )

        return self.reply_generator.generate(
            customer_message=customer_message,
            intent=intent,
            historical_results=historical_results
        )
    # ======================================================
    # ESCALATION DECISION
    # ======================================================

    def decide_escalation(
        self,
        intent,
        confidence,
        retrieval_score,
        customer_message
    ):
        """
        Decide whether the issue should be handled automatically
        or escalated to a human.
        """

        message_lower = customer_message.lower()

        # --------------------------------------------------
        # 1. Explicit high-risk language
        # --------------------------------------------------

        high_risk_keywords = [
            "fraud",
            "fraudulent",
            "stolen",
            "unauthorized",
            "without my permission",
            "someone used my account",
            "someone accessed my account",
            "hacked",
            "threat",
            "threatened",
            "danger",
            "unsafe",
            "assault",
            "harassment"
        ]

        if any(
            keyword in message_lower
            for keyword in high_risk_keywords
        ):
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Potential safety, fraud, or "
                    "unauthorized-account issue requires "
                    "human review."
                )
            }

        # --------------------------------------------------
        # 2. Safety / fraud intent
        # --------------------------------------------------

        if intent == "safety_fraud":
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Safety or fraud-related issue requires "
                    "human review."
                )
            }

        # --------------------------------------------------
        # 3. Low classifier confidence
        # --------------------------------------------------

        if confidence < 0.70:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Intent classification confidence is "
                    "below the auto-handle threshold."
                )
            }

        # --------------------------------------------------
        # 4. Weak historical retrieval
        # --------------------------------------------------

        if retrieval_score < 0.50:
            return {
                "decision": "ESCALATE",
                "reason": (
                    "Historical evidence is not sufficiently "
                    "similar for safe automatic handling."
                )
            }

        # --------------------------------------------------
        # 5. Otherwise auto-handle
        # --------------------------------------------------

        return {
            "decision": "AUTO-HANDLE",
            "reason": (
                "Intent confidence and historical evidence "
                "are sufficient."
            )
        }

    # ======================================================
    # PROCESS CUSTOMER MESSAGE
    # ======================================================

    def process(self, customer_message):
        """
        Run the complete support-agent pipeline.

        Steps:
        1. Classify intent
        2. Retrieve historical conversations
        3. Generate draft reply
        4. Decide auto-handle vs escalation
        """

        # --------------------------------------------------
        # 1. Intent classification
        # --------------------------------------------------

        intent, confidence = self.classifier.predict(
            customer_message
        )

        # --------------------------------------------------
        # 2. Historical retrieval
        # --------------------------------------------------

        historical_results = self.retriever.retrieve(
            customer_message,
            top_k=3
        )

        if historical_results:
            retrieval_score = historical_results[0]["similarity"]
        else:
            retrieval_score = 0.0

        # --------------------------------------------------
        # 3. Generate draft reply
        # --------------------------------------------------

        draft_reply = "[LLM reply generation skipped during bulk evaluation]"

        # --------------------------------------------------
        # 4. Escalation decision
        # --------------------------------------------------

        escalation = self.decide_escalation(
            intent=intent,
            confidence=confidence,
            retrieval_score=retrieval_score,
            customer_message=customer_message
        )

        # --------------------------------------------------
        # 5. Return complete result
        # --------------------------------------------------

        return {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "retrieval_score": retrieval_score,
            "historical_results": historical_results,
            "draft_reply": draft_reply,
            "decision": escalation["decision"],
            "reason": escalation["reason"]
        }


# ==========================================================
# TEST THE AGENT
# ==========================================================

if __name__ == "__main__":

    agent = UberSupportAgent()

    test_messages = [

        "I was charged a cancellation fee even though I did not cancel my ride.",

        "My Uber app is not working.",

        "I lost my phone in an Uber.",

        "My driver was rude and behaved badly.",

        "How do I add a credit card to my account?",

        "Someone used my Uber account without my permission."
    ]

    for message in test_messages:

        print("\n" + "=" * 90)

        print("CUSTOMER MESSAGE:")
        print(message)

        # --------------------------------------------------
        # Process message
        # --------------------------------------------------

        result = agent.process(message)

        # --------------------------------------------------
        # Intent
        # --------------------------------------------------

        print("\nPREDICTED INTENT:")
        print(result["intent"])

        # --------------------------------------------------
        # Classifier confidence
        # --------------------------------------------------

        print("\nCONFIDENCE:")
        print(round(result["confidence"], 4))

        # --------------------------------------------------
        # Retrieval score
        # --------------------------------------------------

        print("\nRETRIEVAL SCORE:")
        print(round(result["retrieval_score"], 4))

        # --------------------------------------------------
        # Draft reply
        # --------------------------------------------------

        print("\nDRAFT REPLY:")
        print(result["draft_reply"])

        # --------------------------------------------------
        # Decision
        # --------------------------------------------------

        print("\nDECISION:")
        print(result["decision"])

        # --------------------------------------------------
        # Reason
        # --------------------------------------------------

        print("\nREASON:")
        print(result["reason"])

        # --------------------------------------------------
        # Historical evidence
        # --------------------------------------------------

        print("\nHISTORICAL EVIDENCE:")

        for i, historical in enumerate(
            result["historical_results"],
            1
        ):

            print(f"\n--- Match {i} ---")

            print(
                "Similarity:",
                round(
                    historical["similarity"],
                    4
                )
            )

            print(
                "Customer:",
                historical["customer_message"]
            )

            print(
                "Uber:",
                historical["uber_response"]
            )