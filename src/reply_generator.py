import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ReplyGenerator:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env"
            )

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        customer_message,
        intent,
        historical_results
    ):

        historical_examples = ""

        for i, result in enumerate(
            historical_results,
            start=1
        ):
            historical_examples += f"""
EXAMPLE {i}

Customer:
{result["customer_message"]}

Uber response:
{result["uber_response"]}

Similarity:
{result["similarity"]:.3f}
"""

        prompt = f"""
You are an AI customer-support reply drafting assistant
for Uber.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL UBER SUPPORT EXAMPLES:
{historical_examples}

Your task is to draft a concise customer-facing response.

GROUNDING RULES:

1. The historical examples are the primary source of
   evidence.

2. Use them to understand how Uber handled similar
   requests.

3. Do NOT copy a historical response word-for-word.

4. Do NOT invent Uber policies, app navigation steps,
   refund rules, eligibility requirements, guarantees,
   deadlines, or procedures.

5. Do NOT invent URLs.

6. Only mention a specific action or procedure when it is
   supported by the historical examples.

7. If the historical examples do not provide enough
   information to answer the customer's question safely,
   ask the customer to contact Uber Support or provide
   additional details.

8. Never expose historical customer names, Twitter handles,
   phone numbers, email addresses, account IDs, or other
   personal information.

9. Do not mention the dataset, classifier, retrieval,
   similarity, or AI.

10. Keep the response short and professional.

Return ONLY the customer-facing response.
"""

        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text.strip()