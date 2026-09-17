# Decision Log

## 1. Selected Uber_Support as the brand
Uber_Support had a sufficiently large number of conversations in the dataset, giving enough historical examples for both intent classification and response retrieval.

## 2. Reconstructed customer-support pairs
Customer messages were paired with historical Uber responses using the response and parent tweet identifiers. This converts the raw tweet dataset into usable support conversations.

## 3. Removed duplicate and empty conversations
Duplicate customer-response pairs and empty customer messages were removed to reduce noise.

## 4. Used a small intent taxonomy
The project uses nine intents:
- fare_payment
- cancellation
- driver_issue
- lost_item
- account_access
- safety_fraud
- technical_issue
- delivery_issue
- general_support

The taxonomy was designed from recurring support themes in the Uber conversations.

## 5. Added general_support
Some customer messages did not clearly belong to a specific operational category. Rather than forcing them into an unrelated intent, general_support was added.

## 6. Created a human-labelled golden set
A 215-example human-labelled set was created for evaluation. It was kept separate from training data.

## 7. Excluded golden examples from training
Golden-set messages were removed from the training data to avoid evaluation leakage.

## 8. Used weak supervision to bootstrap training data
Keyword-based rules were used to create a large labelled training set because manually labelling tens of thousands of tweets was impractical for the assignment timeframe.

## 9. Used word and character TF-IDF
Word features capture common phrases while character features help with noisy Twitter language, spelling variations and abbreviations.

## 10. Used Logistic Regression
Logistic Regression provides a simple, fast and interpretable classifier suitable for a lightweight support-agent prototype.

## 11. Used TF-IDF retrieval for historical responses
TF-IDF retrieval was selected instead of an embedding/vector database approach because it is simple, fast, explainable and sufficient for demonstrating the core retrieval workflow.

## 12. Retrieved multiple historical examples
The agent retrieves the top three historical conversations instead of relying on a single example. This reduces dependence on one potentially noisy match.

## 13. Used an LLM only for response drafting
The classifier determines the intent and the retriever provides historical evidence. The LLM is used to turn that evidence into a concise customer-facing response.

## 14. Added conservative escalation rules
The agent escalates when classifier confidence is low, historical retrieval evidence is weak, or the message contains safety/fraud-related signals.

## 15. Added safety/fraud keyword override
Safety and fraud cases are escalated even if the classifier predicts another intent. This provides a defense-in-depth mechanism for higher-risk conversations.