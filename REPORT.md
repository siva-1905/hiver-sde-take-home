# Hiver SDE Intern Take-Home Assignment
## AI Customer Support Agent for Uber_Support

## 1. Problem Framing

The goal is to build a lightweight AI support agent that:
1. classifies incoming customer messages into support intents;
2. retrieves historically similar Uber conversations;
3. drafts a response grounded in historical resolutions;
4. decides whether the response can be auto-handled or should be escalated.

The system was evaluated using a human-labelled golden set.

---

## 2. Dataset and Brand Selection

The Customer Support on Twitter dataset contains customer-support conversations between users and brands.

Uber_Support was selected because it provided a sufficiently large set of conversations for building a useful prototype.

The raw dataset was transformed into customer-response pairs using the tweet relationship fields.

After cleaning, 56,156 Uber support pairs remained.

---

## 3. Intent Taxonomy

The final taxonomy contains nine intents:

| Intent | Description |
|---|---|
| fare_payment | Charges, fares, payments, credits and billing |
| cancellation | Trip/order cancellation and cancellation charges |
| driver_issue | Driver behaviour, route, pickup and driver complaints |
| lost_item | Lost belongings and missing items |
| account_access | Account, login and access problems |
| safety_fraud | Fraud, unauthorized activity and safety concerns |
| technical_issue | App, payment-system or technical errors |
| delivery_issue | Uber Eats delivery/order issues |
| general_support | General questions not clearly fitting another intent |

---

## 4. Evaluation Setup

The golden evaluation set contains 215 human-labelled examples.

The golden set was not used for training.

Training data was generated using weak keyword supervision from the remaining Uber conversations.

The classifier uses combined word-level and character-level TF-IDF features with Logistic Regression.

---

## 5. Baselines

### Majority baseline

The majority-class classifier always predicts `fare_payment`.

Accuracy: 45.58%

Macro F1: 0.07

### TF-IDF + Logistic Regression baseline

Accuracy: 63.26%

Macro F1: 0.39

### Improved classifier

The improved classifier combines word and character TF-IDF features and uses balanced Logistic Regression.

Accuracy: 65.58%

Macro F1: 0.41

The improvement is modest, but the combined representation provides better handling of noisy support text.

---

## 6. End-to-End Agent

The pipeline is:

Customer message
        ↓
Intent classifier
        ↓
Historical TF-IDF retrieval
        ↓
Top historical support examples
        ↓
LLM response generation
        ↓
Escalation decision
        ↓
Draft response + decision + reason

The response generator is instructed to use historical examples as evidence and avoid inventing policies, procedures, guarantees or URLs.

---

## 7. Escalation Policy

The agent escalates when:

- the message contains high-risk safety/fraud signals;
- the predicted intent is safety_fraud;
- classifier confidence is below 0.70;
- historical retrieval similarity is below 0.50.

Otherwise the message is marked AUTO-HANDLE.

This conservative design prioritizes avoiding unsupported responses over maximizing automation.

---

## 8. Failure Analysis

### 1. Fare vs cancellation confusion

Cancellation and payment complaints often occur together. The model therefore sometimes predicts fare_payment for cancellation cases.

### 2. Driver vs payment confusion

Driver complaints frequently mention charges, overcharging or fees. These payment-related words can dominate the classifier.

### 3. General support ambiguity

General_support acts as a catch-all category, making its boundary with specific intents difficult to learn.

### 4. Rare intents

Some intents have very few golden examples. Their metrics are therefore unstable and should not be interpreted as reliable estimates of real-world performance.

### 5. Multi-intent messages

Some customers describe several problems in one message. A single-label classifier must choose one category even when multiple categories could reasonably apply.

---

## 9. What Is Misleading About My Headline Number?

The headline accuracy of 65.58% should not be interpreted as a complete measure of production readiness.

The golden set contains only 215 examples and is substantially imbalanced across intents. Rare intents have very small evaluation support.

The classifier also produces highly confident predictions that can still be wrong. Therefore classifier confidence should not be treated as a calibrated probability.

Macro F1 is reported alongside accuracy to make performance on less frequent classes visible.

---

## 10. Reply Quality Evaluation

The generated responses are evaluated using an LLM-as-judge rubric covering:

- Helpfulness
- Groundedness
- Safety
- Overall response quality

The judge receives the customer message, generated response and retrieved historical evidence.

A human-labelled subset is also used to compare human and LLM judgements.

---

## 11. Top Failure Examples

Example:

True intent: technical_issue

Predicted intent: delivery_issue

The customer reported several Uber Eats problems including payment-profile errors and inability to create an order. The classifier assigned delivery_issue with very high confidence.

This demonstrates that confidence does not guarantee correctness when messages contain multiple overlapping problems.

---

## 12. Limitations

- Weakly supervised training labels introduce noise.
- Golden-set size is intentionally small.
- Some intents have very few evaluation examples.
- TF-IDF retrieval can return partially relevant historical conversations.
- Single-label classification does not handle multi-intent messages explicitly.
- LLM-generated responses depend on the quality of retrieved historical evidence.
- Classifier confidence is not calibrated.

---

## 13. What I Would Do With One More Week

1. Manually label more training data, especially rare intents.
2. Improve intent definitions and handle multi-intent messages.
3. Replace or augment TF-IDF retrieval with embeddings.
4. Calibrate classifier confidence.
5. Expand the human evaluation set.
6. Measure automation precision separately from overall intent accuracy.
7. Add stronger safety and escalation policies.
8. Test the system on unseen conversation threads rather than only individual messages.

---

## 14. Conclusion

The prototype demonstrates an end-to-end support-agent workflow combining intent classification, historical retrieval, grounded response generation and escalation.

The main result is not simply the classifier accuracy. The evaluation and failure analysis show where the system works, where it fails, and why those failures occur.