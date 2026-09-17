# Uber Support AI Agent

An end-to-end AI customer-support agent built for the Hiver SDE Intern take-home assignment.

The goal is not only to build a support agent, but to evaluate whether it is trustworthy enough to automate customer-support interactions.

---

## 1. Problem Framing

I selected **Uber Support** from the Customer Support on Twitter dataset.

The agent receives an incoming customer message and performs three tasks:

1. Classifies the message into a support intent.
2. Retrieves historically similar Uber Support conversations and uses them as evidence for drafting a response.
3. Decides whether the message should be automatically handled or escalated to a human, with an explicit reason.

For this project, a good support agent should:

* Correctly identify the customer's issue.
* Use historical support interactions as evidence.
* Avoid inventing policies, refunds, procedures, URLs, or guarantees.
* Produce concise and useful responses.
* Escalate uncertain or sensitive cases rather than confidently giving an unsupported answer.

### What I chose not to build

This project intentionally focuses on the core support-agent pipeline rather than production infrastructure.

I did not build:

* A web UI or mobile application.
* A production database.
* Authentication and user management.
* Real-time integration with Uber.
* A full ticketing system.
* Automated execution of refunds, cancellations, or account actions.
* Full-dataset production-scale serving.

The assignment explicitly allows subsampling because the evaluator will not run the system on the complete dataset.

---

## 2. Dataset

The primary dataset is the **Customer Support on Twitter (TWCS)** dataset from Kaggle (`thoughtvector/customer-support-on-twitter`).

The dataset contains approximately 3 million tweets, multi-turn support threads, and multiple brands.

I selected the Uber Support conversations.

After extracting and cleaning the Uber conversations:

* Uber Support conversations: approximately 56K
* Unique customer messages: approximately 54K
* Training examples: 55,940
* Golden evaluation examples: 215

The golden set was kept separate from the training data.

### Dataset availability

The original TWCS dataset is approximately **493 MB** and is intentionally **not committed to this GitHub repository**.

This avoids unnecessarily storing a large raw dataset in Git while keeping the repository lightweight and reproducible.

To run the complete data-processing pipeline from the raw data, download the TWCS dataset separately and place:

```text
twcs.csv
```

at:

```text
data/raw/twcs.csv
```

The raw dataset is not required to inspect the included golden evaluation set or previously generated evaluation artifacts.

---

## 3. Intent Taxonomy

The intent categories were defined from the Uber Support data rather than imported from an unrelated taxonomy.

The final intents are:

* `fare_payment`
* `cancellation`
* `driver_issue`
* `lost_item`
* `account_access`
* `safety_fraud`
* `technical_issue`
* `delivery_issue`
* `support_followup`

Examples include:

| Intent             | Example                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| `fare_payment`     | Questions about charges, fares, payment methods                                                    |
| `cancellation`     | Cancellation and cancellation-fee issues                                                           |
| `driver_issue`     | Driver behavior or driver-related problems                                                         |
| `lost_item`        | Items lost during an Uber trip                                                                     |
| `account_access`   | Login and account-access problems                                                                  |
| `safety_fraud`     | Fraud, unauthorized activity, or safety concerns                                                   |
| `technical_issue`  | App or technical problems                                                                          |
| `delivery_issue`   | Uber Eats/order/delivery problems                                                                  |
| `support_followup` | Follow-up messages referring to previously requested information or an ongoing support interaction |

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       v
Historical Conversation Retrieval
       |
       v
LLM Reply Generator
       |
       v
Safety + Confidence Checks
       |
       +------> AUTO-HANDLE
       |
       +------> ESCALATE
```

### Intent Classification

The classifier uses both word-level and character-level TF-IDF representations.

The two representations are combined into a 100,000-feature training representation.

This allows the classifier to use:

* Word-level semantic patterns.
* Character-level patterns useful for short/noisy support messages and spelling variations.

### Historical Retrieval

The system builds a TF-IDF index over historical Uber customer-support conversations.

For each new message, the system retrieves the top three similar historical conversations.

These conversations provide evidence for the response generator.

### Reply Generation

An LLM generates the customer-facing reply using:

* The customer's message.
* The predicted intent.
* Similar historical Uber Support conversations.

The generation prompt explicitly instructs the model not to invent policies, procedures, URLs, refunds, eligibility rules, or guarantees.

If historical evidence is insufficient, the model is instructed to direct the customer to Uber Support or request additional information.

### Escalation

The system escalates cases when:

* The message contains explicit high-risk safety/fraud language.
* The predicted intent is `safety_fraud`.
* Intent confidence is below 0.70.
* Historical retrieval evidence is below the configured threshold.

Otherwise, the system can return `AUTO-HANDLE`.

Every escalation includes a reason.

---

## 5. Golden Evaluation Set

I created a separate hand-labelled golden evaluation set containing **215 examples**, satisfying the assignment requirement of 150–250 hand-labelled examples.

The examples were sampled from the processed Uber Support conversations and manually assigned to the intent taxonomy.

The golden set was kept separate from the training examples to evaluate the system on unseen examples.

The evaluation contains examples across the supported intent categories, including cancellation, fare/payment, driver issues, delivery issues, account access, safety/fraud, technical issues, lost-item cases, and support follow-ups.

---

## 6. Evaluation Results

### Intent Classifier

The classifier was evaluated on the 215-example golden set.

| Metric      |     Result |
| ----------- | ---------: |
| Accuracy    | **65.58%** |
| Macro F1    | **41.05%** |
| Weighted F1 | **64.08%** |

The difference between weighted and macro F1 is important because the evaluation set is not perfectly balanced across intents.

The classifier performs substantially better on common intents such as cancellation, fare/payment, and delivery issues than on rare intents such as lost-item and safety/fraud examples.

---

## 7. Results vs Baselines

The final intent classifier was compared against two baselines on the same 215-example golden evaluation set.

### Baseline 1: Majority Classifier

The trivial baseline always predicts the most frequent class.

The majority class was:

```text
fare_payment
```

Results:

| Metric      | Majority Baseline |
| ----------- | ----------------: |
| Accuracy    |            45.58% |
| Macro F1    |                7% |
| Weighted F1 |               29% |

This provides a minimum reference point. Its relatively high accuracy is largely explained by the presence of `fare_payment` as the largest class in the evaluation set.

### Baseline 2: TF-IDF + Logistic Regression

The simple ML baseline uses TF-IDF features with Logistic Regression.

Results:

| Metric      | TF-IDF + Logistic Regression |
| ----------- | ---------------------------: |
| Accuracy    |                       63.26% |
| Macro F1    |                          39% |
| Weighted F1 |                          62% |

### Final Classifier

The final classifier combines word-level and character-level TF-IDF representations.

Results:

| Metric      | Final Classifier |
| ----------- | ---------------: |
| Accuracy    |       **65.58%** |
| Macro F1    |          **41%** |
| Weighted F1 |          **64%** |

### Comparison

| Model                        |   Accuracy | Macro F1 | Weighted F1 |
| ---------------------------- | ---------: | -------: | ----------: |
| Majority baseline            |     45.58% |       7% |         29% |
| TF-IDF + Logistic Regression |     63.26% |      39% |         62% |
| **Final classifier**         | **65.58%** |  **41%** |     **64%** |

The final classifier improves substantially over the trivial majority baseline.

Compared with the simpler TF-IDF + Logistic Regression baseline, the improvement is more modest:

* Accuracy: +2.32 percentage points
* Macro F1: approximately +2 percentage points
* Weighted F1: approximately +2 percentage points

This suggests that the combined word + character representation provides an incremental improvement rather than a dramatic change in classification performance.

The baseline comparison is therefore important: the final model performs better than both reference systems, but the results also show that the remaining classification problem is difficult.

---

## 8. Agent Evaluation

The complete agent was evaluated on the 215-example golden set to examine its routing behavior.

Results:

| Decision    | Examples |
| ----------- | -------: |
| AUTO-HANDLE |      164 |
| ESCALATE    |       51 |

This corresponds to approximately:

* 76.3% AUTO-HANDLE
* 23.7% ESCALATE

The agent does not automatically handle every request. Sensitive and uncertain cases are routed toward human review.

---

## 9. Manual End-to-End Tests

Representative cases were tested manually.

### Cancellation

```text
I was charged a cancellation fee even though I did not cancel my ride.
```

Predicted intent:

```text
cancellation
```

The system identified the cancellation-related intent and evaluated the historical evidence before making its handling decision.

### Technical issue

```text
My Uber app is not working.
```

Predicted intent:

```text
technical_issue
```

Decision:

```text
AUTO-HANDLE
```

### Lost item

```text
I lost my phone in an Uber.
```

Predicted intent:

```text
lost_item
```

Decision:

```text
AUTO-HANDLE
```

### Driver issue

```text
My driver was rude and behaved badly.
```

Predicted intent:

```text
driver_issue
```

The system escalated because the historical retrieval evidence was below the configured threshold.

### Payment

```text
How do I add a credit card to my account?
```

Predicted intent:

```text
fare_payment
```

Decision:

```text
AUTO-HANDLE
```

### Safety / unauthorized activity

```text
Someone used my Uber account without my permission.
```

The system treated the message as a high-risk case and escalated it rather than blindly auto-handling it.

---

## 10. Failure Analysis

The evaluation revealed several important failure modes.

### 1. Rare-intent confusion

Rare categories have substantially fewer examples than common categories.

For example, the golden set contained only a small number of `lost_item` and `safety_fraud` examples.

**Hypothesis:**

The classifier has insufficient representative examples for these intents and therefore tends to map some of them to more common categories.

### 2. Support-follow-up ambiguity

Some customer messages refer to a previous interaction without clearly stating a new issue.

Examples may include messages indicating that requested information has already been submitted or that the customer is waiting for a previous support request to be resolved.

**Hypothesis:**

These messages depend heavily on conversation context, while the classifier primarily receives the current message.

### 3. Driver-related ambiguity

Some driver messages describe several issues simultaneously, such as driver behaviour, fare disputes, or trip problems.

**Hypothesis:**

The intent taxonomy is based on the primary support issue, while real customer messages may contain multiple issues.

### 4. Safety/fraud language is difficult to classify

Safety and unauthorized-account messages are relatively rare but require conservative handling.

The escalation layer therefore provides a second safety mechanism independent of the classifier's exact prediction.

**Hypothesis:**

For high-risk cases, detecting risk language is more important than forcing perfect intent classification.

### 5. Short/noisy customer messages

Twitter support messages are often short, informal, misspelled, or contain incomplete context.

**Hypothesis:**

Character-level TF-IDF helps with noisy language, but some messages still lack enough information for reliable classification.

---

## 11. What Is Misleading About My Headline Number?

The headline classifier accuracy is **65.58%**, but this number should not be interpreted as meaning that the agent successfully resolves 65.58% of customer problems.

There are several reasons.

First, classification accuracy measures whether the predicted intent matches the manually assigned label. It does not directly measure whether the final response is useful.

Second, the golden set contains different numbers of examples across intents. Therefore, overall accuracy can hide poor performance on rare categories.

Third, the agent has additional components after classification:

```text
classification
      ↓
retrieval
      ↓
response generation
      ↓
safety / escalation
```

A classification error does not necessarily mean the final interaction is unsafe, because the escalation layer can still route uncertain or high-risk cases to a human.

Conversely, a correct intent prediction does not guarantee that the generated response is helpful.

Therefore, **65.58% is a classifier metric, not an end-to-end customer-support success rate.**

This distinction is important when evaluating whether the system is trustworthy.

---

## 12. LLM-as-Judge

An LLM-based reply-quality evaluation harness was implemented.

The judge evaluates generated replies on:

* Helpfulness
* Groundedness
* Safety
* Overall quality

Each criterion uses a 1–3 scale.

The judge is instructed to return structured JSON containing the four scores and a short explanation.

The implementation is available in:

```text
src/evaluate_replies.py
```

The reply-generation component itself successfully produced customer-facing responses from historical support evidence.

However, the full automated judge evaluation could not be completed because the OpenAI organization reached its daily API request limit during execution.

Consequently, this repository does **not** report an LLM-judge reply-quality score or a human-vs-LLM agreement statistic that was not actually measured.

The judge rubric and evaluation harness remain implemented in `src/evaluate_replies.py`.

Completing the human-agreement study is included as a next-step evaluation task below.

This limitation is explicitly documented rather than replaced with an unsupported or fabricated result.

---

## 13. What I Would Do With One More Week

With another week, I would focus primarily on evaluation and reliability rather than adding more system complexity.

### 1. Improve the golden set

Increase representation of rare and ambiguous intents and create clearer labeling guidelines.

### 2. Improve intent boundaries

Review the confusion matrix and merge or redefine categories that are difficult to distinguish operationally.

### 3. Evaluate retrieval quality separately

Measure whether the retrieved historical conversations are actually relevant to the customer's issue instead of relying only on similarity scores.

### 4. Complete human-vs-LLM judge validation

Have human reviewers independently score a sample of generated replies and measure agreement with the LLM judge.

### 5. Evaluate end-to-end response quality

Create a small manually reviewed test set covering helpfulness, factual grounding, and escalation correctness.

### 6. Improve uncertainty calibration

Tune confidence thresholds using validation data instead of selecting them heuristically.

### 7. Add better handling for multi-intent messages

Allow a message to contain more than one issue while selecting a primary action for routing.

---

## 14. Decision Log

1. **Selected Uber Support** because it provided a sufficiently large collection of support conversations for building and evaluating a focused agent.

2. **Used the Customer Support on Twitter dataset** because it contains real, noisy customer-support interactions.

3. **Built customer-support conversation pairs** rather than treating every tweet independently so that historical responses could be used as evidence.

4. **Defined intents from the Uber data** rather than importing a generic support taxonomy.

5. **Created a separate 215-example golden set** to avoid evaluating only on training examples.

6. **Used both word and character TF-IDF** to handle normal language as well as noisy/short Twitter messages.

7. **Limited retrieval to the top three historical examples** to provide focused evidence to the response generator.

8. **Used historical responses as the primary grounding source** for generated replies.

9. **Explicitly prohibited invented URLs, policies, refunds, procedures, and guarantees** in the reply-generation prompt.

10. **Added confidence-based escalation** so uncertain classifications are not automatically handled.

11. **Added explicit safety/fraud escalation** because high-risk customer messages require more conservative handling.

12. **Used an AUTO-HANDLE / ESCALATE decision** instead of forcing every request through automatic response generation.

13. **Implemented a trivial majority baseline** to establish a minimum classification reference.

14. **Implemented a simpler TF-IDF baseline** to compare the final classifier against a less complex approach.

15. **Included a misleading-headline section** because classifier accuracy alone does not represent end-to-end support quality.

---

## 15. Reproducibility

### Environment setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Reproduce headline results

The main headline classifier results can be reproduced with:

```bash
python src/evaluate_classifier.py
```

This evaluates the classifier against the 215-example golden set and reports accuracy, macro F1, and weighted F1.

### Reproduce baseline results

Run the trivial baseline:

```bash
python src/baseline_majority.py
```

Run the simple TF-IDF + Logistic Regression baseline:

```bash
python src/baseline_tfidf.py
```

The previously generated baseline results are also available in:

```text
results/baseline_results.txt
```

### Run the complete agent evaluation

```bash
python src/evaluate_agent.py
```

### Run the agent manually

```bash
python src/agent.py
```

### LLM-based reply generation

LLM-based reply generation requires an OpenAI API key stored locally in `.env`:

```text
OPENAI_API_KEY=your_api_key
```

The `.env` file is intentionally excluded from Git.

The core classifier and evaluation pipeline can be inspected independently of the LLM reply-generation component.

### Raw dataset

The original TWCS dataset is intentionally not included in GitHub because it is approximately 493 MB.

If running the full raw-data processing pipeline, place:

```text
twcs.csv
```

at:

```text
data/raw/twcs.csv
```

The repository's raw dataset and local environment files are excluded through `.gitignore`.

---

## 16. Project Structure

```text
hiver-sde/

│
├── README.md
├── REPORT.md
├── DECISION_LOG.md
├── requirements.txt
│
├── data/
│   ├── golden_set.csv
│   ├── agent_evaluation.csv
│   ├── failure_analysis.csv
│   │
│   ├── raw/
│   │   └── twcs.csv              # local only, not committed
│   │
│   └── processed/
│       ├── balanced_labeling.csv
│       ├── human_labeled.csv
│       └── intent_labeling.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── results/
│   └── baseline_results.txt
│
└── src/
    ├── agent.py
    ├── intent_classifier.py
    ├── retrieval.py
    ├── reply_generator.py
    ├── evaluate_classifier.py
    ├── evaluate_agent.py
    ├── evaluate_replies.py
    ├── analyze_failures.py
    ├── baseline_majority.py
    ├── baseline_tfidf.py
    └── data-processing scripts
```

---

## 17. Summary

The project demonstrates a complete customer-support automation pipeline:

```text
Historical Support Data
        ↓
Data Cleaning
        ↓
Intent Classification
        ↓
Historical Retrieval
        ↓
Grounded LLM Reply
        ↓
Safety / Confidence Checks
        ↓
AUTO-HANDLE or ESCALATE
```

The system was evaluated using a hand-labelled 215-example golden set and achieved:

**65.58% intent accuracy**

with:

**41.05% macro F1**

and:

**64.08% weighted F1.**

The final classifier improves over both a trivial majority baseline and a simpler TF-IDF + Logistic Regression baseline.

The evaluation also demonstrates that the agent can distinguish between cases suitable for automatic handling and cases that should be escalated to a human.

The main lesson from the evaluation is that a single headline classification number is insufficient to establish trust. A support agent needs classification evaluation, baseline comparisons, failure analysis, grounding checks, and conservative escalation behavior.

The LLM-as-judge rubric and harness were implemented, but the full judge run and human-agreement measurement could not be completed because of an OpenAI API quota limitation during the evaluation run. This limitation is explicitly reported rather than replaced with an unsupported result.
