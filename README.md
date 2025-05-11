1. Import Required Libraries

import random
from sklearn.ensemble import IsolationForest
from transformers import pipeline

random: Used if you want to add random behavior (not currently used).

IsolationForest: Scikit-learn’s unsupervised anomaly detection model.

pipeline: Hugging Face’s interface to use pre-trained NLP models easily.



---

2. Sample Transaction Data

transactions = [
    {"amount": 120.5, "location": "New York", "time": 14},
    {"amount": 20.0, "location": "New York", "time": 9},
    {"amount": 3050.0, "location": "Houston", "time": 2},
    {"amount": 150.0, "location": "Chicago", "time": 12},
    {"amount": 10.0, "location": "New York", "time": 10},
    {"amount": 5000.0, "location": "Los Angeles", "time": 1},
]

List of mock transactions; each dictionary contains:

amount: transaction amount

location: where the transaction occurred

time: 24-hour format




---

3. Define Trusted Locations

trusted_users = ["New York", "Chicago"]

These locations are assumed to be safe and common for the user.

Used in calculating the trust score.



---

4. Extract Features for Modeling

def extract_features(tx):
    return [tx["amount"], tx["time"]]

Returns a simplified feature list (amount + time) for each transaction.

These features are passed into the anomaly detection model.



---

5. Prepare Data for Anomaly Detection

X = [extract_features(tx) for tx in transactions]

Applies extract_features() to each transaction to build the dataset.



---

6. Train the Isolation Forest Model

model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)

Anomaly detection model is created with 30% contamination (assumes ~30% are outliers).

random_state=42 makes results reproducible.

fit(X) trains the model on the transaction data.



---

7. Load the NLP Model

nlp_model = pipeline('text-generation', model='distilgpt2')

Loads Hugging Face’s distilgpt2 model.

Used to generate natural language explanations based on a prompt.



---

8. Trust-Based Scoring Heuristic

def trust_score(transaction):
    score = 100
    if transaction["location"] not in trusted_users:
        score -= 30
    if transaction["time"] < 6 or transaction["time"] > 22:
        score -= 20
    if transaction["amount"] > 1000:
        score -= 30
    return max(score, 0)

Initializes trust at 100.

Penalizes the transaction if:

Location is untrusted (-30)

Time is during odd hours (-20)

Amount is very high (-30)


Returns a score between 0–100.



---

9. Generate LLM-Based Explanation

def generate_explanation(transaction, score):
    prompt = (
        f"The transaction of ${transaction['amount']} at {transaction['time']}h "
        f"in {transaction['location']} was flagged. Trust score: {score}/100. "
        f"Explain why it may be suspicious in simple terms:\n"
    )
    output = nlp_model(prompt, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]

Creates a prompt for the LLM with transaction details.

Uses Hugging Face pipeline to generate 1 explanation of max 100 tokens.

Returns the explanation string.



---

10. Evaluate and Explain Each Transaction

for i, tx in enumerate(transactions):
    score = trust_score(tx)
    prediction = model.predict([extract_features(tx)])
    is_anomaly = prediction[0] == -1

Loops through each transaction.

Calculates trust score and anomaly prediction.

model.predict() returns -1 for anomaly, 1 for normal.



---

11. Display Results

print(f"\n--- Transaction #{i+1} ---")
    print(f"Details: {tx}")
    if is_anomaly:
        print(f"Flagged as suspicious.")
        print(f"Trust Score: {score}/100")
        explanation = generate_explanation(tx, score)
        print("Explanation:")
        print(explanation)
    else:
        print("Transaction appears normal.")

For each transaction:

If suspicious: show trust score and generate an NLP explanation.

Otherwise: print that it’s normal.

--- 

Summary:

This code:

Detects fraud using statistical modeling (Isolation Forest).

Scores user trust using a simple heuristic.

Explains flagged alerts using a real LLM (GPT-2 via Hugging Face).
