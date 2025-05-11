import random
from sklearn.ensemble import IsolationForest
from transformers import pipeline

# Sample transaction data
transactions = [
    {"amount": 120.5, "location": "New York", "time": 14},
    {"amount": 20.0, "location": "New York", "time": 9},
    {"amount": 3050.0, "location": "Houston", "time": 2},
    {"amount": 150.0, "location": "Chicago", "time": 12},
    {"amount": 10.0, "location": "New York", "time": 10},
    {"amount": 5000.0, "location": "Los Angeles", "time": 1},
]

trusted_users = ["New York", "Chicago"]  # trusted locations (proxy for user behavior)

# Extract features for anomaly detection
def extract_features(tx):
    return [tx["amount"], tx["time"]]

X = [extract_features(tx) for tx in transactions]

# Train Isolation Forest
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)

# NLP Explanation Model from Hugging Face
nlp_model = pipeline('text-generation', model='distilgpt2')

# Score each transaction and generate alerts
def trust_score(transaction):
    # A basic heuristic: more trust if location is known/trusted
    score = 100
    if transaction["location"] not in trusted_users:
        score -= 30
    if transaction["time"] < 6 or transaction["time"] > 22:
        score -= 20
    if transaction["amount"] > 1000:
        score -= 30
    return max(score, 0)

def generate_explanation(transaction, score):
    prompt = (
        f"The transaction of ${transaction['amount']} at {transaction['time']}h "
        f"in {transaction['location']} was flagged. Trust score: {score}/100. "
        f"Explain why it may be suspicious in simple terms:\n"
    )
    output = nlp_model(prompt, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]

# Evaluate each transaction
for i, tx in enumerate(transactions):
    score = trust_score(tx)
    prediction = model.predict([extract_features(tx)])
    is_anomaly = prediction[0] == -1

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
