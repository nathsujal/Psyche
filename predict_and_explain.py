import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# Load Models
print("🔹 Loading models...")

# Load sentence-transformer for embeddings
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load trained SVM classifiers
svm_belong = joblib.load('svm_belong_mpnet.pkl')
svm_burden = joblib.load('svm_burden_mpnet.pkl')

# Load explanation generator (FLAN-T5)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
t5_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

device = "cuda" if torch.cuda.is_available() else "cpu"
t5_model = t5_model.to(device)

print("✅ All models loaded successfully!\n")

# --------------------------------------------------
# Define Helper Functions

def generate_embeddings(text):
    """Generate MPNet embeddings for the input text."""
    return embedder.encode([text], convert_to_numpy=True)

def predict_labels(text):
    """Predict belongingness and burdensomeness from input text."""
    emb = generate_embeddings(text)
    belong_pred = svm_belong.predict(emb)[0]
    burden_pred = svm_burden.predict(emb)[0]
    return belong_pred, burden_pred

def generate_explanation(text, belong_pred, burden_pred):
    """Generate human-readable explanation using FLAN-T5."""
    labels = []
    if belong_pred == 1:
        labels.append("feels isolated or lacking belongingness")
    if burden_pred == 1:
        labels.append("feels like a burden to others")
    if not labels:
        labels.append("shows no strong signs of either feeling")

    combined_label = " and ".join(labels)
    prompt = f"Text: {text}\nExplain why the person {combined_label}."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(**inputs, max_new_tokens=200)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation

# --------------------------------------------------
# Interactive Input Loop

print("💬 Mental Health Text Analyzer (Belong/Burden)\n")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("Enter text: ").strip()
    if user_text.lower() == "exit":
        print("👋 Exiting...")
        break

    if not user_text:
        continue

    # Predict
    belong_pred, burden_pred = predict_labels(user_text)

    # Interpret
    belong_label = "YES" if belong_pred == 1 else "NO"
    burden_label = "YES" if burden_pred == 1 else "NO"

    # Generate Explanation
    explanation = generate_explanation(user_text, belong_pred, burden_pred)

    print("\n🧠 Prediction Results:")
    print(f" - Thwarted Belongingness: {belong_label}")
    print(f" - Perceived Burdensomeness: {burden_label}")
    print(f"\n💬 Explanation: {explanation}\n")
    print("-" * 60)