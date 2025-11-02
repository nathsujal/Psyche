# --------------------------------------------------
# GPTFX-Style Mental Health Detector & Explainer
# Streamlit version using Hugging Face + SVM models
# --------------------------------------------------

import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# Page Config
st.set_page_config(
    page_title="GPTFX Mental Health Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 GPTFX-style Mental Health Detector & Explainer")
st.markdown("Detect *Thwarted Belongingness* and *Perceived Burdensomeness* in text and get AI-generated explanations.")
st.divider()

# --------------------------------------------------
# Load Models (cached for speed)
@st.cache_resource
def load_models():
    st.info("Loading models... please wait ⏳")
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    svm_belong = joblib.load('svm_belong_mpnet.pkl')
    svm_burden = joblib.load('svm_burden_mpnet.pkl')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t5_model = t5_model.to(device)
    return embedder, svm_belong, svm_burden, tokenizer, t5_model, device

embedder, svm_belong, svm_burden, tokenizer, t5_model, device = load_models()
st.success("✅ Models loaded successfully!")

# --------------------------------------------------
# Helper Functions
def predict_labels(text):
    emb = embedder.encode([text], convert_to_numpy=True)
    belong_pred = svm_belong.predict(emb)[0]
    burden_pred = svm_burden.predict(emb)[0]
    return belong_pred, burden_pred

def generate_explanation(text, belong_pred, burden_pred):
    if belong_pred == 1 and burden_pred == 1:
        desc = "feels both isolated and like a burden to others"
    elif belong_pred == 1:
        desc = "feels isolated and disconnected from others"
    elif burden_pred == 1:
        desc = "feels like a burden or causing trouble for others"
    else:
        desc = "shows no strong signs of belongingness or burden"

    prompt = f"Text: {text}\nExplain why the person {desc}."
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------------------------------------
# User Input
user_text = st.text_area("✍️ Enter text to analyze:", height=200, placeholder="Type or paste a Reddit-like post here...")

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text before running analysis.")
    else:
        with st.spinner("Analyzing text... please wait"):
            belong_pred, burden_pred = predict_labels(user_text)
            explanation = generate_explanation(user_text, belong_pred, burden_pred)

        st.divider()
        st.subheader("🧠 Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Thwarted Belongingness", "YES" if belong_pred else "NO", delta=None)
        col2.metric("Perceived Burdensomeness", "YES" if burden_pred else "NO", delta=None)

        st.markdown("### 💬 Explanation")
        st.write(explanation)

        st.divider()
        st.caption("⚙️ Model: all-mpnet-base-v2 embeddings + SVM + FLAN-T5 for explanations.")

