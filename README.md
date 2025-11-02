# 🧠 GPTFX: Mental Health Detection & Explanation App

**GPTFX** (GPT for eXplainability) is an AI-powered framework that detects **Thwarted Belongingness** and **Perceived Burdensomeness** in user-written text — two key psychological indicators related to mental health.  
The app also generates **human-like explanations** for each prediction using a **FLAN-T5** model.

Built with:
- 🧩 Sentence-Transformers (`all-mpnet-base-v2`) for free embeddings  
- ⚙️ SVM models for classification  
- 💬 FLAN-T5 for explanation generation  
- 🌐 Streamlit for the interactive user interface

---

## 🚀 Features

✅ Detects **Belongingness** and **Burdensomeness** emotions  
✅ Generates short, natural **explanations** for predictions  
✅ Runs completely **locally** — no OpenAI API required  
✅ Built with **free Hugging Face models**  
✅ Easy to deploy on **Streamlit Cloud** or **Hugging Face Spaces**

---

## 🧰 Installation Guide

### 1 Clone the Repository
```bash
git clone https://github.com/nathsujal/Psyche
cd Psyche
```

### 2 Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate     # for macOS/Linux
.venv\bin\activate            # for Windows
```

### 3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4 Running the Streamlit App
```bash
streamlit run app.py
```