import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import matplotlib.pyplot as plt

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Multi-Model Sentiment Dashboard",
    layout="wide",
    page_icon="📊"
)

# ===========================================================
# GLOBAL STYLES (BACKGROUND + ICON BUTTONS)
# ===========================================================
st.markdown("""
<style>
.stApp {
    background-color: #F4F1FF;
}

.nav-btn button {
    height: 110px;
    width: 100%;
    border-radius: 20px;
    font-size: 18px;
    font-weight: bold;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

.nav-btn button:hover {
    background-color: #E8DDFF;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ===========================================================
# HEADER
# ===========================================================
st.markdown("""
<div style="background-color:#8A2BE2;padding:18px;border-radius:14px;margin-bottom:15px;">
    <h1 style="color:yellow;text-align:center;">
        Sentiment Analysis of College Survey
    </h1>
</div>
""", unsafe_allow_html=True)

# ===========================================================
# SESSION STATE
# ===========================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "combined_texts" not in st.session_state:
    st.session_state.combined_texts = None
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "menu" not in st.session_state:
    st.session_state.menu = "Upload File"

# ===========================================================
# MODEL DEFINITION
# ===========================================================
class DistilBERT_BiLSTM_Model(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(out.last_hidden_state)
        return self.fc(lstm_out[:, -1, :])

# ===========================================================
# LOAD MODELS
# ===========================================================
@st.cache_resource
def load_distilbert_lstm():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBERT_BiLSTM_Model()
    model.load_state_dict(torch.load("distilbert_bilstm_model.pth", map_location="cpu"))
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_roberta():
    return (
        AutoModelForSequenceClassification.from_pretrained("saved_roberta_model").eval(),
        AutoTokenizer.from_pretrained("saved_roberta_model")
    )

@st.cache_resource
def load_distilbert_hf():
    return (
        AutoModelForSequenceClassification.from_pretrained("saved_model").eval(),
        AutoTokenizer.from_pretrained("saved_model")
    )

# ===========================================================
# PREDICTION FUNCTION
# ===========================================================
def batch_predict(texts, model, tokenizer, name, batch_size=32):
    label_map = {0: "Unhappy", 1: "Neutral", 2: "Happy"}
    preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")

        with torch.no_grad():
            if name == "DistilBERT+BiLSTM":
                logits = model(enc["input_ids"], enc["attention_mask"])
            else:
                logits = model(**enc).logits

        preds.extend([label_map[i] for i in torch.argmax(logits, 1).tolist()])

    return preds

# ===========================================================
# SIDEBAR (ONLY CONTROLS)
# ===========================================================
st.sidebar.header("⚙ Controls")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["DistilBERT+BiLSTM", "RoBERTa", "DistilBERT (HuggingFace)"]
)

uploaded_file = st.sidebar.file_uploader("Upload Survey CSV", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

df = st.session_state.df

# ===========================================================
# LOAD MODEL ONCE
# ===========================================================
if df is not None and st.session_state.model is None:
    if model_option == "DistilBERT+BiLSTM":
        st.session_state.model, st.session_state.tokenizer = load_distilbert_lstm()
    elif model_option == "RoBERTa":
        st.session_state.model, st.session_state.tokenizer = load_roberta()
    else:
        st.session_state.model, st.session_state.tokenizer = load_distilbert_hf()

# ===========================================================
# ICON NAVIGATION
# ===========================================================
st.markdown("## 📌 Dashboard Navigation")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("📤\nUpload"):
        st.session_state.menu = "Upload File"
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    if st.button("📄\nData"):
        st.session_state.menu = "Data"

with c3:
    if st.button("📊\nCharts"):
        st.session_state.menu = "Sentiment Charts"

with c4:
    if st.button("📝\nReport"):
        st.session_state.menu = "Classification Report"

with c5:
    if st.button("📌\nColumns"):
        st.session_state.menu = "Column Analysis"

menu = st.session_state.menu

# ===========================================================
# PROCESS DATA
# ===========================================================
if df is not None:
    unwanted = ["Serial_No","Department","Year","Collaboration_Feedback","Like_Most","Improvement_Areas"]
    cols = [c for c in df.columns if c not in unwanted]

    if st.session_state.combined_texts is None:
        st.session_state.combined_texts = df[cols].astype(str).agg(" ".join, axis=1)

    if "Sentiment" not in df.columns:
        df["Sentiment"] = batch_predict(
            st.session_state.combined_texts.tolist(),
            st.session_state.model,
            st.session_state.tokenizer,
            model_option
        )
        st.session_state.df = df

# ===========================================================
# PAGES
# ===========================================================
if menu == "Upload File":
    st.header("📤 Upload CSV File")
    st.info("Use the sidebar to upload your survey file.")

elif menu == "Data":
    st.header("📄 Survey Data")
    if df is not None:
        st.dataframe(df)
    else:
        st.warning("Upload CSV first")

elif menu == "Sentiment Charts":
    st.header("📊 Overall Sentiment")
    if df is not None:
        fig, ax = plt.subplots()
        df["Sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

elif menu == "Classification Report":
    st.header("📝 Sentiment Summary")
    if df is not None:
        st.write(df["Sentiment"].value_counts())
        st.write((df["Sentiment"].value_counts(normalize=True)*100).round(2))

elif menu == "Column Analysis":
    st.header("📌 Column-wise Analysis")
    if df is not None:
        for col in cols:
            st.subheader(col)
            preds = batch_predict(
                df[col].astype(str).tolist(),
                st.session_state.model,
                st.session_state.tokenizer,
                model_option
            )
            counts = pd.Series(preds).value_counts()
            st.write(counts)
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            st.pyplot(fig)

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
Made with ❤️ | Multi-Model Sentiment Dashboard
</p>
""", unsafe_allow_html=True)
