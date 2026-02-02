import streamlit as st 
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ===========================================================
# PAGE SETTINGS
# ===========================================================
st.set_page_config(page_title="Multi-Model Sentiment Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<div style="background-color:#8A2BE2;padding:18px;border-radius:10px;margin-bottom:10px;">
    <h1 style="color:yellow;text-align:center;">Sentiment Analysis of College Survey</h1>
</div>
""", unsafe_allow_html=True)

# ===========================================================
# SESSION STATE INITIALIZATION
# ===========================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "combined_texts" not in st.session_state:
    st.session_state.combined_texts = None
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

# ===========================================================
# DISTILBERT + BiLSTM MODEL
# ===========================================================
class DistilBERT_BiLSTM_Model(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=3):
        super(DistilBERT_BiLSTM_Model, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = output.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits

# ===========================================================
# LOAD MODELS (CACHED)
# ===========================================================
@st.cache_resource
def load_distilbert_lstm():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBERT_BiLSTM_Model()
    state = torch.load("distilbert_bilstm_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained("saved_roberta_model")
    model = AutoModelForSequenceClassification.from_pretrained("saved_roberta_model")
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_distilbert_hf():
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    model.eval()
    return model, tokenizer

# ===========================================================
# FAST BATCH PREDICTION FUNCTION
# ===========================================================
def batch_predict(text_list, model, tokenizer, model_name, batch_size=32):
    preds = []
    label_map = {0: "Unhappy", 1: "Neutral", 2: "Happy"}

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]

        if model_name == "DistilBERT+BiLSTM":
            enc = tokenizer(batch, padding=True, truncation=True,
                            return_tensors="pt", max_length=64)
            with torch.no_grad():
                logits = model(enc["input_ids"], enc["attention_mask"])
                batch_pred = torch.argmax(logits, dim=1).tolist()
        else:
            enc = tokenizer(batch, padding=True, truncation=True,
                            return_tensors="pt", max_length=64)
            with torch.no_grad():
                logits = model(**enc).logits
                batch_pred = torch.argmax(logits, dim=1).tolist()

        preds.extend([label_map[p] for p in batch_pred])

    return preds


# ===========================================================
# SIDEBAR
# ===========================================================
st.sidebar.markdown("<p class='sidebar-title'>Dashboard Menu</p>", unsafe_allow_html=True)

model_option = st.sidebar.selectbox(
    "Select Model",
    ["DistilBERT+BiLSTM", "RoBERTa", "DistilBERT (HuggingFace)"]
)

menu = st.sidebar.radio(
    "Navigate",
    ["Upload File", "Data", "Sentiment Charts", "Classification Report", "Column Analysis"]
)

uploaded_file = st.sidebar.file_uploader(" Upload Survey CSV", type=["csv"])

# ===========================================================
# LOAD FILE INTO SESSION STATE
# ===========================================================
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

# Use df from session
df = st.session_state.df

# ===========================================================
# PROCESS & PREDICT ONLY ONCE
# ===========================================================
if df is not None:

    unwanted = ["Serial_No","Department","Year","Collaboration_Feedback","Like_Most","Improvement_Areas"]
    cols = [c for c in df.columns if c not in unwanted]

    if st.session_state.combined_texts is None:
        st.session_state.combined_texts = df[cols].astype(str).agg(" ".join, axis=1)

    # Load model only once
    if st.session_state.model is None:
        if model_option == "DistilBERT+BiLSTM":
            st.session_state.model, st.session_state.tokenizer = load_distilbert_lstm()
        elif model_option == "RoBERTa":
            st.session_state.model, st.session_state.tokenizer = load_roberta()
        else:
            st.session_state.model, st.session_state.tokenizer = load_distilbert_hf()

    # Predict once
    if "Sentiment" not in df.columns:
        st.sidebar.write("Running AI Model… ⏳")
        df["Sentiment"] = batch_predict(
            st.session_state.combined_texts.tolist(),
            st.session_state.model,
            st.session_state.tokenizer,
            model_option
        )
        st.session_state.df = df  # update session state


# ===========================================================
# UPLOAD PAGE
# ===========================================================
if menu == "Upload File":
    st.header("📤 Upload Your Survey CSV File")
    st.write("Use sidebar to upload CSV.")

# ===========================================================
# DATA PAGE
# ===========================================================
if menu == "Data":
    if df is not None:
        st.header("📄 Survey Dataset")
        st.dataframe(df)
    else:
        st.warning("Upload CSV first.")

# ===========================================================
# SENTIMENT CHARTS
# ===========================================================
if menu == "Sentiment Charts":
    if df is not None:
        st.header("📊 Overall Sentiment Distribution")

        fig1, ax1 = plt.subplots(figsize=(3,3))
        df["Sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax1, ylabel="")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(4,2))
        df["Sentiment"].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Upload CSV first.")

# ===========================================================
# CLASSIFICATION REPORT
# ===========================================================
if menu == "Classification Report":
    if df is not None:
        st.header("📝 Sentiment Summary")
        st.write(df["Sentiment"].value_counts())
        st.write((df["Sentiment"].value_counts(normalize=True)*100).round(2))
    else:
        st.warning("Upload CSV first.")

# ===========================================================
# COLUMN ANALYSIS
# ===========================================================
if menu == "Column Analysis":
    if df is not None:
        st.header("📌 Column-wise Sentiment Analysis")

        for col in cols:
            st.subheader(f"🔹 {col}")

            col_preds = batch_predict(
                df[col].astype(str).tolist(),
                st.session_state.model,
                st.session_state.tokenizer,
                model_option
            )

            df[col + "_sentiment"] = col_preds
            counts = df[col + "_sentiment"].value_counts()

            st.write(counts)

            fig, ax = plt.subplots(figsize=(4,3))
            counts.plot(kind="bar", ax=ax)
            st.pyplot(fig)

    else:
        st.warning("Upload CSV first.")

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("""
<hr>
<p style='text-align:center; color: grey;'>
Made with ❤️ using Multiple Models | College Survey Dashboard  
</p>
""", unsafe_allow_html=True)
