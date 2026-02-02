import os
import re
import pandas as pd
import nltk
import contractions
import emoji
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------
# Step 1: NLTK Setup
# ------------------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------
file_path = r"C:\Users\tnavy\OneDrive\Documents\AI PROJECT\survey.csv"
df = pd.read_csv(file_path)

feedback_columns = [
    'Faculty_Feedback', 'Infrastructure_Feedback', 'Library_Feedback',
    'Labs_Feedback', 'Placements_Feedback', 'Internship_Feedback',
    'Extracurricular_Feedback', 'IndustryConnect_Feedback', 'Collaboration_Feedback'
]

# Combine all feedback columns
df["Combined_Feedback"] = df[feedback_columns].fillna("").agg(" ".join, axis=1)

label_col = "Overall_Sentiment"
text_col = "Combined_Feedback"

df = df[[text_col, label_col]].dropna()

# Encode labels
df[label_col] = df[label_col].astype('category')
label2id = dict(enumerate(df[label_col].cat.categories))
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df[label_col].cat.codes

# ------------------------------------------------------------
# Step 3: Clean Text
# ------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = contractions.fix(text)
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

df["cleaned"] = df[text_col].apply(clean_text)

# ------------------------------------------------------------
# Step 4: Split Data
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# Step 5: Tokenizer
# ------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = FeedbackDataset(X_train, y_train)
test_ds = FeedbackDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# ------------------------------------------------------------
# Step 6: DistilBERT + BiLSTM Model
# ------------------------------------------------------------
class DistilBertBiLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(out.last_hidden_state)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

# ------------------------------------------------------------
# Step 7: Train Model
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertBiLSTM(num_labels=len(label2id)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attn)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# ------------------------------------------------------------
# Step 8: Evaluate
# ------------------------------------------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attn)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=df[label_col].cat.categories))

# ------------------------------------------------------------
# Step 9: Save Model
# ------------------------------------------------------------
torch.save(model.state_dict(), "distilbert_bilstm_model.pth")
print("\nModel saved successfully!")
