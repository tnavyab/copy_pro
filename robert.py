import os
import re
import pandas as pd
import nltk
import contractions
import emoji
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------
# STEP 1: Setup
# ------------------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------------------------
# STEP 2: Load your dataset
# ------------------------------------------------------------
file_path = r"C:\Users\tnavy\OneDrive\Documents\AI PROJECT\students_feedback_predicted.csv"
df = pd.read_csv(file_path, encoding='cp1252')

feedback_columns = [
    "Faculty_Feedback", "Infrastructure_Feedback", "Library_Feedback",
    "Labs_Feedback", "Placements_Feedback", "Internship_Feedback",
    "Extracurricular_Feedback", "IndustryConnect_Feedback",
    "Collaboration_Feedback"
]

df["Combined_Feedback"] = df[feedback_columns].fillna("").agg(" ".join, axis=1)

# ------------------------------------------------------------
# STEP 3: Text cleaning
# ------------------------------------------------------------
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

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
    return ' '.join(words)

df["Cleaned_Feedback"] = df["Combined_Feedback"].apply(clean_text)

# ------------------------------------------------------------
# STEP 4: Prepare features and labels
# ------------------------------------------------------------
X = df["Cleaned_Feedback"].tolist()
y = df["Overall_Sentiment"].astype('category').cat.codes.tolist()

label2id = dict(enumerate(df["Overall_Sentiment"].astype('category').cat.categories))
id2label = {v: k for k, v in label2id.items()}

# ------------------------------------------------------------
# STEP 5: Split data
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# STEP 6: Tokenization (RoBERTa)
# ------------------------------------------------------------
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_data = FeedbackDataset(X_train, y_train, tokenizer)
test_data = FeedbackDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4)

# ------------------------------------------------------------
# STEP 7: Model setup (RoBERTa)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(set(y))
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ------------------------------------------------------------
# STEP 8: Training
# ------------------------------------------------------------
epochs = 1
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# ------------------------------------------------------------
# STEP 9: Evaluation
# ------------------------------------------------------------
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print("\nAccuracy:", round(accuracy_score(true_labels, preds), 3))
print("\nClassification Report:\n",
      classification_report(true_labels, preds, target_names=label2id.values()))

# ------------------------------------------------------------
# STEP 10: Testing new feedback
# ------------------------------------------------------------
new_feedback = [
    "Faculty members are very helpful and supportive.",
    "Infrastructure is poor and placements are disappointing.",
    "College is okay, not too bad overall."
]

new_clean = [clean_text(fb) for fb in new_feedback]

enc = tokenizer(new_clean, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**enc).logits
    pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

print("\nPredictions:")
for fb, p in zip(new_feedback, pred_labels):
    print(f"'{fb}' → {label2id[p]}")

# ------------------------------------------------------------
# STEP 11: Save model
# ------------------------------------------------------------
model.save_pretrained("saved_roberta_model")
tokenizer.save_pretrained("saved_roberta_model")

print("RoBERTa model saved successfully!")
