'''
Hybrid Transformer - FT
BERT contextualized embeddings x numerical features (readerbench indices)

Pre-req: Make sure you have transformers installed 

!pip install transformers
!pip install transformers[torch]
!pip install accelerate -U
!pip install torch torchtext
!pip install tqdm
'''

import torch
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path='ro_fulltext_250.csv'
numerical_path='ro_fulltext_rbi_250.csv'

all_data = pd.read_csv(data_path)
numerical_data = pd.read_csv(numerical_path)

# KW
df = pd.read_csv(data_path)
df = df.fillna(0)

index_columns = df.columns[2:]
results = []

for index_column in index_columns:
    groups = [df[index_column][df['author'] == author] for author in df['author'].unique()]

    if any(len(set(group)) > 1 for group in groups):
        stat, p_value = kruskal(*groups)
        results.append((index_column, stat, p_value))

results.sort(key=lambda x: x[1], reverse=True)
top_results = results[:100]

all_data.rename(columns={'author': 'label'}, inplace=True)
numerical_data.rename(columns={'author': 'label'}, inplace=True)

train_data, test_data = train_test_split(all_data, test_size=0.2, stratify=all_data['label'], random_state=42)

train_numerical_data, test_numerical_data = train_test_split(numerical_data, test_size=0.2, stratify=numerical_data['label'], random_state=42)

train_texts = train_data['text'].tolist()
train_authors = train_data['label'].tolist()
test_texts = test_data['text'].tolist()
test_authors = test_data['label'].tolist()

label_map = {author: idx for idx, author in enumerate(set(train_authors))}

train_int_labels = [label_map[author] for author in train_authors]
test_int_labels = [label_map[author] for author in test_authors]

top_features = [result[0] for result in top_results]
train_numerical_features = train_numerical_data[top_features]
test_numerical_features = test_numerical_data[top_features]

tokenizer = AutoTokenizer.from_pretrained("readerbench/robert-base")
model = AutoModel.from_pretrained("readerbench/robert-base")
model = model.to(device)

class TextDataset(Dataset):
    def __init__(self, texts, labels, numerical_features, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.numerical_features = torch.tensor(numerical_features.values, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {
            key: val[idx].to(device) for key, val in self.encodings.items()
        }
        item["labels"] = self.labels[idx].to(device)
        item["numerical_features"] = self.numerical_features[idx].to(device)
        return item

    def __len__(self):
        return len(self.labels)

train_numerical_features = (train_numerical_features - train_numerical_features.mean()) / train_numerical_features.std()
test_numerical_features = (test_numerical_features - test_numerical_features.mean()) / test_numerical_features.std()
train_dataset = TextDataset(train_texts, train_int_labels, train_numerical_features, tokenizer)
test_dataset = TextDataset(test_texts, test_int_labels, test_numerical_features, tokenizer)

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, textual_feature_dim, numerical_feature_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.bert = model
        self.dropout = torch.nn.Dropout(0.2)
        self.fc_text = torch.nn.Linear(textual_feature_dim, 768)
        self.fc_numerical = torch.nn.Linear(numerical_feature_dim, 100)
        self.fc_final = torch.nn.Linear(868, num_classes)

    def forward(self, input_ids, attention_mask, numerical_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = outputs.pooler_output
        text_embedding = self.fc_text(self.dropout(text_embedding))
        numerical_embedding = self.fc_numerical(numerical_features)
        multimodal_vector = torch.cat([text_embedding, numerical_embedding], dim=1)
        output = self.fc_final(self.dropout(multimodal_vector))
        return output

multimodal_classifier = MultimodalClassifier(768, 100, len(set(train_int_labels)))
multimodal_classifier = multimodal_classifier.to(device)

training_args = TrainingArguments(
    output_dir="./bert-base-classifier",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=32,
    logging_dir="./logs",
)

optimizer = torch.optim.AdamW(multimodal_classifier.parameters(), lr=1e-5, weight_decay=0.01)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

for epoch in tqdm(range(training_args.num_train_epochs), desc="Training"):
    for batch in torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        numerical_features = batch['numerical_features']
        labels = batch['labels']
        outputs = multimodal_classifier(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_logits = []
for batch in test_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    numerical_features = batch['numerical_features']
    with torch.no_grad():
        outputs = multimodal_classifier(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
        all_logits.append(outputs)
all_logits = torch.cat(all_logits, dim=0)
predicted_labels = all_logits.argmax(dim=1)

accuracy = accuracy_score(test_int_labels, predicted_labels.cpu())
precision = precision_score(test_int_labels, predicted_labels.cpu(), average="weighted")
recall = recall_score(test_int_labels, predicted_labels.cpu(), average="weighted")
f1 = f1_score(test_int_labels, predicted_labels.cpu(), average="weighted")
error_rate = 1 - accuracy

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Error Rate: {error_rate:.4f}")

class_report = classification_report(test_int_labels, predicted_labels.cpu(), target_names=list(label_map.keys()))
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(test_int_labels, predicted_labels.cpu())
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.xlabel("Predicted Authors")
plt.ylabel("True Authors")
plt.title("Confusion Matrix")
plt.show()
