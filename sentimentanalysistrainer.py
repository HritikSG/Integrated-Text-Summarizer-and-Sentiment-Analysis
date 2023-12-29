import torch

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.optim import AdamW
import time
from sentimentanalysispreprocessor import train_texts, train_labels, val_texts, val_labels

# Record start time
start_time = time.time()

# Initialize tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create torch dataset
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataloaders
train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Initialize model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
#model = model.to('cuda')  # if cuda is not available
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)
#Load_Path
model_path = r"C:\Users\hriti\OneDrive\Desktop\Train_senti\wetransfer_sentiment_model_roberta-3_2023-11-10_2253\sentiment_model_RoBERTa-3" # Replace this with the actual path


# Load the model
model = RobertaForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(3):  # number of epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')

# Save the model
model.save_pretrained('sentiment_model_RoBERTa')

# Record end time
end_time = time.time()

print("Time required to fine-tune: ", end_time - start_time)

# Evaluate the model
        labels = batch['labels'].to('cuda')
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
model.eval()
predictions = []
true_labels = []
for batch in val_loader:
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    labels = batch['labels'].to('cuda')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(predicted_labels)
    true_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

print(f'Accuracy: {accuracy}')
print(f'F1-score: {f1}')
print(f'Confusion matrix:\n {conf_matrix}')
