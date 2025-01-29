import json
import numpy as np
import torch
import torch.nn as nn
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

# Step 1: Define Dataset Class
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length):
        self.sentences = [word_tokenize(item[0].lower()) for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

        # Encode labels
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        # Save the label encoder classes after fitting
        np.save('label_encoder_classes.npy', self.label_encoder.classes_)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = [self.tokenizer[word] if word in self.tokenizer else self.tokenizer['<UNK>'] for word in self.sentences[idx]]
        tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.encoded_labels[idx], dtype=torch.long)

# Step 2: Define Collate Function for Padding
def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    return sentences_padded, torch.tensor(labels, dtype=torch.long)

# Step 3: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Step 4: Load Data

train_path = './json/is_train.json'
val_path = './json/is_val.json'
test_path = './json/is_test.json'

# Open and load the JSON file
with open(train_path, 'r') as train:
    train_data = json.load(train)
with open(val_path, 'r') as val:
    val_data = json.load(val)
with open(test_path, 'r') as test:
    test_data = json.load(test)

# Step 5: Preprocessing
# Build Vocabulary
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for item in data:
        words = word_tokenize(item[0].lower())
        for word in words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(train_data)

# Save the tokenizer (vocabulary) to a file
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(vocab, f)

label_encoder = LabelEncoder()
max_length = 30

train_dataset = IntentDataset(train_data, vocab, label_encoder, max_length)
val_dataset = IntentDataset(val_data, vocab, label_encoder, max_length)
test_dataset = IntentDataset(test_data, vocab, label_encoder, max_length)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Step 6: Initialize Model
vocab_size = len(vocab)
embed_dim = 200
hidden_dim = 256
output_dim = len(label_encoder.classes_)

model = RNNModel(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

# Step 7: Train Model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sentences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    # Save the entire model after training
    torch.save(model, 'intent_model_full.pth')  # Save the entire model
    # Download the model to your local machine
    #from google.colab import files
    #files.download('intent_model_full.pth')

train_model(model, train_loader, criterion, optimizer)

# Step 8: Evaluate Model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sentences, labels in dataloader:
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):
    plt.figure(figsize=(40, 30))  # Increase figure size for better readability
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")  # Remove annotations for clarity if there are too many classes

    # Set labels with rotation and font size adjustments
    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0, fontsize=8)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    # Adjust layout to fit labels
    plt.tight_layout()
    plt.show()

accuracy, precision, recall, f1, cm = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
plot_confusion_matrix(cm, label_encoder.classes_)

# Step 9: Test Evaluation
test_accuracy, test_precision, test_recall, test_f1, test_cm = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1-score: {test_f1:.4f}")
plot_confusion_matrix(test_cm, label_encoder.classes_)