import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('punkt')

data = {
    'text': [
        "I feel great today!",
        "I am so sad about the news.",
        "This is frustrating!",
        "I'm filled with joy.",
        "That was such a disappointment."
    ],
    'label': ['joy', 'sadness', 'anger', 'joy', 'sadness']
}

df = pd.DataFrame(data)

label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_mapping)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

train_tokens = [word_tokenize(text.lower()) for text in train_texts]
test_tokens = [word_tokenize(text.lower()) for text in test_texts]

counter = Counter(word for tokens in train_tokens for word in tokens)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
vocab['<PAD>'] = 0

def tokens_to_sequence(tokens, vocab):
    return [vocab.get(token, 0) for token in tokens]

train_sequences = [tokens_to_sequence(tokens, vocab) for tokens in train_tokens]
test_sequences = [tokens_to_sequence(tokens, vocab) for tokens in test_tokens]

train_sequences = pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True)
test_sequences = pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True)
train_labels = torch.tensor(train_labels.values)
test_labels = torch.tensor(test_labels.values)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.fc(rnn_out[:, -1, :]) 
        return out

vocab_size = len(vocab)
embed_size = 50
hidden_size = 128
num_classes = len(label_mapping)
learning_rate = 0.001
num_epochs = 10
batch_size = 2

train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = RNNModel(vocab_size, embed_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for texts, labels in train_loader:
        outputs = model(texts)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_mapping.keys()))

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
