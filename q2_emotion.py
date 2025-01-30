from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from collections import Counter

dataset = load_dataset("emotion")
print(dataset['train'][0])

texts = [example['text'] for example in dataset['train']]
labels = [example['label'] for example in dataset['train']]

def build_vocab(texts):
    words = []
    for text in texts:
        words.extend(text.split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(words).items())}
    vocab["<unk>"] = 0  
    return vocab

vocab = build_vocab(texts)
vocab_size = len(vocab)

def custom_tokenize(example, vocab, max_length=32):
    tokenized_text = [vocab.get(word, vocab["<unk>"]) for word in example['text'].split()]
    tokenized_text = tokenized_text[:max_length] + [vocab["<unk>"]] * (max_length - len(tokenized_text))
    return {'input_ids': tokenized_text, 'label': example['label']}

tokenized_dataset = dataset.map(lambda example: custom_tokenize(example, vocab), batched=False)

tokenized_dataset.set_format(type="torch", columns=["input_ids", "label"])

batch_size = 64
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)
test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=batch_size)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.dropout(self.embedding(input_ids))
        outputs, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNClassifier(
    vocab_size=vocab_size,
    embed_size=128,
    hidden_size=128,
    output_size=6,
    num_layers=1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

def train_model(model, dataloader, optimizer, criterion, epochs=20):
    model.train()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids, labels = (
                batch['input_ids'].to(device),
                batch['label'].to(device),
            )

            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask=None)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Time per Epoch: {epoch_time:.2f}s")

def evaluate_model(model, dataloader):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = (
                batch['input_ids'].to(device),
                batch['label'].to(device),
            )

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask=None)

            preds.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    return true_labels, preds


def plot_confusion_matrix(true_labels, preds, class_names):
    cm = confusion_matrix(true_labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if 'label' in dataset['train'].features:
    class_names = dataset['train'].features['label'].names
else:
    raise ValueError("The dataset does not contain a 'label' feature with class names.")

train_model(model, train_dataloader, optimizer, criterion, epochs=20)
true_labels, preds = evaluate_model(model, val_dataloader)

plot_confusion_matrix(true_labels, preds, class_names)

torch.save(model, '/content/drive/My Drive/RNN_Q.pth')

def predict_emotion(model, vocab, text, max_length=32):
    model.eval()
    with torch.no_grad():
        tokenized_text = [vocab.get(word, vocab["<unk>"]) for word in text.split()]
        tokenized_text = tokenized_text[:max_length] + [vocab["<unk>"]] * (max_length - len(tokenized_text))
        
        input_tensor = torch.tensor([tokenized_text]).to(device)
        
        output = model(input_tensor)
        predicted_label = torch.argmax(output, axis=1).item()
        
        return class_names[predicted_label]

user_text = input("Enter a sentence: ")
predicted_emotion = predict_emotion(model, vocab, user_text)
print(f"Predicted Emotion: {predicted_emotion}")