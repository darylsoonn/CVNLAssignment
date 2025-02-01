import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
EPOCHS_ROBOFLOW = 10  # Roboflow Init Training
EPOCHS_ASL = 3  # ASL Fine tuning
EPOCHS_FINAL_ROBOFLOW = 7  # Roboflow Fine Tuning
LEARNING_RATE = 0.0001
IMG_SIZE = 64
NUM_CLASSES = 26 # A-Z
WEIGHT_DECAY = 1e-4  # Regularization to reduce overfitting
EARLY_STOPPING_PATIENCE = 3  # Stop training if val loss does not improve

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image Augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Roboflow Init Training
roboflow_dir = '/content/American-Sign-Language-Letters-1/train'
roboflow_dataset = datasets.ImageFolder(root=roboflow_dir, transform=train_transform)

# Artificial Dataset Expansion (~4000 images)
expanded_dataset = roboflow_dataset.samples * 3
roboflow_dataset.samples = expanded_dataset

# Split training and validation
train_size_robo = int(0.8 * len(roboflow_dataset))
val_size_robo = len(roboflow_dataset) - train_size_robo
train_dataset_robo, val_dataset_robo = random_split(roboflow_dataset, [train_size_robo, val_size_robo])

train_dataset_robo.dataset.transform = train_transform
val_dataset_robo.dataset.transform = val_transform

# ASL Fine tuneing
data_dir_asl = '/content/drive/MyDrive/asl_alphabet_dataset/train'
asl_dataset = datasets.ImageFolder(root=data_dir_asl, transform=train_transform)

# Limit sample size (10k image)
subset_size_asl = min(10000, len(asl_dataset))
subset_indices_asl = torch.randperm(len(asl_dataset))[:subset_size_asl]
asl_subset = Subset(asl_dataset, subset_indices_asl)

# Split training and validation
train_size_asl = int(0.8 * len(asl_subset))
val_size_asl = len(asl_subset) - train_size_asl
train_dataset_asl, val_dataset_asl = random_split(asl_subset, [train_size_asl, val_size_asl])

train_dataset_asl.dataset.transform = train_transform
val_dataset_asl.dataset.transform = val_transform

# DataLoaders
train_loader_robo = DataLoader(train_dataset_robo, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader_robo = DataLoader(val_dataset_robo, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
train_loader_asl = DataLoader(train_dataset_asl, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader_asl = DataLoader(val_dataset_asl, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# CNN Model (3, 64, 128, 256, 512)
class aslCNN(nn.Module):
    def __init__(self, num_classes):
        super(aslCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Init Model
model = aslCNN(num_classes=NUM_CLASSES).to(device)

# Loss Fn and Optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training Fn with Early Stop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, phase):
    model.train()
    best_val_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} ({phase} Phase)")
        running_loss = 0.0

        # Training
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % (len(train_loader) // 10) == 0:
                print(f"Progress: {batch_idx / len(train_loader) * 100:.1f}%")

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Validation Loss: {val_losses[-1]:.4f}")

        # Early Stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/content/drive/MyDrive/asl_model_hailmary.pth")
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early Stopping Triggered!")
                break

# Flow plan
train_model(model, train_loader_robo, val_loader_robo, criterion, optimizer, EPOCHS_ROBOFLOW, "Roboflow")
train_model(model, train_loader_asl, val_loader_asl, criterion, optimizer, EPOCHS_ASL, "ASL")
train_model(model, train_loader_robo, val_loader_robo, criterion, optimizer, EPOCHS_FINAL_ROBOFLOW, "Roboflow_Final")

def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Run model evaluation after final training
evaluate_model(model, val_loader_robo)