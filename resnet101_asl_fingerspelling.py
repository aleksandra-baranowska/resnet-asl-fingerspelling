# -*- coding: utf-8 -*-
"""resnet_asl_fingerspelling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O47_QcWxgod6KfDFyV6K16BcTNw1OHAT
"""

!pip install kaggle datasets transformers torch torchvision

# Import necessary libraries
import os
import zipfile
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from huggingface_hub import HfApi

# Early preprocessing of the data which includes signs other than the alphabet and a test dataset which is not usable due to its small size.

# Download the dataset from Kaggle
!kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset

# Unzip the file
with zipfile.ZipFile('/content/aslamerican-sign-language-aplhabet-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/aslamerican-sign-language-aplhabet-dataset')

# Remove images which are not relevant
dir_to_remove = ['/content/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train/del',
                 '/content/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train/nothing',
                 '/content/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train/space'] # list of unnecessary directories

# Go through the list and remove
for path in dir_to_remove:
    if os.path.exists(path):
        shutil.rmtree(path)

# Set path to dataset
dataset_dir = "/content/aslamerican-sign-language-aplhabet-dataset/ASL_Alphabet_Dataset/asl_alphabet_train"

# Data augmentation
preprocess = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop & resize
    transforms.RandomRotation(15),  # Small rotation for variation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess)

# Split the dataset 70/20/10%
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Test subset
test_set = random_split(dataset, [1000, len(dataset) - 1000])

# Check sizes of the split dataset
print(f"Train size: {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size: {test_size}")

# Apply ResNet preprocessing
train_dataset.dataset.transform = preprocess
val_dataset.dataset.transform = preprocess
test_dataset.dataset.transform = preprocess

#Create Dataloaders for train, validation, and test sets
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check the shape of a batch
batch = next(iter(train_loader))
print(batch)

# Load Pretrained ResNet-101
model = models.resnet101(pretrained=True)

# Initialise device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify the classifier for ASL classification (26 classes)
num_classes = 26
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace last layer

# Unfreeze all layers for full fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Move model to device (GPU or CPU)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (Choose one)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Learning rate scheduler (Optional, but improves training stability)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decays LR every 5 epochs

# Training loop
def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):  # Unpacking batch directly
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)  # ResNet forward pass

        # Compute loss
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping and optimization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / total_batches
    print(f"Training Loss: {epoch_loss:.4f}")

    return epoch_loss


# Evaluation loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:  # Unpacking batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # Predicted labels
            predicted_indices = torch.argmax(logits, dim=1)

            # Calculate metrics
            correct_predictions = (predicted_indices == labels).sum().item()
            total_correct += correct_predictions
            total_samples += len(labels)

            # Compute validation loss
            loss = criterion(logits, labels)
            val_loss += loss.item()

    # Compute overall metrics
    accuracy = total_correct / total_samples
    avg_val_loss = val_loss / len(dataloader)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    return avg_val_loss, accuracy

# Set number of epochs
num_epochs = 10

# Training and evaluation
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(
        model=model,
        dataloader=tqdm(train_loader, desc=f"Epoch {epoch+1}"),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    val_loss, val_accuracy = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device
    )

# Save model state_dict (weights)
torch.save(model.state_dict(), "pytorch_model.bin")

# Save the preprocessing (if applicable)
torch.save(preprocess, "preprocess.pth")

# Push to HuggingFace
api = HfApi()
token = ''

api.upload_file(
    path_or_fileobj="/content/pytorch_model.bin",
    path_in_repo="pytorch_model.bin",
    repo_id="aalof/resnet101-asl-fingerspelling",
    token=token
)

api.upload_file(
    path_or_fileobj="/content/preprocess.pth",
    path_in_repo="preprocess.pth",
    repo_id="aalof/resnet101-asl-fingerspelling",
    token=token
)

def test(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []

    # For storing class-wise F1 scores
    per_class_f1 = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch  # ResNet uses image tensors directly
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            logits = model(inputs)  # ResNet directly outputs logits

            # Predicted labels
            predicted_indices = torch.argmax(logits, dim=1)

            # Collect predictions and true labels for F1 score calculation
            all_predictions.extend(predicted_indices.cpu().tolist())
            all_true_labels.extend(labels.cpu().tolist())

            # Calculate accuracy
            correct_predictions = (predicted_indices == labels).sum().item()
            total_correct += correct_predictions
            total_samples += len(labels)

    # Compute overall accuracy
    accuracy = total_correct / total_samples

    # Compute weighted F1 score
    weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

    # Compute per-class F1 scores
    class_f1_scores = f1_score(all_true_labels, all_predictions, average=None, zero_division=0)

    # Print the results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Weighted F1 Score: {weighted_f1:.4f}")

    # Map F1 scores to class indices
    for i, score in enumerate(class_f1_scores):
        per_class_f1[i] = score  # Maps class index to F1 score

    # Print F1 scores per class
    for class_idx, f1 in per_class_f1.items():
        print(f"Class {class_idx} - F1 Score: {f1:.4f}")

    return accuracy, weighted_f1, per_class_f1

test_accuracy, test_weighted_f1, test_per_class_f1 = test(
    model=model,  # Your trained ResNet model
    dataloader=test_loader,  # Test set DataLoader
    device=device
)