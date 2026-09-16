"""Train PyTorch ResNet-18 on Real Accident Dataset.

Fine-tunes ResNet-18 transfer learning backbone on the Accident/Non Accident dataset:
- Freezes early feature extractor
- Fine-tunes layer4 and classification head
- Evaluates on test split (Accuracy, Precision, Recall, F1)
- Saves weights to models/accident_resnet18.pth for real Grad-CAM visual attention
"""
from __future__ import annotations

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

def train_accident_classifier(
    data_dir: str = "data/accident_images",
    models_dir: str = "models",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    print("=" * 60)
    print("  TRAINING DEEP LEARNING ACCIDENT DETECTOR (RESNET-18)  ")
    print("=" * 60)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path):
        print(f"Error: Dataset train directory not found at {train_path}")
        return None

    # Standard ImageNet transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\n[Data] Loading ImageFolder datasets...")
    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_transforms)

    print(f"Classes: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    # Transfer Learning with ResNet-18
    print("[Model] Initializing pretrained ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze earlier conv layers, fine-tune layer4 + fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    history = {"train_loss": [], "val_acc": []}

    print(f"\n[Training] Starting fine-tuning for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2%}")

    train_duration = time.time() - start_time
    print(f"\n[Done] Training completed in {train_duration:.1f}s.")

    # Final Evaluation on Test Split
    print("\n--- Evaluating on Independent Test Set (100 images) ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", pos_label=0 if train_dataset.class_to_idx.get("Accident") == 0 else 1)
    # Target class: Accident
    acc_idx = train_dataset.class_to_idx.get("Accident", 0)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", pos_label=acc_idx)

    print(f"Test Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Test Precision: {p:.4f}")
    print(f"Test Recall:    {r:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")

    # Save Model Weights
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "accident_resnet18.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"\n[Saved] ResNet-18 weights saved -> {weights_path}")

    # Plot Confusion Matrix
    metrics_dir = os.path.join(models_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"ResNet-18 Test Confusion Matrix\nAccuracy: {acc*100:.1f}% | F1: {f1:.3f}")
    plt.tight_layout()
    cm_plot_path = os.path.join(metrics_dir, "resnet18_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrix plot -> {cm_plot_path}")

    return {
        "model_path": weights_path,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "classes": train_dataset.classes,
    }


if __name__ == "__main__":
    train_accident_classifier()
