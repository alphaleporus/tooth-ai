#!/usr/bin/env python3
"""
Train EfficientNet-B0 classifier for FDI tooth numbering.
Takes ROI crops and classifies them into 32 FDI classes.
"""

import argparse
import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import wandb
from tqdm import tqdm
import timm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class ROIDataset(Dataset):
    """Dataset for ROI tooth images."""
    
    def __init__(self, data_dir: str, labels_csv: str, transform=None):
        """
        Args:
            data_dir: Directory containing ROI images
            labels_csv: Path to CSV file with labels
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels
        self.samples = []
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                label = int(row['fdi_label'])
                # Map to 0-31 range (assuming 32 classes)
                label = label - 1 if label > 0 else 0  # Adjust if needed
                self.samples.append((filename, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.data_dir, filename)
        
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            # Return black image if file not found
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_augmentations(is_training: bool = True):
    """Get data augmentation transforms."""
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def compute_class_weights(dataset: ROIDataset) -> torch.Tensor:
    """Compute class weights for handling imbalance."""
    class_counts = defaultdict(int)
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    total = len(dataset)
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)
    
    for cls_id, count in class_counts.items():
        weights[cls_id] = total / (num_classes * count)
    
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B0 for FDI classification')
    parser.add_argument('--data', type=str, required=True, help='ROI dataset directory')
    parser.add_argument('--epochs', type=int, default=35, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--out', type=str, required=True, help='Output model path')
    parser.add_argument('--wandb-project', type=str, default='tooth-poc', help='wandb project')
    parser.add_argument('--wandb-name', type=str, default='effnet_roi_fdi', help='wandb run name')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of classes')
    parser.add_argument('--image-size', type=int, default=128, help='Input image size')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ROIDataset(
        os.path.join(args.data, 'train'),
        os.path.join(args.data, 'train_labels.csv'),
        transform=get_augmentations(is_training=True)
    )
    
    val_dataset = ROIDataset(
        os.path.join(args.data, 'val'),
        os.path.join(args.data, 'val_labels.csv'),
        transform=get_augmentations(is_training=False)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=args.num_classes)
    model = model.to(device)
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    print(f"Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler (cosine decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = args.out.replace('.pth', '_best.pth')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }, best_model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # Classification report
    class_names = [f'Class_{i+1}' for i in range(args.num_classes)]
    report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Plot confusion matrix
    cm_path = args.out.replace('.pth', '_confusion_matrix.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': vars(args),
        'classification_report': report
    }, args.out)
    
    print(f"\nTraining complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final model saved to: {args.out}")
    print(f"  Best model saved to: {best_model_path}")
    
    wandb.finish()


if __name__ == '__main__':
    main()



