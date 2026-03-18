"""DEEPVISION Training Module

Handles dataset management, training, and validation for custom models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm


class DeepFakeDataset(Dataset):
    """Dataset class for deepfake detection"""
    
    def __init__(
        self,
        real_images: List[Path],
        ai_images: List[Path],
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # Add real images (label 0)
        for img_path in real_images:
            self.image_paths.append(img_path)
            self.labels.append(0)
        
        # Add AI images (label 1)
        for img_path in ai_images:
            self.image_paths.append(img_path)
            self.labels.append(1)
        
        self.image_size = image_size
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label


class Trainer:
    """Training manager for deepfake detection models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer: str = "adam"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Setup optimizer
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "loss": running_loss / len(train_loader),
            "accuracy": 100.0 * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            "loss": running_loss / len(val_loader),
            "accuracy": 100.0 * correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 5
    ) -> Dict:
        """Full training loop"""
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save checkpoint
            if checkpoint_dir and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(checkpoint_dir / "best_model.pt", epoch)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint["epoch"]


def create_data_loaders(
    train_dir: Path,
    val_split: float = 0.2,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Get all images
    real_images = list((train_dir / "real").glob("*.jpg"))
    real_images += list((train_dir / "real").glob("*.png"))
    ai_images = list((train_dir / "ai").glob("*.jpg"))
    ai_images += list((train_dir / "ai").glob("*.png"))
    
    # Shuffle
    np.random.shuffle(real_images)
    np.random.shuffle(ai_images)
    
    # Split
    val_size = int(len(real_images) * val_split)
    
    train_real = real_images[val_size:]
    val_real = real_images[:val_size]
    
    val_size_ai = int(len(ai_images) * val_split)
    train_ai = ai_images[val_size_ai:]
    val_ai = ai_images[:val_size_ai]
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = DeepFakeDataset(train_real, train_ai, transform=transform)
    val_dataset = DeepFakeDataset(val_real, val_ai, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader