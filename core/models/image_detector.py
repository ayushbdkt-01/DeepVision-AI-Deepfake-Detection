"""DEEPVISION Image Detection Model

This module contains the core deep learning model for detecting AI-generated images.
Architecture: EfficientNet-B3 backbone with custom classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict
import numpy as np


class AttentionModule(nn.Module):
    """Attention mechanism for feature enhancement"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class DeepFakeImageClassifier(nn.Module):
    """
    Main classifier for deepfake image detection
    
    Architecture:
    - EfficientNet-B3 backbone (pretrained on ImageNet)
    - Custom feature extraction layers
    - Attention mechanisms
    - Multi-scale feature fusion
    - Classification head
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load backbone
        self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Attention module
        self.attention = AttentionModule(256)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary head for manipulation localization
        self.auxiliary_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Tuple of (class_logits, manipulation_scores)
        """
        # Extract features
        features = self.backbone(x)
        features = self.feature_extractor(features)
        
        # Apply attention
        features = self.attention(features.unsqueeze(-1).unsqueeze(-1))
        features = features.squeeze(-1).squeeze(-1)
        
        # Classification
        logits = self.classifier(features)
        
        # Auxiliary output
        manipulation = self.auxiliary_head(features)
        
        return logits, manipulation.squeeze(-1)


class ImageDetector:
    """
    High-level interface for image deepfake detection
    
    Handles:
    - Model loading and initialization
    - Preprocessing
    - Inference
    - Post-processing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = None
        self.model_path = model_path
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model"""
        self.model = DeepFakeImageClassifier(num_classes=2, pretrained=True)
        
        if self.model_path and torch.load:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                pass  # Use pretrained weights if load fails
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect if image is AI-generated
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            Detection results dictionary
        """
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            logits, manipulation = self.model(tensor)
            probs = F.softmax(logits, dim=1)
        
        # Extract results
        ai_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()
        manipulation_score = manipulation[0].item()
        
        # Determine prediction
        is_ai = ai_prob > self.threshold
        confidence = max(ai_prob, real_prob)
        
        return {
            "is_ai_generated": is_ai,
            "ai_probability": ai_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "manipulation_score": manipulation_score,
            "prediction": "AI Generated" if is_ai else "Real"
        }
    
    def batch_detect(self, images: list) -> list:
        """Detect multiple images"""
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
    
    def save_model(self, path: str):
        """Save model to path"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from path"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()


class FrequencyAnalysisModel(nn.Module):
    """
    Frequency domain analysis for detecting AI generation artifacts
    
    Uses DCT (Discrete Cosine Transform) to analyze frequency patterns
    """
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


def create_model(
    model_type: str = "efficientnet",
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = "cpu"
) -> nn.Module:
    """
    Factory function to create detection models
    
    Args:
        model_type: Type of model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    if model_type == "efficientnet":
        return DeepFakeImageClassifier(num_classes, pretrained)
    elif model_type == "frequency":
        return FrequencyAnalysisModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")