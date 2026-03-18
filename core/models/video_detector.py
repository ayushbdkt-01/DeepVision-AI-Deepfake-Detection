"""DEEPVISION Video Detection Model

This module handles video deepfake detection including frame analysis,
temporal consistency checking, and motion anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
from collections import deque


class TemporalConsistencyModule(nn.Module):
    """Module for analyzing temporal consistency in videos"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # x: (B, T, C) where T is sequence length
        lstm_out, _ = self.lstm(x)
        
        # Attention
        weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * weights, dim=1)
        
        return self.classifier(context)


class VideoFrameEncoder(nn.Module):
    """Encode individual video frames"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)


class VideoDeepfakeDetector(nn.Module):
    """
    Video deepfake detection model
    
    Combines:
    - Frame-level features
    - Temporal consistency analysis
    - Motion pattern detection
    """
    
    def __init__(
        self,
        num_frames: int = 16,
        hidden_size: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Frame encoder
        self.frame_encoder = VideoFrameEncoder(feature_dim=256)
        
        # Temporal consistency module
        self.temporal_module = TemporalConsistencyModule(hidden_size=hidden_size)
        
        # Spatial 3D convolution for motion
        self.spatial_3d = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Tuple of (logits, temporal_consistency)
        """
        batch_size, channels, num_frames, height, width = x.shape
        
        # Extract frame features
        frame_features = []
        for t in range(num_frames):
            frame = x[:, :, t, :, :]
            features = self.frame_encoder(frame)
            frame_features.append(features)
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)  # (B, T, 256)
        
        # Temporal analysis
        temporal_out = self.temporal_module(temporal_features)  # (B, 2)
        
        # Spatial-temporal analysis
        x_3d = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x_3d = x_3d.reshape(batch_size, channels, num_frames, height, width)
        spatial_features = self.spatial_3d(x_3d)
        spatial_features = spatial_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Fusion
        combined = torch.cat([temporal_features.mean(dim=1), spatial_features], dim=1)
        logits = self.fusion(combined)
        
        return logits, temporal_out


class VideoDetector:
    """
    High-level interface for video deepfake detection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        num_frames: int = 16,
        threshold: float = 0.5
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.num_frames = num_frames
        self.model = None
        self.frame_buffer = deque(maxlen=num_frames)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = VideoDeepfakeDetector(num_frames=self.num_frames)
        
        if self.model is not None:
            try:
                state_dict = torch.load(self.model, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                pass
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_frames(
        self, 
        video_path: str, 
        max_frames: int = 32,
        sample_rate: int = 1
    ) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for model input"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Sample frames if needed
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert to tensor
        tensors = [transform(frame).unsqueeze(0) for frame in frames]
        batch = torch.cat(tensors, dim=0).unsqueeze(0)
        
        return batch.to(self.device)
    
    def detect_video(self, video_path: str) -> Dict:
        """Detect deepfake in video"""
        # Extract frames
        frames = self.extract_frames(video_path, max_frames=32)
        
        if not frames:
            return {
                "is_ai_generated": False,
                "ai_probability": 0.0,
                "real_probability": 0.0,
                "confidence": 0.0,
                "error": "Could not extract frames from video"
            }
        
        # Preprocess
        tensor = self.preprocess_frames(frames)
        
        # Inference
        with torch.no_grad():
            logits, temporal = self.model(tensor)
            probs = F.softmax(logits, dim=1)
        
        ai_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()
        confidence = max(ai_prob, real_prob)
        
        return {
            "is_ai_generated": ai_prob > self.threshold,
            "ai_probability": ai_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "temporal_consistency": temporal[0].cpu().numpy().tolist(),
            "num_frames_analyzed": len(frames),
            "prediction": "AI Generated" if ai_prob > self.threshold else "Real"
        }


class VideoFrameAnalyzer:
    """Analyze individual frames in video for localized manipulation detection"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.detector = None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze single frame"""
        # Placeholder for frame-level analysis
        return {
            "is_manipulated": False,
            "manipulation_region": None,
            "confidence": 0.0
        }
    
    def analyze_all_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Analyze all frames"""
        return [self.analyze_frame(frame) for frame in frames]


def create_video_model(
    num_frames: int = 16,
    device: str = "cpu"
) -> nn.Module:
    """Create video detection model"""
    return VideoDeepfakeDetector(num_frames=num_frames)