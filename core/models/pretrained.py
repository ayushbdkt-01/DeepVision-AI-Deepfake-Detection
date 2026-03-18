"""DEEPVISION Pretrained Model Loading Utilities

Utilities for loading pretrained models and checkpoints.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, List
import requests
import os


MODEL_REGISTRY = {
    "efficientnet_b3": {
        "url": "https://download.pytorch.org/models/efficientnet_b3_ra2-cf784f40.pth",
        "filename": "efficientnet_b3.pth",
        "description": "EfficientNet-B3 for image classification"
    },
    "resnet50": {
        "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "filename": "resnet50.pth",
        "description": "ResNet50 for image classification"
    },
    "vit_base": {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "filename": "vit_base.pth",
        "description": "Vision Transformer Base"
    }
}


def get_available_models() -> List[str]:
    """Get list of available pretrained models"""
    return list(MODEL_REGISTRY.keys())


def download_model(
    model_name: str,
    save_dir: Path,
    force: bool = False
) -> Optional[Path]:
    """
    Download pretrained model
    
    Args:
        model_name: Name of model to download
        save_dir: Directory to save model
        force: Force re-download even if exists
        
    Returns:
        Path to downloaded model or None
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_REGISTRY[model_name]
    save_path = save_dir / model_info["filename"]
    
    if save_path.exists() and not force:
        return save_path
    
    try:
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return save_path
        
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return None


def load_pretrained_model(
    model_name: str,
    model_path: Optional[Path] = None,
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load a pretrained model
    
    Args:
        model_name: Name of the model
        model_path: Optional path to local model file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from torchvision import models
    
    if model_path and model_path.exists():
        # Load from local path
        state_dict = torch.load(model_path, map_location=device)
    else:
        # Load from torchvision
        if model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights='IMAGENET1K_V1')
            return model.to(device)
        elif model_name == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V1')
            return model.to(device)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            return model.to(device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return state_dict


class ModelCheckpoint:
    """Handle model checkpointing"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        filename: str = "checkpoint.pt"
    ):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        return save_path
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = "checkpoint.pt"
    ) -> Dict:
        """Load model checkpoint"""
        load_path = self.checkpoint_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints"""
        return sorted(self.checkpoint_dir.glob("*.pt"))


def get_model_info(model_name: str) -> Dict:
    """Get information about a model"""
    if model_name not in MODEL_REGISTRY:
        return {"error": f"Unknown model: {model_name}"}
    
    return MODEL_REGISTRY[model_name]