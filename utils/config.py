"""DEEPVISION Configuration Module"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for detection models"""
    image_model_name: str = "efficientnet_b3"
    video_model_name: str = "r3d_18"
    confidence_threshold: float = 0.5
    num_classes: int = 2
    input_size: tuple = (224, 224)
    device: str = "cpu"


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    max_image_size: int = 50 * 1024 * 1024  # 50MB
    max_video_size: int = 500 * 1024 * 1024  # 500MB
    max_video_duration: int = 300  # 5 minutes
    batch_size: int = 32
    num_workers: int = 4
    video_frame_sample_rate: int = 1


@dataclass
class UIConfig:
    """Configuration for UI settings"""
    page_title: str = "DEEPVISION - Deepfake Detection"
    page_icon: str = "🔍"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme: str = "dark"


@dataclass
class PathsConfig:
    """Configuration for file paths"""
    project_root: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    sample_data_dir: Optional[Path] = None
    temp_dir: Optional[Path] = None

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.project_root / "data" / "checkpoints"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "data" / "logs"
        if self.sample_data_dir is None:
            self.sample_data_dir = self.project_root / "data" / "sample_data"
        if self.temp_dir is None:
            self.temp_dir = self.project_root / "data" / "temp"


class Config:
    """Main configuration class"""

    def __init__(self):
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.ui = UIConfig()
        self.paths = PathsConfig()

        # Set device
        self._set_device()

    def _set_device(self):
        """Set processing device"""
        import torch
        if torch.cuda.is_available():
            self.model.device = "cuda"
        else:
            self.model.device = "cpu"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()

        # Override from environment
        device_env = os.getenv("DEEPVISION_DEVICE")
        if device_env:
            config.model.device = device_env

        model_env = os.getenv("DEEPVISION_MODEL")
        if model_env:
            config.model.image_model_name = model_env

        return config


# Global configuration instance
config = Config()