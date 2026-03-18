"""DEEPVISION Helper Functions"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Union, List, Tuple, Optional
from datetime import datetime
import numpy as np
from PIL import Image


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte)
    return sha256_hash.hexdigest()


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """Format file size to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_temp_dir() -> Path:
    """Create a temporary directory for processing"""
    temp_dir = Path(tempfile.mkdtemp(prefix="deepvision_"))
    return temp_dir


def clean_temp_files(directory: Path, max_age_hours: int = 24):
    """Clean temporary files older than specified hours"""
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                file_path.unlink()


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_video_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid video"""
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'}
    return Path(file_path).suffix.lower() in valid_extensions


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def save_results_json(results: dict, output_path: Path):
    """Save results to JSON file"""
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results_json(input_path: Path) -> dict:
    """Load results from JSON file"""
    import json
    with open(input_path, 'r') as f:
        return json.load(f)


def calculate_confidence_interval(
    predictions: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for predictions"""
    import scipy.stats as stats
    
    if not predictions:
        return 0.0, 0.0
    
    mean = np.mean(predictions)
    std = np.std(predictions)
    n = len(predictions)
    
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * (std / np.sqrt(n))
    
    return mean - margin, mean + margin


def class_to_label(prediction: int) -> str:
    """Convert class prediction to label"""
    return "AI Generated" if prediction == 1 else "Real"


def get_model_summary(model) -> dict:
    """Get summary of model architecture"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def get_device_info() -> dict:
    """Get information about available compute devices"""
    import torch
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info["devices"].append({
                "name": torch.cuda.get_device_name(i),
                "memory": torch.cuda.get_device_properties(i).total_memory
            })
    
    return info