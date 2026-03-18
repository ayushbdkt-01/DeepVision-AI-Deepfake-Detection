"""DEEPVISION Image Preprocessing Module

Handles image loading, validation, and preprocessing for the detection models.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
from PIL import Image
import io


class ImageProcessor:
    """
    Image preprocessing for deepfake detection
    
    Handles:
    - Image loading from various formats
    - Validation and error handling
    - Preprocessing for model input
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        # Load with PIL for better format support
        img = Image.open(file_path)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        return img_array
    
    def load_from_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes"""
        img = Image.open(io.BytesIO(data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate image format and quality"""
        if image is None or image.size == 0:
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        return True
    
    def preprocess(
        self,
        image: np.ndarray,
        resize: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image
            resize: Whether to resize to target size
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        # Resize
        if resize:
            image = cv2.resize(image, self.target_size)
        
        # Normalize to [0, 1]
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def augment(
        self,
        image: np.ndarray,
        horizontal_flip: bool = True,
        rotation: bool = True,
        brightness: bool = True
    ) -> list:
        """
        Apply data augmentation
        
        Args:
            image: Input image
            horizontal_flip: Apply horizontal flip
            rotation: Apply random rotation
            brightness: Apply brightness adjustment
            
        Returns:
            List of augmented images
        """
        augmented = [image.copy()]
        
        # Horizontal flip
        if horizontal_flip:
            flipped = cv2.flip(image, 1)
            augmented.append(flipped)
        
        # Rotation
        if rotation:
            h, w = image.shape[:2]
            angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        # Brightness
        if brightness:
            factor = np.random.uniform(0.8, 1.2)
            brightened = np.clip(image * factor, 0, 255).astype(np.uint8)
            augmented.append(brightened)
        
        return augmented
    
    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (64, 64),
        stride: int = 32
    ) -> list:
        """Extract patches from image"""
        h, w = image.shape[:2]
        ph, pw = patch_size
        
        patches = []
        for y in range(0, h - ph + 1, stride):
            for x in range(0, w - pw + 1, stride):
                patch = image[y:y+ph, x:x+pw]
                patches.append(patch)
        
        return patches
    
    def resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        
        return padded, (new_w, new_h)


def validate_image_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate image file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, "File does not exist"
    
    if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
        return False, f"Unsupported format: {path.suffix}"
    
    if path.stat().st_size > ImageProcessor.MAX_SIZE:
        return False, f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB"
    
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image: {str(e)}"