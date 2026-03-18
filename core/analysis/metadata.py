"""DEEPVISION Analysis Module - Metadata Analysis

Analyzes file metadata to detect AI-generated images.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import struct


class MetadataAnalyzer:
    """
    Analyze image metadata to detect AI-generated images.
    
    AI-generated images often lack proper EXIF data or have
    inconsistent metadata.
    """
    
    def __init__(self):
        self.threshold = 0.5
    
    def extract_exif(self, file_path: str) -> Dict:
        """Extract EXIF data from image"""
        exif_data = {}
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(file_path) as img:
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
                        
        except Exception as e:
            exif_data["error"] = str(e)
        
        return exif_data
    
    def analyze_metadata(self, metadata: Dict) -> Dict:
        """
        Analyze metadata for AI generation indicators
        
        Returns:
            Analysis results
        """
        indicators = []
        issues = []
        
        # Check for missing metadata
        if not metadata or len(metadata) < 3:
            indicators.append(0.3)
            issues.append("Missing or minimal metadata")
        
        # Check for missing camera info (common in AI images)
        camera_tags = ['Make', 'Model', 'LensModel']
        has_camera = any(tag in metadata for tag in camera_tags)
        
        if not has_camera:
            indicators.append(0.2)
            issues.append("No camera information")
        
        # Check for missing timestamp
        time_tags = ['DateTime', 'DateTimeOriginal', 'CreateDate']
        has_time = any(tag in metadata for tag in time_tags)
        
        if not has_time:
            indicators.append(0.1)
            issues.append("No timestamp information")
        
        # Check software tag (AI generators often add software info)
        software_tags = ['Software', 'ProcessingSoftware']
        has_software = any(tag in metadata for tag in software_tags)
        
        if has_software:
            software = metadata.get('Software', '')
            if any(gen in software.lower() for gen in ['midjourney', 'stable diffusion', 'dall-e', 'ai', 'generated']):
                indicators.append(0.8)
                issues.append(f"AI software detected: {software}")
        
        # Check for suspicious GPS data
        if 'GPSInfo' in metadata:
            # GPS data present - good sign for real photo
            pass
        else:
            # No GPS - could be either
            indicators.append(0.05)
        
        # Calculate AI probability
        ai_probability = min(sum(indicators) / max(len(indicators), 1), 1.0)
        
        return {
            "ai_probability": ai_probability,
            "indicators": indicators,
            "issues": issues,
            "is_ai_generated": ai_probability > self.threshold,
            "metadata_count": len(metadata) if metadata else 0
        }
    
    def analyze(self, metadata: Dict) -> Dict:
        """Analyze metadata dictionary"""
        return self.analyze_metadata(metadata)
    
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze file metadata"""
        metadata = self.extract_exif(file_path)
        return self.analyze(metadata)


class FileSignatureAnalyzer:
    """Analyze file signatures for AI-generated content"""
    
    KNOWN_AI_SIGNATURES = [
        b'Midjourney',
        b'Stable Diffusion',
        b'DALL-E',
        b'Dream',
        b'MJ'
    ]
    
    def analyze_signature(self, file_path: str) -> Dict:
        """Analyze file header for known signatures"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            for signature in self.KNOWN_AI_SIGNATURES:
                if signature in header:
                    return {
                        "detected": True,
                        "signature": signature.decode('utf-8', errors='ignore'),
                        "ai_probability": 0.9
                    }
            
            return {
                "detected": False,
                "signature": None,
                "ai_probability": 0.0
            }
            
        except Exception as e:
            return {
                "detected": False,
                "error": str(e),
                "ai_probability": 0.0
            }


def test_metadata_analyzer():
    """Test the metadata analyzer"""
    analyzer = MetadataAnalyzer()
    
    # Test with empty metadata
    result = analyzer.analyze({})
    print("Empty metadata result:", result)
    
    # Test with camera metadata
    test_meta = {
        'Make': 'Canon',
        'Model': 'EOS R5',
        'DateTime': '2024:01:01 12:00:00'
    }
    result = analyzer.analyze(test_meta)
    print("Real camera metadata result:", result)


if __name__ == "__main__":
    test_metadata_analyzer()