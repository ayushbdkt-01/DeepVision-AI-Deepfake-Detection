"""DEEPVISION Analysis Module - Artifacts Detection

Detects common AI generation artifacts like faces, inconsistent lighting,
and semantic anomalies.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ArtifactRegion:
    """Region containing detected artifact"""
    x: int
    y: int
    width: int
    height: int
    score: float
    artifact_type: str


class ArtifactDetector:
    """
    Detect common AI generation artifacts
    
    Types of artifacts:
    - Face inconsistencies
    - Lighting anomalies
    - Texture irregularities
    - Geometric distortions
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.face_cascade = None
        self._load_detectors()
    
    def _load_detectors(self):
        """Load pre-trained detectors"""
        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            pass
    
    def detect_face_regions(self, image: np.ndarray) -> List[ArtifactRegion]:
        """Detect faces in image"""
        if self.face_cascade is None:
            return []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        regions = []
        for (x, y, w, h) in faces:
            regions.append(ArtifactRegion(
                x=x, y=y, width=w, height=h,
                score=0.8, artifact_type="face"
            ))
        
        return regions
    
    def analyze_lighting_consistency(self, image: np.ndarray) -> Dict:
        """
        Analyze lighting consistency across the image
        
        AI-generated images often have inconsistent lighting
        """
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value = hsv[:,:,2]
        else:
            value = image
        
        # Divide image into grid
        grid_size = 4
        h, w = value.shape
        block_h, block_w = h // grid_size, w // grid_size
        
        brightness_values = []
        for i in range(grid_size):
            for j in range(grid_size):
                block = value[
                    i*block_h:(i+1)*block_h,
                    j*block_w:(j+1)*block_w
                ]
                brightness_values.append(np.mean(block))
        
        # Check variance in brightness
        brightness_std = np.std(brightness_values)
        brightness_mean = np.mean(brightness_values)
        
        # High variance can indicate AI generation
        lighting_score = min(brightness_std / 50, 1.0)
        
        return {
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "lighting_score": lighting_score,
            "consistent": brightness_std < 20
        }
    
    def detect_texture_irregularities(self, image: np.ndarray) -> Dict:
        """
        Detect texture irregularities
        
        AI-generated images may have unnatural texture patterns
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gabor filter bank
        gabor_responses = []
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel(
                (21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(np.mean(filtered))
        
        # Compute entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            "gabor_responses": gabor_responses,
            "entropy": entropy,
            "texture_score": 1 - min(entropy / 8, 1.0)
        }
    
    def detect_geometric_distortions(self, image: np.ndarray) -> Dict:
        """
        Detect geometric distortions common in AI images
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze contour shapes
        irregular_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Very circular or very elongated shapes can be suspicious
                    if circularity < 0.3 or circularity > 0.95:
                        irregular_shapes += 1
        
        distortion_score = min(irregular_shapes / 10, 1.0)
        
        return {
            "num_contours": len(contours),
            "irregular_shapes": irregular_shapes,
            "distortion_score": distortion_score
        }
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Perform complete artifact analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Analysis results
        """
        # Detect face regions
        face_regions = self.detect_face_regions(image)
        
        # Analyze lighting
        lighting = self.analyze_lighting_consistency(image)
        
        # Analyze texture
        texture = self.detect_texture_irregularities(image)
        
        # Detect geometric distortions
        geometric = self.detect_geometric_distortions(image)
        
        # Calculate AI probability
        ai_indicators = []
        
        if not lighting["consistent"]:
            ai_indicators.append(lighting["lighting_score"])
        
        if texture["texture_score"] > 0.5:
            ai_indicators.append(texture["texture_score"])
        
        if geometric["distortion_score"] > 0.3:
            ai_indicators.append(geometric["distortion_score"])
        
        ai_probability = min(sum(ai_indicators) / max(len(ai_indicators), 1), 1.0)
        
        return {
            "ai_probability": ai_probability,
            "face_regions": [
                {"x": r.x, "y": r.y, "w": r.width, "h": r.height, "score": r.score}
                for r in face_regions
            ],
            "lighting_analysis": lighting,
            "texture_analysis": texture,
            "geometric_analysis": geometric,
            "is_ai_generated": ai_probability > 0.5
        }


def test_artifact_detector():
    """Test artifact detector"""
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    detector = ArtifactDetector()
    result = detector.analyze(test_image)
    
    print("Artifact Detection Result:")
    print(f"  AI Probability: {result['ai_probability']:.3f}")


if __name__ == "__main__":
    test_artifact_detector()