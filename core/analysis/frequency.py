"""DEEPVISION Analysis Module - Frequency Analysis

Analyzes images in the frequency domain to detect AI generation artifacts.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.fftpack import dct, idct
from scipy import ndimage
import cv2


class FrequencyAnalyzer:
    """
    Analyze frequency domain features to detect AI-generated images.
    
    AI-generated images often have characteristic patterns in the frequency
    domain due to the upsampling and interpolation operations in generative models.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.threshold = 0.5
    
    def compute_dct(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D DCT of image
        
        Args:
            image: Grayscale image
            
        Returns:
            DCT coefficients
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        return dct(dct(image.T, norm='ortho').T, norm='ortho')
    
    def analyze_frequency_pattern(self, dct_coeffs: np.ndarray) -> Dict:
        """
        Analyze DCT coefficients for AI generation patterns
        
        AI-generated images often show:
        - Higher energy in middle frequencies
        - Less natural roll-off in the spectrum
        - Specific artifacts in certain frequency bands
        """
        # Compute statistics
        mean_energy = np.mean(np.abs(dct_coeffs))
        std_energy = np.std(np.abs(dct_coeffs))
        
        # Analyze frequency bands
        h, w = dct_coeffs.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency region
        low_freq = dct_coeffs[:h//4, :w//4]
        low_energy = np.mean(np.abs(low_freq))
        
        # Mid frequency region
        mid_freq = dct_coeffs[h//4:3*h//4, w//4:3*w//4]
        mid_energy = np.mean(np.abs(mid_freq))
        
        # High frequency region
        high_freq = dct_coeffs[3*h//4:, 3*w//4:]
        high_energy = np.mean(np.abs(high_freq))
        
        # Calculate ratios
        mid_to_low_ratio = mid_energy / (low_energy + 1e-10)
        high_to_low_ratio = high_energy / (low_energy + 1e-10)
        
        return {
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "low_energy": low_energy,
            "mid_energy": mid_energy,
            "high_energy": high_energy,
            "mid_to_low_ratio": mid_to_low_ratio,
            "high_to_low_ratio": high_to_low_ratio
        }
    
    def detect_upscale_artifacts(self, image: np.ndarray) -> float:
        """
        Detect upscaling artifacts common in AI-generated images
        
        Returns:
            Score indicating likelihood of upscaling (0-1)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply unsharp mask to enhance edges
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # Compute Laplacian variance (edge sharpness)
        laplacian = cv2.Laplacian(sharpened, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # High variance can indicate synthetic sharpening
        normalized_score = min(laplacian_var / 1000, 1.0)
        
        return normalized_score
    
    def analyze_color_patterns(self, image: np.ndarray) -> Dict:
        """Analyze color channel correlations"""
        if len(image.shape) != 3:
            return {"correlation": 0}
        
        # Compute correlation between color channels
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        rg_correlation = np.corrcoef(r.flatten(), g.flatten())[0,1]
        rb_correlation = np.corrcoef(r.flatten(), b.flatten())[0,1]
        gb_correlation = np.corrcoef(g.flatten(), b.flatten())[0,1]
        
        avg_correlation = (abs(rg_correlation) + abs(rb_correlation) + abs(gb_correlation)) / 3
        
        return {
            "rg_correlation": rg_correlation,
            "rb_correlation": rb_correlation,
            "gb_correlation": gb_correlation,
            "avg_correlation": avg_correlation
        }
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Perform complete frequency analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute DCT
        dct_coeffs = self.compute_dct(gray)
        
        # Analyze frequency pattern
        freq_analysis = self.analyze_frequency_pattern(dct_coeffs)
        
        # Detect upscale artifacts
        upscale_score = self.detect_upscale_artifacts(image)
        
        # Analyze color patterns
        color_analysis = self.analyze_color_patterns(image)
        
        # Combine into AI probability
        # Higher mid-frequency energy and upscale artifacts indicate AI
        ai_indicators = []
        
        if freq_analysis["mid_to_low_ratio"] > 0.3:
            ai_indicators.append(0.3)
        if upscale_score > 0.5:
            ai_indicators.append(upscale_score * 0.4)
        
        # AI images often have less natural color correlation
        if color_analysis.get("avg_correlation", 0) > 0.99:
            ai_indicators.append(0.3)
        
        ai_probability = min(sum(ai_indicators) / max(len(ai_indicators), 1), 1.0)
        
        return {
            "ai_probability": ai_probability,
            "frequency_analysis": freq_analysis,
            "upscale_score": upscale_score,
            "color_analysis": color_analysis,
            "is_ai_generated": ai_probability > self.threshold
        }


def test_frequency_analyzer():
    """Test the frequency analyzer"""
    # Create synthetic test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(test_image)
    
    print("Frequency Analysis Result:")
    print(f"  AI Probability: {result['ai_probability']:.3f}")
    print(f"  Is AI Generated: {result['is_ai_generated']}")


if __name__ == "__main__":
    test_frequency_analyzer()