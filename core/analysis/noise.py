"""DEEPVISION Analysis Module - Noise Analysis

Analyzes noise patterns in images to detect AI generation artifacts.
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class NoiseAnalyzer:
    """
    Analyze noise patterns to detect AI-generated images.
    
    Natural images have characteristic noise patterns, while AI-generated
    images often show uniform or unusual noise characteristics.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.threshold = 0.5
    
    def estimate_noise_level(self, image: np.ndarray) -> Dict:
        """
        Estimate noise level in image using MAD estimator
        
        Returns:
            Dictionary with noise estimates
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use MAD (Median Absolute Deviation) estimator
        # This is robust to edges
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        
        # Convert to sigma estimate
        sigma = mad / 0.6745
        
        return {
            "sigma": sigma,
            "mean": np.mean(gray),
            "std": np.std(gray),
            "median": median
        }
    
    def analyze_noise_frequency(self, image: np.ndarray) -> Dict:
        """
        Analyze noise in frequency domain
        
        Natural images have noise with specific spectral characteristics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply high-pass filter to extract noise
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]) / 8.0
        
        noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Compute FFT
        f = np.fft.fft2(noise)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Compute radial profile
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Binned radial profile
        max_radius = min(center_x, center_y)
        bins = np.linspace(0, max_radius, 20)
        
        radial_profile = []
        for i in range(len(bins) - 1):
            mask = (radius >= bins[i]) & (radius < bins[i+1])
            radial_profile.append(np.mean(magnitude[mask]))
        
        # Compute spectral slope (noise characteristic)
        spectral_slope = 0
        if len(radial_profile) > 2:
            x_vals = np.arange(len(radial_profile))
            spectral_slope = np.polyfit(x_vals, np.log1p(radial_profile), 1)[0]
        
        return {
            "radial_profile": radial_profile,
            "spectral_slope": spectral_slope,
            "max_magnitude": np.max(magnitude),
            "mean_magnitude": np.mean(magnitude)
        }
    
    def detect_block_artifacts(self, image: np.ndarray) -> float:
        """
        Detect block artifacts common in compressed AI images
        
        Returns:
            Block artifact score (0-1)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute horizontal and vertical gradients
        grad_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        grad_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
        
        # Look for periodic patterns (block boundaries)
        block_size = 8
        
        # Horizontal block boundaries
        h, w = gray.shape
        h_blocks = h // block_size
        horizontal_score = 0
        
        for i in range(h_blocks):
            row = i * block_size
            if row + block_size < h:
                boundary = grad_x[row, :]
                variance = np.var(boundary)
                horizontal_score += variance
        
        horizontal_score /= h_blocks
        
        # Vertical block boundaries
        w_blocks = w // block_size
        vertical_score = 0
        
        for i in range(w_blocks):
            col = i * block_size
            if col + block_size < w:
                boundary = grad_y[:, col]
                variance = np.var(boundary)
                vertical_score += variance
        
        vertical_score /= w_blocks
        
        # Normalize
        artifact_score = (horizontal_score + vertical_score) / 2000
        return min(artifact_score, 1.0)
    
    def analyze_local_variance(self, image: np.ndarray, window_size: int = 16) -> Dict:
        """Analyze local variance patterns"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute local variance
        local_vars = ndimage.generic_filter(
            gray.astype(np.float32),
            np.var,
            size=window_size
        )
        
        return {
            "mean_variance": np.mean(local_vars),
            "std_variance": np.std(local_vars),
            "max_variance": np.max(local_vars),
            "min_variance": np.min(local_vars),
            "variance_uniformity": np.std(local_vars) / (np.mean(local_vars) + 1e-10)
        }
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Perform complete noise analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Analysis results
        """
        # Estimate noise level
        noise_est = self.estimate_noise_level(image)
        
        # Analyze noise frequency
        freq_analysis = self.analyze_noise_frequency(image)
        
        # Detect block artifacts
        block_score = self.detect_block_artifacts(image)
        
        # Analyze local variance
        variance_analysis = self.analyze_local_variance(image)
        
        # Compute AI probability
        ai_indicators = []
        
        # Unusually low noise can indicate AI generation
        if noise_est["sigma"] < 2.0:
            ai_indicators.append(0.3)
        
        # Unusual spectral characteristics
        if abs(freq_analysis["spectral_slope"]) < 0.1:
            ai_indicators.append(0.2)
        
        # High block artifacts
        if block_score > 0.3:
            ai_indicators.append(block_score)
        
        # Unusual variance patterns
        if variance_analysis["variance_uniformity"] < 0.5:
            ai_indicators.append(0.3)
        
        ai_probability = min(sum(ai_indicators) / max(len(ai_indicators), 1), 1.0)
        
        return {
            "ai_probability": ai_probability,
            "noise_estimation": noise_est,
            "frequency_analysis": freq_analysis,
            "block_artifact_score": block_score,
            "variance_analysis": variance_analysis,
            "is_ai_generated": ai_probability > self.threshold
        }


def test_noise_analyzer():
    """Test the noise analyzer"""
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    analyzer = NoiseAnalyzer()
    result = analyzer.analyze(test_image)
    
    print("Noise Analysis Result:")
    print(f"  AI Probability: {result['ai_probability']:.3f}")
    print(f"  Block Artifact Score: {result['block_artifact_score']:.3f}")


if __name__ == "__main__":
    test_noise_analyzer()