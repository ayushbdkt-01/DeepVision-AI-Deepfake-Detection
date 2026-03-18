"""DEEPVISION Video Preprocessing Module

Handles video loading, frame extraction, and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import subprocess
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video information container"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


class VideoProcessor:
    """
    Video preprocessing for deepfake detection
    
    Handles:
    - Video loading and validation
    - Frame extraction
    - Video information extraction
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'}
    MAX_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_DURATION = 300  # 5 minutes
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        frame_sample_rate: int = 1
    ):
        self.target_size = target_size
        self.frame_sample_rate = frame_sample_rate
    
    def get_video_info(self, video_path: Union[str, Path]) -> VideoInfo:
        """Get video information"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec
        )
    
    def validate_video(self, video_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate video file"""
        path = Path(video_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {path.suffix}"
        
        if path.stat().st_size > self.MAX_SIZE:
            return False, f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB"
        
        try:
            info = self.get_video_info(video_path)
            if info.duration > self.MAX_DURATION:
                return False, f"Video too long: {info.duration:.1f}s (max {self.MAX_DURATION}s)"
            return True, ""
        except Exception as e:
            return False, f"Invalid video: {str(e)}"
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        max_frames: int = 32,
        sample_rate: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            sample_rate: Sample every N frames (overrides frame_sample_rate)
            
        Returns:
            List of frames as numpy arrays
        """
        rate = sample_rate or self.frame_sample_rate
        frames = []
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def extract_key_frames(
        self,
        video_path: Union[str, Path],
        num_key_frames: int = 8
    ) -> List[np.ndarray]:
        """Extract key frames using scene detection"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        prev_frame = None
        frame_diffs = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = np.sum(np.abs(frame_rgb.astype(float) - prev_frame.astype(float)))
                frame_diffs.append((len(frames), diff))
            
            prev_frame = frame_rgb
        
        cap.release()
        
        # Select key frames based on scene changes
        if len(frame_diffs) <= num_key_frames:
            return self.extract_frames(video_path, max_frames=num_key_frames, sample_rate=1)
        
        # Select frames with largest differences
        frame_diffs.sort(key=lambda x: x[1], reverse=True)
        key_frame_indices = sorted([idx for idx, _ in frame_diffs[:num_key_frames]])
        
        # Extract those frames
        cap = cv2.VideoCapture(str(video_path))
        key_frames = []
        
        for i, frame_idx in enumerate(key_frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                key_frames.append(frame_rgb)
        
        cap.release()
        return key_frames
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess single frame"""
        # Resize
        frame = cv2.resize(frame, self.target_size)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def extract_audio_features(self, video_path: Union[str, Path]) -> dict:
        """Extract audio features for audio deepfake detection"""
        # Placeholder for audio analysis
        return {
            "has_audio": False,
            "sample_rate": 0,
            "duration": 0
        }
    
    def save_frames(self, frames: List[np.ndarray], output_dir: Path):
        """Save frames to directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"frame_{i:04d}.jpg"), frame_bgr)


def validate_video_file(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate video file"""
    processor = VideoProcessor()
    return processor.validate_video(file_path)