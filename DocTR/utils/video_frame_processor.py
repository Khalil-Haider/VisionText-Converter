# utils/video_processor.py
import cv2
import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class Frame:
    """Data class to store frame data"""
    image: np.ndarray
    timestamp: float

class video_frame_processor:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute a perceptual hash of the frame"""
        small_frame = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        hash_string = ''.join(['1' if pixel > mean else '0' for pixel in gray.flatten()])
        return hash_string

    def are_frames_similar(self, hash1: str, hash2: str) -> bool:
        """Compare two frame hashes"""
        if not hash1 or not hash2:
            return False
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        similarity = 1 - (hamming_distance / len(hash1))
        return similarity > self.similarity_threshold

    def extract_unique_frames(self, video_path: str, sample_rate: float = 1.0) -> List[Frame]:
        """Extract unique frames from video"""
        frames = []
        previous_hash = None
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                current_hash = self.compute_frame_hash(frame)
                
                if not previous_hash or not self.are_frames_similar(current_hash, previous_hash):
                    timestamp = frame_count / fps
                    frames.append(Frame(frame, timestamp))
                    previous_hash = current_hash

            frame_count += 1

        cap.release()
        return frames

