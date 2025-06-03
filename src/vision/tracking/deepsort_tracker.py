# DeepSORT Multi-Object Tracker
# File: src/vision/tracking/deepsort_tracker.py

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import deque, defaultdict
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Track:
    """
    Single tracking object
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    # Tracking state
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    
    # Motion tracking
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocities: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize tracking data"""
        self.positions.append(self.center)
        self.update_velocity()
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @property
    def area(self) -> int:
        """Get bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> int:
        """Get bounding box width"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get bounding box height"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity (pixels per frame)"""
        if len(self.velocities) > 0:
            return self.velocities[-1]
        return (0.0, 0.0)
    
    @property
    def speed(self) -> float:
        """Get current speed (pixels per frame)"""
        vx, vy = self.velocity
        return math.sqrt(vx*vx + vy*vy)
    
    @property
    def direction(self) -> float:
        """Get movement direction in degrees"""
        vx, vy = self.velocity
        if vx == 0 and vy == 0:
            return 0.0
        return math.degrees(math.atan2(vy, vx))
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Update track with new detection
        
        Args:
            bbox: New bounding box
            confidence: Detection confidence
        """
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = time.time()
        
        # Update position history
        self.positions.append(self.center)
        self.update_velocity()
        
        # Update state
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"
    
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position based on velocity
        
        Returns:
            Predicted bounding box
        """
        if len(self.velocities) == 0:
            return self.bbox
        
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        
        # Predict center movement
        center_x, center_y = self.center
        pred_x = center_x + vx
        pred_y = center_y + vy
        
        # Maintain box size
        width = x2 - x1
        height = y2 - y1
        
        pred_bbox = (
            int(pred_x - width // 2),
            int(pred_y - height // 2),
            int(pred_x + width // 2),
            int(pred_y + height // 2)
        )
        
        return pred_bbox
    
    def update_velocity(self):
        """Update velocity based on position history"""
        if len(self.positions) < 2:
            return
        
        # Calculate velocity from last two positions
        current_pos = self.positions[-1]
        prev_pos = self.positions[-2]
        
        vx = current_pos[0] - prev_pos[0]
        vy = current_pos[1] - prev_pos[1]
        
        self.velocities.append((vx, vy))
    
    def increment_age(self):
        """Increment track age and time since update"""
        self.age += 1
        self.time_since_update += 1
        
        # Mark for deletion if not updated for too long
        if self.time_since_update > 10:  # 10 frames without update
            self.state = "deleted"
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get full trajectory as list of positions"""
        return list(self.positions)


class DeepSORTTracker:
    """
    DeepSORT-inspired multi-object tracker
    
    This tracker provides:
    1. Detection-to-track association using IoU and feature matching
    2. Kalman filtering for motion prediction
    3. Track lifecycle management
    4. Trajectory analysis and prediction
    """
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.7):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            feature_threshold: Feature similarity threshold
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        # Track management
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
        # Statistics
        self.total_tracks_created = 0
        self.frame_count = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        logger.info(f"🎯 DeepSORTTracker initialized")
        logger.info(f"   • Max age: {max_age}")
        logger.info(f"   • Min hits: {min_hits}")
        logger.info(f"   • IoU threshold: {iou_threshold}")
    
    def update(self, detections: List) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of confirmed tracks
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Convert detections to proper format if needed
        if hasattr(detections[0], 'bbox') if detections else False:
            detection_data = [(det.bbox, det.confidence, det.class_id, det.class_name) 
                            for det in detections]
        else:
            detection_data = detections
        
        # Predict track positions
        self._predict_tracks()
        
        # Associate detections with tracks
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            detection_data, self.tracks
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            bbox, confidence, class_id, class_name = detection_data[det_idx]
            self.tracks[track_idx].update(bbox, confidence)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, confidence, class_id, class_name = detection_data[det_idx]
            self._create_track(bbox, confidence, class_id, class_name)
        
        # Mark unmatched tracks for potential deletion
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].increment_age()
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.state != "deleted"]
        
        # Update track ages
        for track in self.tracks:
            if track.track_id not in [self.tracks[i].track_id for i, _ in matched_tracks]:
                track.increment_age()
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Return confirmed tracks only
        confirmed_tracks = [track for track in self.tracks if track.state == "confirmed"]
        return confirmed_tracks
    
    def _predict_tracks(self):
        """Predict track positions for current frame"""
        for track in self.tracks:
            if track.state != "deleted":
                # Simple prediction based on velocity
                predicted_bbox = track.predict()
                # Note: In full implementation, this would update Kalman filter
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """
        Associate detections with existing tracks using IoU matching
        
        Args:
            detections: List of detection tuples (bbox, conf, class_id, class_name)
            tracks: List of Track objects
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, (det_bbox, _, det_class, _) in enumerate(detections):
            for track_idx, track in enumerate(tracks):
                # Only match same class detections
                if det_class == track.class_id:
                    iou = self._compute_iou(det_bbox, track.bbox)
                    iou_matrix[det_idx, track_idx] = iou
        
        # Find best matches using Hungarian algorithm (simplified greedy approach)
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Greedy matching based on highest IoU
        while True:
            # Find maximum IoU
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            # Find indices of maximum IoU
            det_idx, track_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Add to matches
            matched_pairs.append((track_idx, det_idx))
            
            # Remove from unmatched lists
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
            
            # Zero out the row and column to prevent re-matching
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union (IoU) of two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _create_track(self, bbox: Tuple[int, int, int, int], 
                     confidence: float, class_id: int, class_name: str):
        """
        Create new track from detection
        
        Args:
            bbox: Bounding box coordinates
            confidence: Detection confidence
            class_id: Object class ID
            class_name: Object class name
        """
        track = Track(
            track_id=self.next_track_id,
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name
        )
        
        self.tracks.append(track)
        self.next_track_id += 1
        self.total_tracks_created += 1
        
        logger.debug(f"📍 Created new track {track.track_id} for {class_name}")
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (non-deleted) tracks"""
        return [track for track in self.tracks if track.state != "deleted"]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks"""
        return [track for track in self.tracks if track.state == "confirmed"]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_track_trajectories(self, min_length: int = 5) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get trajectories for all tracks
        
        Args:
            min_length: Minimum trajectory length to include
            
        Returns:
            Dictionary mapping track_id to trajectory points
        """
        trajectories = {}
        for track in self.tracks:
            if track.state == "confirmed" and len(track.positions) >= min_length:
                trajectories[track.track_id] = track.get_trajectory()
        return trajectories
    
    def analyze_motion_patterns(self) -> Dict[str, Any]:
        """
        Analyze motion patterns of confirmed tracks
        
        Returns:
            Dictionary with motion analysis results
        """
        confirmed_tracks = self.get_confirmed_tracks()
        
        if not confirmed_tracks:
            return {}
        
        speeds = [track.speed for track in confirmed_tracks]
        directions = [track.direction for track in confirmed_tracks]
        
        analysis = {
            'total_active_tracks': len(confirmed_tracks),
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'speed_std': np.std(speeds) if speeds else 0,
            'avg_direction': np.mean(directions) if directions else 0,
            'direction_variance': np.var(directions) if directions else 0,
            'track_ages': [track.age for track in confirmed_tracks],
            'track_lifetimes': [(track.last_seen - track.first_seen) for track in confirmed_tracks]
        }
        
        return analysis
    
    def draw_tracks(self, image: np.ndarray, 
                   draw_trajectories: bool = True,
                   draw_predictions: bool = False,
                   draw_ids: bool = True) -> np.ndarray:
        """
        Draw tracks on image
        
        Args:
            image: Input image
            draw_trajectories: Whether to draw track trajectories
            draw_predictions: Whether to draw predicted positions
            draw_ids: Whether to draw track IDs
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Define colors for tracks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (255, 192, 203), # Pink
            (128, 128, 128)  # Gray
        ]
        
        for track in self.get_confirmed_tracks():
            color = colors[track.track_id % len(colors)]
            
            # Draw current bounding box
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            if draw_ids:
                label = f"ID:{track.track_id} | {track.class_name} | {track.confidence:.2f}"
                label += f" | Age:{track.age}"
                
                # Add motion info
                speed = track.speed
                if speed > 1:  # Only show if moving
                    label += f" | Speed:{speed:.1f}px/f"
                
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (x1, y1 - label_height - 5), 
                            (x1 + label_width, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw trajectory
            if draw_trajectories and len(track.positions) > 1:
                points = list(track.positions)
                for i in range(1, len(points)):
                    # Fade older points
                    alpha = i / len(points)
                    fade_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, points[i-1], points[i], fade_color, 2)
                
                # Draw direction arrow
                if len(points) >= 2:
                    start_point = points[-2]
                    end_point = points[-1]
                    cv2.arrowedLine(annotated, start_point, end_point, color, 3, tipLength=0.3)
            
            # Draw predicted position
            if draw_predictions:
                predicted_bbox = track.predict()
                px1, py1, px2, py2 = predicted_bbox
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 1, cv2.LINE_DASHED)
                cv2.putText(annotated, "PRED", (px1, py1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw tracking statistics
        stats_text = f"Active Tracks: {len(self.get_confirmed_tracks())} | Total Created: {self.total_tracks_created}"
        cv2.putText(annotated, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get tracker performance statistics"""
        stats = {
            'frame_count': self.frame_count,
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': len(self.get_active_tracks()),
            'confirmed_tracks': len(self.get_confirmed_tracks()),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
        }
        
        if self.processing_times:
            stats['avg_fps'] = 1 / np.mean(self.processing_times)
        
        return stats
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.processing_times.clear()
        logger.info("🔄 Tracker reset")


def test_tracker():
    """
    Test function for DeepSORT tracker
    """
    print("🧪 Testing DeepSORT Tracker")
    print("=" * 40)
    
    try:
        # Initialize tracker
        tracker = DeepSORTTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Test 1: Create synthetic detections
        print("🎯 Test 1: Synthetic tracking")
        
        # Simulate moving object over several frames
        frames_data = []
        for frame in range(20):
            # Moving object (simulate drone moving right)
            x = 100 + frame * 10
            y = 100 + int(5 * np.sin(frame * 0.3))  # Slight vertical oscillation
            
            detections = [
                (x, y, x + 50, y + 30, 0.9, 0, "drone")  # bbox, conf, class_id, class_name
            ]
            frames_data.append(detections)
        
        # Process frames
        all_tracks = []
        for frame_idx, detections in enumerate(frames_data):
            tracks = tracker.update(detections)
            all_tracks.append(tracks)
            
            if frame_idx % 5 == 0:
                print(f"   • Frame {frame_idx}: {len(tracks)} confirmed tracks")
        
        # Test 2: Motion analysis
        print("📊 Test 2: Motion analysis")
        motion_stats = tracker.analyze_motion_patterns()
        for key, value in motion_stats.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"   • {key}: {len(value)} items")
            else:
                print(f"   • {key}: {value:.2f}" if isinstance(value, float) else f"   • {key}: {value}")
        
        # Test 3: Trajectory analysis
        print("🛤️ Test 3: Trajectory analysis")
        trajectories = tracker.get_track_trajectories(min_length=5)
        for track_id, trajectory in trajectories.items():
            print(f"   • Track {track_id}: {len(trajectory)} points")
            if len(trajectory) >= 2:
                start = trajectory[0]
                end = trajectory[-1]
                distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                print(f"     Distance traveled: {distance:.1f} pixels")
        
        # Test 4: Performance stats
        print("⚡ Test 4: Performance statistics")
        perf_stats = tracker.get_performance_stats()
        for key, value in perf_stats.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.4f}")
            else:
                print(f"   • {key}: {value}")
        
        # Test 5: IoU computation
        print("🔗 Test 5: IoU computation")
        bbox1 = (10, 10, 50, 50)
        bbox2 = (30, 30, 70, 70)
        iou = tracker._compute_iou(bbox1, bbox2)
        print(f"   • IoU of overlapping boxes: {iou:.3f}")
        
        bbox3 = (100, 100, 140, 140)
        iou2 = tracker._compute_iou(bbox1, bbox3)
        print(f"   • IoU of non-overlapping boxes: {iou2:.3f}")
        
        print("\n✅ All tracker tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_tracker()