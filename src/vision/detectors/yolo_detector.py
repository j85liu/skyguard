# YOLO Drone Detector - Core Detection Engine
# File: src/vision/detectors/yolo_detector.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """
    Single detection result
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    timestamp: Optional[float] = None
    
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

class YOLODetector:
    """
    YOLO-based drone detection engine
    
    This class provides:
    1. Model loading and initialization
    2. Real-time inference on images/video streams
    3. Detection result processing and filtering
    4. Performance monitoring and optimization
    """
    
    def __init__(self, 
                 model_path: str = "yolo11n.pt",
                 confidence_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 device: str = "auto",
                 imgsz: int = 640):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            imgsz: Input image size
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.imgsz = imgsz
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        self.frame_count = 0
        
        # Load model
        self.model = self._load_model()
        
        # Get class names
        self.class_names = self.model.names
        logger.info(f"📊 Loaded {len(self.class_names)} classes: {list(self.class_names.values())}")
        
        logger.info(f"🚀 YOLODetector initialized")
        logger.info(f"   • Model: {model_path}")
        logger.info(f"   • Device: {self.device}")
        logger.info(f"   • Confidence: {confidence_threshold}")
        logger.info(f"   • Image size: {imgsz}")
    
    def _load_model(self) -> YOLO:
        """
        Load YOLO model with error handling
        
        Returns:
            YOLO: Loaded model
        """
        try:
            logger.info(f"🤖 Loading model: {self.model_path}")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                logger.info("📥 Downloading pretrained model...")
            
            # Load model
            model = YOLO(self.model_path)
            
            # Move to device
            model.to(self.device)
            
            # Set model parameters
            model.overrides['conf'] = self.confidence_threshold
            model.overrides['iou'] = self.nms_threshold
            model.overrides['imgsz'] = self.imgsz
            model.overrides['verbose'] = False  # Reduce logging noise
            
            logger.info(f"✅ Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def detect(self, 
               image: np.ndarray, 
               return_raw: bool = False) -> List[Detection]:
        """
        Detect drones in a single image
        
        Args:
            image: Input image (BGR format)
            return_raw: Whether to return raw YOLO results
            
        Returns:
            List of Detection objects
        """
        if image is None or image.size == 0:
            logger.warning("⚠️ Empty image provided")
            return []
        
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                imgsz=self.imgsz,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0])
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_detections += len(detections)
            self.frame_count += 1
            
            # Keep only recent timing data
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-50:]
            
            if return_raw:
                return detections, results[0]
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return []
    
    def _process_results(self, result) -> List[Detection]:
        """
        Process YOLO results into Detection objects
        
        Args:
            result: YOLO result object
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract data
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        current_time = time.time()
        
        # Create Detection objects
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Convert to integers
            bbox = tuple(map(int, box))
            
            # Get class name
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            detection = Detection(
                bbox=bbox,
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_name,
                timestamp=current_time
            )
            
            detections.append(detection)
        
        return detections
    
    def detect_video_stream(self, 
                           source: Any,
                           max_frames: Optional[int] = None,
                           show_display: bool = True,
                           save_path: Optional[str] = None) -> List[List[Detection]]:
        """
        Detect drones in video stream
        
        Args:
            source: Video source (file path, webcam index, URL)
            max_frames: Maximum frames to process (None = all)
            show_display: Whether to display results
            save_path: Path to save output video
            
        Returns:
            List of detection lists (one per frame)
        """
        logger.info(f"🎥 Starting video detection: {source}")
        
        # Open video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"❌ Could not open video source: {source}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📺 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check frame limit
                if max_frames and frame_num >= max_frames:
                    break
                
                # Detect drones
                detections = self.detect(frame)
                all_detections.append(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add performance info
                if len(self.inference_times) > 0:
                    avg_time = np.mean(self.inference_times[-10:])
                    fps_text = f"FPS: {1/avg_time:.1f} | Detections: {len(detections)}"
                    cv2.putText(annotated_frame, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show display
                if show_display:
                    cv2.imshow('Drone Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                frame_num += 1
                
                # Progress logging
                if frame_num % 100 == 0:
                    logger.info(f"🎬 Processed {frame_num}/{total_frames if total_frames > 0 else '?'} frames")
        
        except KeyboardInterrupt:
            logger.info("⚠️ Video processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
        
        logger.info(f"✅ Video processing complete: {frame_num} frames")
        return all_detections
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Detection],
                       draw_confidence: bool = True,
                       draw_class: bool = True) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: List of detections
            draw_confidence: Whether to draw confidence scores
            draw_class: Whether to draw class names
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Define colors for different classes
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
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = []
            if draw_class:
                label_parts.append(detection.class_name)
            if draw_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            if detection.track_id is not None:
                label_parts.append(f"ID:{detection.track_id}")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            if label:
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, y1 - label_height - 10), 
                            (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            center = detection.center
            cv2.circle(annotated, center, 3, color, -1)
        
        return annotated
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        recent_times = self.inference_times[-50:] if len(self.inference_times) > 50 else self.inference_times
        
        stats = {
            'avg_inference_time': np.mean(recent_times),
            'max_inference_time': np.max(recent_times),
            'min_inference_time': np.min(recent_times),
            'avg_fps': 1 / np.mean(recent_times),
            'total_frames': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / max(self.frame_count, 1),
            'device': self.device,
            'model_path': self.model_path
        }
        
        return stats
    
    def update_thresholds(self, 
                         confidence: Optional[float] = None,
                         nms: Optional[float] = None):
        """
        Update detection thresholds
        
        Args:
            confidence: New confidence threshold
            nms: New NMS threshold
        """
        if confidence is not None:
            self.confidence_threshold = confidence
            self.model.overrides['conf'] = confidence
            logger.info(f"📊 Updated confidence threshold: {confidence}")
        
        if nms is not None:
            self.nms_threshold = nms
            self.model.overrides['iou'] = nms
            logger.info(f"📊 Updated NMS threshold: {nms}")
    
    def benchmark(self, 
                  test_image: Optional[np.ndarray] = None,
                  num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark detector performance
        
        Args:
            test_image: Image to test with (creates random if None)
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"🏁 Running performance benchmark ({num_runs} runs)...")
        
        # Create test image if not provided
        if test_image is None:
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup runs
        for _ in range(5):
            self.detect(test_image)
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start = time.time()
            detections = self.detect(test_image)
            end = time.time()
            times.append(end - start)
            
            if (i + 1) % 20 == 0:
                logger.info(f"   • Progress: {i + 1}/{num_runs}")
        
        # Calculate statistics
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_fps': 1 / np.mean(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99)
        }
        
        logger.info("📊 Benchmark Results:")
        logger.info(f"   • Average: {results['avg_time']*1000:.1f}ms ({results['avg_fps']:.1f} FPS)")
        logger.info(f"   • Min/Max: {results['min_time']*1000:.1f}ms / {results['max_time']*1000:.1f}ms")
        logger.info(f"   • P95/P99: {results['p95_time']*1000:.1f}ms / {results['p99_time']*1000:.1f}ms")
        
        return results


def test_detector():
    """
    Test function for YOLO detector
    """
    print("🧪 Testing YOLO Detector")
    print("=" * 40)
    
    try:
        # Initialize detector
        detector = YOLODetector(
            model_path="yolo11n.pt",  # Will download if not exists
            confidence_threshold=0.3,
            device="auto"
        )
        
        # Test 1: Create test image
        print("🖼️ Test 1: Synthetic image detection")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_image)
        print(f"   • Detections: {len(detections)}")
        
        # Test 2: Performance benchmark
        print("🏁 Test 2: Performance benchmark")
        benchmark_results = detector.benchmark(test_image, num_runs=10)
        print(f"   • Average FPS: {benchmark_results['avg_fps']:.1f}")
        
        # Test 3: Webcam test (if available)
        print("📹 Test 3: Webcam test (press 'q' to quit)")
        try:
            # Try webcam for a few frames
            detections_list = detector.detect_video_stream(
                source=0,  # Webcam
                max_frames=10,
                show_display=True
            )
            print(f"   • Processed {len(detections_list)} frames")
        except Exception as e:
            print(f"   • Webcam test skipped: {e}")
        
        # Test 4: Performance stats
        print("📊 Test 4: Performance statistics")
        stats = detector.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.3f}")
            else:
                print(f"   • {key}: {value}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_detector()