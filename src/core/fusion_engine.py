# SkyGuard Fusion Engine - Main Processing Pipeline
# File: src/core/fusion_engine.py

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """
    Multi-modal detection result
    """
    timestamp: float
    frame_id: int
    
    # Vision results
    vision_detections: List = field(default_factory=list)
    vision_tracks: List = field(default_factory=list)
    
    # RF results (placeholder for future implementation)
    rf_detections: List = field(default_factory=list)
    rf_signals: Dict = field(default_factory=dict)
    
    # Acoustic results (placeholder for future implementation)
    acoustic_detections: List = field(default_factory=list)
    acoustic_features: Dict = field(default_factory=dict)
    
    # Fusion results
    fused_detections: List = field(default_factory=list)
    threat_level: str = "none"  # none, low, medium, high, critical
    confidence_score: float = 0.0
    
    # Performance metrics
    processing_time: float = 0.0
    vision_fps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'vision_detections': len(self.vision_detections),
            'vision_tracks': len(self.vision_tracks),
            'rf_detections': len(self.rf_detections),
            'acoustic_detections': len(self.acoustic_detections),
            'fused_detections': len(self.fused_detections),
            'threat_level': self.threat_level,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'vision_fps': self.vision_fps
        }

@dataclass 
class ThreatAssessment:
    """
    Threat assessment result
    """
    track_id: int
    threat_level: str
    confidence: float
    reasons: List[str]
    risk_factors: Dict[str, float]
    
    # Behavioral analysis
    speed_anomaly: bool = False
    direction_anomaly: bool = False
    restricted_area: bool = False
    formation_flying: bool = False
    
    # Temporal analysis
    persistence: float = 0.0  # How long has this threat been present
    escalation: float = 0.0   # Is threat level increasing

class SkyGuardFusionEngine:
    """
    Main SkyGuard processing pipeline integrating all detection modalities
    
    This engine provides:
    1. Multi-modal sensor integration
    2. Real-time processing pipeline
    3. Threat assessment and alerting
    4. Performance monitoring
    5. Data logging and analysis
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_vision: bool = True,
                 enable_rf: bool = False,
                 enable_acoustic: bool = False):
        """
        Initialize fusion engine
        
        Args:
            config_path: Path to configuration file
            enable_vision: Enable computer vision pipeline
            enable_rf: Enable RF signal processing
            enable_acoustic: Enable acoustic processing
        """
        # Configuration
        self.config = self._load_config(config_path)
        
        # Module enablement
        self.enable_vision = enable_vision
        self.enable_rf = enable_rf
        self.enable_acoustic = enable_acoustic
        
        # Processing components
        self.vision_detector = None
        self.vision_tracker = None
        self.rf_processor = None
        self.acoustic_processor = None
        
        # Processing queues and threads
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.is_running = False
        
        # State management
        self.frame_id = 0
        self.detection_history = deque(maxlen=1000)
        self.threat_history = deque(maxlen=500)
        self.performance_stats = {}
        
        # Callbacks
        self.detection_callbacks: List[Callable] = []
        self.threat_callbacks: List[Callable] = []
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"🚀 SkyGuard Fusion Engine initialized")
        logger.info(f"   • Vision: {'✅' if enable_vision else '❌'}")
        logger.info(f"   • RF: {'✅' if enable_rf else '❌'}")
        logger.info(f"   • Acoustic: {'✅' if enable_acoustic else '❌'}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'vision': {
                'model_path': 'yolo11n.pt',
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45,
                'imgsz': 640
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            },
            'threat_assessment': {
                'speed_threshold': 50.0,  # pixels per frame
                'restricted_zones': [],
                'formation_distance': 100,  # pixels
                'persistence_threshold': 30  # frames
            },
            'performance': {
                'target_fps': 30,
                'max_processing_time': 0.1  # seconds
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
                logger.info(f"📖 Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config: {e}, using defaults")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize processing components based on enabled modules"""
        
        # Initialize vision components
        if self.enable_vision:
            try:
                # Import here to avoid dependencies if not used
                from src.vision.detectors.yolo_detector import YOLODetector
                from src.vision.tracking.deepsort_tracker import DeepSORTTracker
                
                vision_config = self.config['vision']
                self.vision_detector = YOLODetector(
                    model_path=vision_config['model_path'],
                    confidence_threshold=vision_config['confidence_threshold'],
                    nms_threshold=vision_config['nms_threshold'],
                    imgsz=vision_config['imgsz']
                )
                
                tracking_config = self.config['tracking']
                self.vision_tracker = DeepSORTTracker(
                    max_age=tracking_config['max_age'],
                    min_hits=tracking_config['min_hits'],
                    iou_threshold=tracking_config['iou_threshold']
                )
                
                logger.info("✅ Vision components initialized")
                
            except ImportError as e:
                logger.error(f"❌ Failed to import vision components: {e}")
                self.enable_vision = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize vision: {e}")
                self.enable_vision = False
        
        # Initialize RF components (placeholder)
        if self.enable_rf:
            logger.info("📡 RF processing not yet implemented")
            self.enable_rf = False
        
        # Initialize acoustic components (placeholder)
        if self.enable_acoustic:
            logger.info("🔊 Acoustic processing not yet implemented")
            self.enable_acoustic = False
    
    def start_processing(self):
        """Start the processing pipeline"""
        if self.is_running:
            logger.warning("⚠️ Processing already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("🎬 Processing pipeline started")
    
    def stop_processing(self):
        """Stop the processing pipeline"""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("🛑 Processing pipeline stopped")
    
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single frame through all enabled modalities
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            DetectionResult with all processing results
        """
        start_time = time.time()
        
        # Create result object
        result = DetectionResult(
            timestamp=time.time(),
            frame_id=self.frame_id
        )
        
        # Vision processing
        if self.enable_vision and self.vision_detector:
            try:
                # Detect objects
                detections = self.vision_detector.detect(frame)
                result.vision_detections = detections
                
                # Track objects
                if self.vision_tracker:
                    tracks = self.vision_tracker.update(detections)
                    result.vision_tracks = tracks
                
                # Calculate vision FPS
                vision_stats = self.vision_detector.get_performance_stats()
                result.vision_fps = vision_stats.get('avg_fps', 0)
                
            except Exception as e:
                logger.error(f"❌ Vision processing failed: {e}")
        
        # RF processing (placeholder)
        if self.enable_rf:
            # TODO: Implement RF signal processing
            pass
        
        # Acoustic processing (placeholder)
        if self.enable_acoustic:
            # TODO: Implement acoustic processing
            pass
        
        # Fusion and threat assessment
        result.fused_detections, result.threat_level, result.confidence_score = self._perform_fusion(result)
        
        # Performance tracking
        result.processing_time = time.time() - start_time
        
        # Store in history
        self.detection_history.append(result)
        
        # Threat assessment
        threats = self._assess_threats(result)
        if threats:
            self.threat_history.extend(threats)
            # Trigger threat callbacks
            for callback in self.threat_callbacks:
                try:
                    callback(threats)
                except Exception as e:
                    logger.error(f"❌ Threat callback failed: {e}")
        
        # Trigger detection callbacks
        for callback in self.detection_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"❌ Detection callback failed: {e}")
        
        self.frame_id += 1
        return result
    
    def _processing_loop(self):
        """Main processing loop for threaded operation"""
        logger.info("🔄 Processing loop started")
        
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                result = self.process_frame(frame)
                
                # Put result in output queue
                try:
                    self.result_queue.put(result, timeout=0.1)
                except queue.Full:
                    logger.warning("⚠️ Result queue full, dropping result")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Processing loop error: {e}")
        
        logger.info("🔄 Processing loop ended")
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add frame to processing queue
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame was added, False if queue is full
        """
        try:
            self.frame_queue.put(frame, timeout=0.01)
            return True
        except queue.Full:
            logger.warning("⚠️ Frame queue full, dropping frame")
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[DetectionResult]:
        """
        Get processing result from queue
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            DetectionResult or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _perform_fusion(self, result: DetectionResult) -> tuple:
        """
        Perform multi-modal fusion of detection results
        
        Args:
            result: Detection result with individual modality results
            
        Returns:
            Tuple of (fused_detections, threat_level, confidence_score)
        """
        fused_detections = []
        
        # Currently only vision is implemented
        if result.vision_tracks:
            fused_detections = result.vision_tracks.copy()
        elif result.vision_detections:
            fused_detections = result.vision_detections.copy()
        
        # Calculate overall threat level
        threat_level = "none"
        confidence_score = 0.0
        
        if fused_detections:
            # Simple threat assessment based on number of detections
            num_detections = len(fused_detections)
            
            if num_detections >= 5:
                threat_level = "critical"
                confidence_score = 0.9
            elif num_detections >= 3:
                threat_level = "high"
                confidence_score = 0.8
            elif num_detections >= 2:
                threat_level = "medium"
                confidence_score = 0.6
            elif num_detections >= 1:
                threat_level = "low"
                confidence_score = 0.4
        
        return fused_detections, threat_level, confidence_score
    
    def _assess_threats(self, result: DetectionResult) -> List[ThreatAssessment]:
        """
        Assess threat level for each detected object
        
        Args:
            result: Detection result
            
        Returns:
            List of threat assessments
        """
        threats = []
        
        if not result.vision_tracks:
            return threats
        
        threat_config = self.config['threat_assessment']
        
        for track in result.vision_tracks:
            # Initialize threat assessment
            threat = ThreatAssessment(
                track_id=track.track_id,
                threat_level="low",
                confidence=0.5,
                reasons=[],
                risk_factors={}
            )
            
            # Speed analysis
            speed = track.speed if hasattr(track, 'speed') else 0
            if speed > threat_config['speed_threshold']:
                threat.speed_anomaly = True
                threat.reasons.append(f"High speed: {speed:.1f} px/frame")
                threat.risk_factors['speed'] = min(1.0, speed / threat_config['speed_threshold'])
            
            # Persistence analysis
            age = track.age if hasattr(track, 'age') else 0
            if age > threat_config['persistence_threshold']:
                threat.persistence = min(1.0, age / threat_config['persistence_threshold'])
                threat.reasons.append(f"Persistent presence: {age} frames")
                threat.risk_factors['persistence'] = threat.persistence
            
            # Formation flying detection (simplified)
            nearby_tracks = [t for t in result.vision_tracks 
                           if t.track_id != track.track_id and 
                           self._calculate_distance(track.center, t.center) < threat_config['formation_distance']]
            
            if len(nearby_tracks) >= 2:
                threat.formation_flying = True
                threat.reasons.append(f"Formation flying: {len(nearby_tracks)} nearby objects")
                threat.risk_factors['formation'] = min(1.0, len(nearby_tracks) / 3.0)
            
            # Calculate overall threat level
            risk_score = sum(threat.risk_factors.values())
            
            if risk_score >= 2.0:
                threat.threat_level = "critical"
                threat.confidence = 0.9
            elif risk_score >= 1.5:
                threat.threat_level = "high"
                threat.confidence = 0.8
            elif risk_score >= 1.0:
                threat.threat_level = "medium"
                threat.confidence = 0.7
            elif risk_score >= 0.5:
                threat.threat_level = "low"
                threat.confidence = 0.6
            else:
                threat.threat_level = "none"
                threat.confidence = 0.3
            
            if threat.threat_level != "none":
                threats.append(threat)
        
        return threats
    
    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def process_video_stream(self, 
                           source: Any,
                           max_frames: Optional[int] = None,
                           show_display: bool = True,
                           save_path: Optional[str] = None) -> List[DetectionResult]:
        """
        Process video stream through complete pipeline
        
        Args:
            source: Video source (file, webcam, URL)
            max_frames: Maximum frames to process
            show_display: Whether to show real-time display
            save_path: Path to save annotated video
            
        Returns:
            List of detection results
        """
        logger.info(f"🎥 Starting video stream processing: {source}")
        
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
        
        all_results = []
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check frame limit
                if max_frames and frame_num >= max_frames:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                all_results.append(result)
                
                # Create visualization
                annotated_frame = self.draw_results(frame, result)
                
                # Show display
                if show_display:
                    cv2.imshow('SkyGuard Detection', annotated_frame)
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
        return all_results
    
    def draw_results(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            result: Detection result
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw vision results
        if self.enable_vision and self.vision_tracker and result.vision_tracks:
            annotated = self.vision_tracker.draw_tracks(
                annotated, 
                draw_trajectories=True,
                draw_ids=True
            )
        elif result.vision_detections and self.vision_detector:
            annotated = self.vision_detector.draw_detections(
                annotated, 
                result.vision_detections
            )
        
        # Draw threat level indicator
        threat_colors = {
            "none": (0, 255, 0),      # Green
            "low": (0, 255, 255),     # Yellow
            "medium": (0, 165, 255),  # Orange
            "high": (0, 0, 255),      # Red
            "critical": (128, 0, 128) # Purple
        }
        
        threat_color = threat_colors.get(result.threat_level, (255, 255, 255))
        
        # Threat level banner
        banner_height = 60
        cv2.rectangle(annotated, (0, 0), (image.shape[1], banner_height), threat_color, -1)
        
        # Threat text
        threat_text = f"THREAT LEVEL: {result.threat_level.upper()}"
        if result.confidence_score > 0:
            threat_text += f" ({result.confidence_score:.2f})"
        
        cv2.putText(annotated, threat_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Performance info
        perf_y = banner_height + 30
        perf_info = [
            f"Frame: {result.frame_id}",
            f"Processing: {result.processing_time*1000:.1f}ms",
            f"Vision FPS: {result.vision_fps:.1f}",
            f"Detections: {len(result.vision_detections)}",
            f"Tracks: {len(result.vision_tracks)}"
        ]
        
        for i, info in enumerate(perf_info):
            cv2.putText(annotated, info, (10, perf_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timeline of recent detections
        timeline_y = image.shape[0] - 50
        timeline_width = min(500, image.shape[1] - 20)
        
        # Draw timeline background
        cv2.rectangle(annotated, (10, timeline_y), (10 + timeline_width, timeline_y + 40), 
                     (50, 50, 50), -1)
        
        # Draw recent detection history
        if len(self.detection_history) > 1:
            recent_results = list(self.detection_history)[-50:]  # Last 50 frames
            
            for i, hist_result in enumerate(recent_results):
                x_pos = 10 + int((i / len(recent_results)) * timeline_width)
                
                # Color based on number of detections
                if len(hist_result.fused_detections) > 0:
                    intensity = min(255, len(hist_result.fused_detections) * 50)
                    color = (0, intensity, 255)  # Orange-red gradient
                    cv2.line(annotated, (x_pos, timeline_y + 5), (x_pos, timeline_y + 35), color, 2)
        
        cv2.putText(annotated, "Detection Timeline", (10, timeline_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def add_detection_callback(self, callback: Callable[[DetectionResult], None]):
        """Add callback for detection events"""
        self.detection_callbacks.append(callback)
        logger.info(f"📡 Added detection callback: {callback.__name__}")
    
    def add_threat_callback(self, callback: Callable[[List[ThreatAssessment]], None]):
        """Add callback for threat events"""
        self.threat_callbacks.append(callback)
        logger.info(f"🚨 Added threat callback: {callback.__name__}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.detection_history:
            return {}
        
        recent_results = list(self.detection_history)[-100:]  # Last 100 frames
        
        processing_times = [r.processing_time for r in recent_results]
        vision_fps_values = [r.vision_fps for r in recent_results if r.vision_fps > 0]
        detection_counts = [len(r.fused_detections) for r in recent_results]
        
        summary = {
            'total_frames_processed': self.frame_id,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'avg_vision_fps': np.mean(vision_fps_values) if vision_fps_values else 0,
            'avg_detections_per_frame': np.mean(detection_counts) if detection_counts else 0,
            'total_threats_detected': len(self.threat_history),
            'queue_sizes': {
                'frame_queue': self.frame_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'enabled_modules': {
                'vision': self.enable_vision,
                'rf': self.enable_rf,
                'acoustic': self.enable_acoustic
            }
        }
        
        # Add component-specific stats
        if self.vision_detector:
            vision_stats = self.vision_detector.get_performance_stats()
            summary['vision_stats'] = vision_stats
        
        if self.vision_tracker:
            tracking_stats = self.vision_tracker.get_performance_stats()
            summary['tracking_stats'] = tracking_stats
        
        return summary
    
    def save_detection_log(self, file_path: str):
        """Save detection history to file"""
        try:
            log_data = {
                'metadata': {
                    'total_frames': self.frame_id,
                    'created_at': time.time(),
                    'config': self.config
                },
                'detections': [result.to_dict() for result in self.detection_history],
                'threats': [
                    {
                        'track_id': threat.track_id,
                        'threat_level': threat.threat_level,
                        'confidence': threat.confidence,
                        'reasons': threat.reasons,
                        'risk_factors': threat.risk_factors
                    }
                    for threat in self.threat_history
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"💾 Detection log saved to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save detection log: {e}")


def test_fusion_engine():
    """
    Test function for fusion engine
    """
    print("🧪 Testing SkyGuard Fusion Engine")
    print("=" * 50)
    
    try:
        # Test 1: Initialize engine
        print("🚀 Test 1: Engine initialization")
        engine = SkyGuardFusionEngine(
            enable_vision=True,
            enable_rf=False,
            enable_acoustic=False
        )
        print(f"   • Vision enabled: {engine.enable_vision}")
        print(f"   • Components initialized: {engine.vision_detector is not None}")
        
        # Test 2: Process synthetic frame
        print("🖼️ Test 2: Single frame processing")
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = engine.process_frame(test_frame)
        
        print(f"   • Frame ID: {result.frame_id}")
        print(f"   • Processing time: {result.processing_time*1000:.1f}ms")
        print(f"   • Threat level: {result.threat_level}")
        print(f"   • Vision detections: {len(result.vision_detections)}")
        
        # Test 3: Threaded processing
        print("🔄 Test 3: Threaded processing")
        engine.start_processing()
        
        # Add some test frames
        for i in range(5):
            test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            success = engine.add_frame(test_frame)
            print(f"   • Frame {i} added: {success}")
            time.sleep(0.1)
        
        # Get results
        results_received = 0
        for _ in range(5):
            result = engine.get_result(timeout=2.0)
            if result:
                results_received += 1
            else:
                break
        
        print(f"   • Results received: {results_received}")
        engine.stop_processing()
        
        # Test 4: Performance summary
        print("📊 Test 4: Performance summary")
        summary = engine.get_performance_summary()
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"   • {key}: {len(value)} items")
            elif isinstance(value, float):
                print(f"   • {key}: {value:.4f}")
            else:
                print(f"   • {key}: {value}")
        
        # Test 5: Callbacks
        print("📡 Test 5: Callback system")
        
        detection_count = 0
        threat_count = 0
        
        def detection_callback(result):
            nonlocal detection_count
            detection_count += 1
            print(f"     Detection callback: Frame {result.frame_id}")
        
        def threat_callback(threats):
            nonlocal threat_count
            threat_count += len(threats)
            print(f"     Threat callback: {len(threats)} threats")
        
        engine.add_detection_callback(detection_callback)
        engine.add_threat_callback(threat_callback)
        
        # Process a frame to trigger callbacks
        result = engine.process_frame(test_frame)
        print(f"   • Detection callbacks triggered: {detection_count}")
        print(f"   • Threat callbacks triggered: {threat_count}")
        
        # Test 6: Webcam test (if available)
        print("📹 Test 6: Webcam test (5 frames)")
        try:
            results = engine.process_video_stream(
                source=0,  # Webcam
                max_frames=5,
                show_display=True
            )
            print(f"   • Webcam frames processed: {len(results)}")
        except Exception as e:
            print(f"   • Webcam test skipped: {e}")
        
        print("\n✅ All fusion engine tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fusion engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_fusion_engine()