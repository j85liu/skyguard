# Complete Vision Pipeline Test
# File: tests/integration/test_vision_pipeline.py
# Run from project root: python tests/integration/test_vision_pipeline.py

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import our components
try:
    from vision.detectors.yolo_detector import YOLODetector, Detection
    from vision.tracking.deepsort_tracker import DeepSORTTracker, Track
    from core.fusion_engine import SkyGuardFusionEngine, DetectionResult
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from project root and components are available")
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class VisionPipelineTest:
    """
    Comprehensive test suite for the complete vision pipeline
    """
    
    def __init__(self):
        """Initialize test suite"""
        self.detector = None
        self.tracker = None
        self.fusion_engine = None
        self.test_results = {}
        
        print("🧪 Vision Pipeline Test Suite")
        print("=" * 60)
    
    def setup_components(self):
        """Setup all pipeline components"""
        print("🔧 Setting up components...")
        
        try:
            # Initialize detector
            self.detector = YOLODetector(
                model_path="yolo11n.pt",
                confidence_threshold=0.25,
                device="auto",
                imgsz=640
            )
            print("✅ YOLO Detector initialized")
            
            # Initialize tracker
            self.tracker = DeepSORTTracker(
                max_age=30,
                min_hits=3,
                iou_threshold=0.3
            )
            print("✅ DeepSORT Tracker initialized")
            
            # Initialize fusion engine
            self.fusion_engine = SkyGuardFusionEngine(
                enable_vision=True,
                enable_rf=False,
                enable_acoustic=False
            )
            print("✅ Fusion Engine initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Component setup failed: {e}")
            return False
    
    def test_individual_components(self):
        """Test each component individually"""
        print("\n🔍 Testing Individual Components")
        print("-" * 40)
        
        results = {}
        
        # Test 1: YOLO Detector
        print("1️⃣ Testing YOLO Detector...")
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run detection
            start_time = time.time()
            detections = self.detector.detect(test_image)
            detection_time = time.time() - start_time
            
            # Run benchmark
            benchmark_results = self.detector.benchmark(test_image, num_runs=10)
            
            results['detector'] = {
                'detection_time': detection_time,
                'detections_found': len(detections),
                'avg_fps': benchmark_results['avg_fps'],
                'avg_time_ms': benchmark_results['avg_time'] * 1000
            }
            
            print(f"   ✅ Detection time: {detection_time*1000:.1f}ms")
            print(f"   ✅ Average FPS: {benchmark_results['avg_fps']:.1f}")
            print(f"   ✅ Detections: {len(detections)}")
            
        except Exception as e:
            print(f"   ❌ Detector test failed: {e}")
            results['detector'] = {'error': str(e)}
        
        # Test 2: DeepSORT Tracker
        print("\n2️⃣ Testing DeepSORT Tracker...")
        try:
            # Create synthetic detection sequence
            synthetic_detections = []
            for frame in range(10):
                # Moving object
                x = 100 + frame * 10
                y = 100 + int(5 * np.sin(frame * 0.3))
                
                detection = Detection(
                    bbox=(x, y, x + 50, y + 30),
                    confidence=0.9,
                    class_id=0,
                    class_name="drone"
                )
                synthetic_detections.append([detection])
            
            # Process through tracker
            all_tracks = []
            for frame_detections in synthetic_detections:
                tracks = self.tracker.update(frame_detections)
                all_tracks.append(tracks)
            
            # Analyze results
            final_tracks = all_tracks[-1]
            total_tracks_created = self.tracker.total_tracks_created
            
            results['tracker'] = {
                'total_tracks_created': total_tracks_created,
                'final_confirmed_tracks': len(final_tracks),
                'track_persistence': len(final_tracks) > 0
            }
            
            print(f"   ✅ Total tracks created: {total_tracks_created}")
            print(f"   ✅ Final confirmed tracks: {len(final_tracks)}")
            print(f"   ✅ Track persistence: {len(final_tracks) > 0}")
            
        except Exception as e:
            print(f"   ❌ Tracker test failed: {e}")
            results['tracker'] = {'error': str(e)}
        
        # Test 3: Fusion Engine
        print("\n3️⃣ Testing Fusion Engine...")
        try:
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Process single frame
            start_time = time.time()
            result = self.fusion_engine.process_frame(test_image)
            processing_time = time.time() - start_time
            
            results['fusion'] = {
                'processing_time': processing_time,
                'frame_id': result.frame_id,
                'threat_level': result.threat_level,
                'vision_enabled': self.fusion_engine.enable_vision
            }
            
            print(f"   ✅ Processing time: {processing_time*1000:.1f}ms")
            print(f"   ✅ Frame ID: {result.frame_id}")
            print(f"   ✅ Threat level: {result.threat_level}")
            
        except Exception as e:
            print(f"   ❌ Fusion engine test failed: {e}")
            results['fusion'] = {'error': str(e)}
        
        self.test_results['individual_components'] = results
        return results
    
    def test_integrated_pipeline(self):
        """Test the complete integrated pipeline"""
        print("\n🔗 Testing Integrated Pipeline")
        print("-" * 40)
        
        try:
            # Create test video sequence
            frames = []
            for i in range(20):
                frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Add some moving objects (simulate drones)
                for j in range(2):  # 2 moving objects
                    x = 50 + i * 15 + j * 100
                    y = 50 + j * 200 + int(10 * np.sin(i * 0.2 + j))
                    
                    # Draw rectangle to simulate drone
                    cv2.rectangle(frame, (x, y), (x + 40, y + 25), (255, 255, 255), -1)
                
                frames.append(frame)
            
            print(f"🎬 Created {len(frames)} test frames")
            
            # Process through complete pipeline
            all_results = []
            processing_times = []
            
            for i, frame in enumerate(frames):
                start_time = time.time()
                result = self.fusion_engine.process_frame(frame)
                processing_time = time.time() - start_time
                
                all_results.append(result)
                processing_times.append(processing_time)
                
                if (i + 1) % 5 == 0:
                    print(f"   📋 Processed frame {i + 1}/{len(frames)}")
            
            # Analyze results
            total_detections = sum(len(r.vision_detections) for r in all_results)
            total_tracks = sum(len(r.vision_tracks) for r in all_results)
            avg_processing_time = np.mean(processing_times)
            max_processing_time = np.max(processing_times)
            
            # Threat analysis
            threat_levels = [r.threat_level for r in all_results]
            unique_threats = set(threat_levels)
            
            pipeline_results = {
                'frames_processed': len(all_results),
                'total_detections': total_detections,
                'total_tracks': total_tracks,
                'avg_processing_time': avg_processing_time,
                'max_processing_time': max_processing_time,
                'avg_fps': 1 / avg_processing_time if avg_processing_time > 0 else 0,
                'unique_threat_levels': list(unique_threats),
                'final_threat_level': all_results[-1].threat_level if all_results else 'none'
            }
            
            print(f"\n📊 Pipeline Results:")
            print(f"   • Frames processed: {pipeline_results['frames_processed']}")
            print(f"   • Total detections: {pipeline_results['total_detections']}")
            print(f"   • Total tracks: {pipeline_results['total_tracks']}")
            print(f"   • Average processing time: {avg_processing_time*1000:.1f}ms")
            print(f"   • Average FPS: {pipeline_results['avg_fps']:.1f}")
            print(f"   • Threat levels observed: {unique_threats}")
            
            self.test_results['integrated_pipeline'] = pipeline_results
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Integrated pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def test_performance_benchmarks(self):
        """Test performance under various conditions"""
        print("\n⚡ Performance Benchmarks")
        print("-" * 40)
        
        benchmarks = {}
        
        # Test different image sizes
        image_sizes = [(320, 320), (640, 640), (1280, 720)]
        
        for width, height in image_sizes:
            print(f"\n📏 Testing {width}x{height} images...")
            
            try:
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # Benchmark detector
                times = []
                for _ in range(10):
                    start = time.time()
                    detections = self.detector.detect(test_image)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                fps = 1 / avg_time if avg_time > 0 else 0
                
                benchmarks[f"{width}x{height}"] = {
                    'avg_time_ms': avg_time * 1000,
                    'fps': fps,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000
                }
                
                print(f"   ✅ Average: {avg_time*1000:.1f}ms ({fps:.1f} FPS)")
                print(f"   ✅ Min/Max: {np.min(times)*1000:.1f}ms / {np.max(times)*1000:.1f}ms")
                
            except Exception as e:
                print(f"   ❌ Benchmark failed for {width}x{height}: {e}")
                benchmarks[f"{width}x{height}"] = {'error': str(e)}
        
        self.test_results['performance_benchmarks'] = benchmarks
        return benchmarks
    
    def test_webcam_integration(self, duration: int = 10):
        """Test with real webcam if available"""
        print(f"\n📹 Webcam Integration Test ({duration}s)")
        print("-" * 40)
        
        try:
            # Try to open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("⚠️ Webcam not available, skipping test")
                return {'skipped': True, 'reason': 'No webcam available'}
            
            print("✅ Webcam opened successfully")
            
            # Test parameters
            start_time = time.time()
            frame_count = 0
            detection_count = 0
            track_count = 0
            processing_times = []
            
            print(f"🎬 Recording for {duration} seconds (press 'q' to quit early)...")
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through pipeline
                proc_start = time.time()
                result = self.fusion_engine.process_frame(frame)
                proc_time = time.time() - proc_start
                
                processing_times.append(proc_time)
                frame_count += 1
                detection_count += len(result.vision_detections)
                track_count += len(result.vision_tracks)
                
                # Draw results
                annotated_frame = self.fusion_engine.draw_results(frame, result)
                
                # Add test info
                test_info = f"Test Frame: {frame_count} | Processing: {proc_time*1000:.1f}ms"
                cv2.putText(annotated_frame, test_info, (10, annotated_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('SkyGuard Webcam Test', annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Calculate statistics
            actual_duration = time.time() - start_time
            avg_fps = frame_count / actual_duration if actual_duration > 0 else 0
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            
            webcam_results = {
                'frames_processed': frame_count,
                'duration': actual_duration,
                'avg_fps': avg_fps,
                'avg_processing_time': avg_processing_time,
                'total_detections': detection_count,
                'total_tracks': track_count,
                'avg_detections_per_frame': detection_count / max(frame_count, 1),
                'avg_tracks_per_frame': track_count / max(frame_count, 1)
            }
            
            print(f"\n📊 Webcam Test Results:")
            print(f"   • Frames processed: {frame_count}")
            print(f"   • Duration: {actual_duration:.1f}s")
            print(f"   • Average FPS: {avg_fps:.1f}")
            print(f"   • Processing time: {avg_processing_time*1000:.1f}ms")
            print(f"   • Total detections: {detection_count}")
            print(f"   • Total tracks: {track_count}")
            
            self.test_results['webcam_integration'] = webcam_results
            return webcam_results
            
        except Exception as e:
            print(f"❌ Webcam test failed: {e}")
            return {'error': str(e)}
    
    def test_memory_usage(self):
        """Test memory usage under continuous operation"""
        print("\n💾 Memory Usage Test")
        print("-" * 40)
        
        try:
            import psutil
            process = psutil.Process()
            
            # Initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Initial memory usage: {initial_memory:.1f} MB")
            
            memory_samples = [initial_memory]
            
            # Process many frames
            for i in range(100):
                test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                result = self.fusion_engine.process_frame(test_frame)
                
                if (i + 1) % 20 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    print(f"   📈 Frame {i + 1}: {current_memory:.1f} MB")
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            max_memory = max(memory_samples)
            
            memory_results = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'max_memory_mb': max_memory,
                'memory_increase_mb': memory_increase,
                'memory_samples': memory_samples
            }
            
            print(f"\n📊 Memory Usage Results:")
            print(f"   • Initial: {initial_memory:.1f} MB")
            print(f"   • Final: {final_memory:.1f} MB")
            print(f"   • Increase: {memory_increase:.1f} MB")
            print(f"   • Peak: {max_memory:.1f} MB")
            
            if memory_increase > 100:  # MB
                print("   ⚠️ Significant memory increase detected")
            else:
                print("   ✅ Memory usage stable")
            
            self.test_results['memory_usage'] = memory_results
            return memory_results
            
        except ImportError:
            print("⚠️ psutil not available, skipping memory test")
            return {'skipped': True, 'reason': 'psutil not available'}
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            return {'error': str(e)}
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 Test Report Generation")
        print("=" * 60)
        
        report = {
            'test_timestamp': time.time(),
            'test_duration': time.time(),  # Will be updated
            'results': self.test_results,
            'summary': {}
        }
        
        # Calculate summary statistics
        summary = {}
        
        # Component tests
        if 'individual_components' in self.test_results:
            comp_results = self.test_results['individual_components']
            summary['components_tested'] = len(comp_results)
            summary['components_passed'] = sum(1 for r in comp_results.values() if 'error' not in r)
        
        # Pipeline performance
        if 'integrated_pipeline' in self.test_results:
            pipeline = self.test_results['integrated_pipeline']
            if 'avg_fps' in pipeline:
                summary['pipeline_fps'] = pipeline['avg_fps']
                summary['pipeline_performance'] = 'good' if pipeline['avg_fps'] > 10 else 'needs_optimization'
        
        # Benchmark results
        if 'performance_benchmarks' in self.test_results:
            benchmarks = self.test_results['performance_benchmarks']
            summary['benchmark_configs_tested'] = len(benchmarks)
        
        report['summary'] = summary
        
        # Print summary
        print("📊 Test Summary:")
        for key, value in summary.items():
            print(f"   • {key}: {value}")
        
        # Save to file
        try:
            report_path = Path("test_report_vision_pipeline.json")
            with open(report_path, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Test report saved to: {report_path}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        return report
    
    def run_all_tests(self, include_webcam: bool = True, webcam_duration: int = 10):
        """Run complete test suite"""
        print("🚀 Starting Complete Vision Pipeline Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Setup
        if not self.setup_components():
            print("❌ Component setup failed, aborting tests")
            return False
        
        # Run tests
        tests_to_run = [
            ("Individual Components", self.test_individual_components),
            ("Integrated Pipeline", self.test_integrated_pipeline),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Memory Usage", self.test_memory_usage),
        ]
        
        if include_webcam:
            tests_to_run.append(("Webcam Integration", lambda: self.test_webcam_integration(webcam_duration)))
        
        passed_tests = 0
        total_tests = len(tests_to_run)
        
        for test_name, test_func in tests_to_run:
            try:
                print(f"\n{'='*20} {test_name} {'='*20}")
                result = test_func()
                if result and 'error' not in result:
                    passed_tests += 1
                    print(f"✅ {test_name} completed successfully")
                else:
                    print(f"❌ {test_name} failed")
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
        
        # Generate report
        test_duration = time.time() - start_time
        self.test_results['test_duration'] = test_duration
        
        report = self.generate_test_report()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🎯 TEST SUITE COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Tests passed: {passed_tests}/{total_tests}")
        print(f"⏱️ Total duration: {test_duration:.1f}s")
        print(f"📊 Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Vision pipeline is ready.")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️ Most tests passed, minor issues detected.")
        else:
            print("❌ Significant issues detected, investigation needed.")
        
        return passed_tests == total_tests


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Pipeline Test Suite")
    parser.add_argument("--no-webcam", action="store_true", help="Skip webcam tests")
    parser.add_argument("--webcam-duration", type=int, default=10, help="Webcam test duration (seconds)")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = VisionPipelineTest()
    
    if args.quick:
        # Quick tests only
        print("🏃 Running quick tests...")
        success = test_suite.setup_components()
        if success:
            test_suite.test_individual_components()
            test_suite.generate_test_report()
    else:
        # Full test suite
        success = test_suite.run_all_tests(
            include_webcam=not args.no_webcam,
            webcam_duration=args.webcam_duration
        )
    
    return success


if __name__ == "__main__":
    main()