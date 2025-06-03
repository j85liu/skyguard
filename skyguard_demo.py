# SkyGuard Demo Script
# File: demo/skyguard_demo.py
# Run from project root: python demo/skyguard_demo.py

import sys
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.fusion_engine import SkyGuardFusionEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from project root")
    sys.exit(1)

class SkyGuardDemo:
    """
    Interactive demo of the SkyGuard drone detection system
    """
    
    def __init__(self):
        """Initialize demo"""
        self.engine = None
        self.demo_modes = {
            'webcam': self.webcam_demo,
            'video': self.video_demo,
            'synthetic': self.synthetic_demo,
            'benchmark': self.benchmark_demo
        }
        
        print("🛡️ SkyGuard Drone Detection System Demo")
        print("=" * 50)
    
    def initialize_engine(self):
        """Initialize the SkyGuard engine"""
        print("🚀 Initializing SkyGuard engine...")
        
        try:
            self.engine = SkyGuardFusionEngine(
                enable_vision=True,
                enable_rf=False,
                enable_acoustic=False
            )
            
            # Add demo callbacks
            self.engine.add_detection_callback(self.detection_callback)
            self.engine.add_threat_callback(self.threat_callback)
            
            print("✅ SkyGuard engine initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            return False
    
    def detection_callback(self, result):
        """Callback for detection events"""
        if len(result.fused_detections) > 0:
            print(f"🎯 Detection: {len(result.fused_detections)} objects detected (Frame {result.frame_id})")
    
    def threat_callback(self, threats):
        """Callback for threat events"""
        for threat in threats:
            print(f"🚨 THREAT: Track {threat.track_id} - Level: {threat.threat_level} ({threat.confidence:.2f})")
            print(f"   Reasons: {', '.join(threat.reasons)}")
    
    def webcam_demo(self, duration: int = 60):
        """Live webcam demonstration"""
        print(f"\n📹 Webcam Demo ({duration}s)")
        print("-" * 30)
        print("Controls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save current frame")
        print("  • Press 'r' to reset tracking")
        print()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return False
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam opened successfully")
        print("🎬 Starting live detection...")
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                result = self.engine.process_frame(frame)
                
                # Update counters
                frame_count += 1
                detection_count += len(result.fused_detections)
                
                # Create display frame
                display_frame = self.engine.draw_results(frame, result)
                
                # Add demo info
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                
                info_text = [
                    f"SkyGuard Demo - {remaining:.0f}s remaining",
                    f"Frame: {frame_count} | Detections: {detection_count}",
                    f"Press 'q' to quit, 's' to save, 'r' to reset"
                ]
                
                y_offset = 70
                for i, text in enumerate(info_text):
                    cv2.putText(display_frame, text, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('SkyGuard Live Demo', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Demo stopped by user")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"skyguard_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset tracking
                    if self.engine.vision_tracker:
                        self.engine.vision_tracker.reset()
                    print("🔄 Tracking reset")
        
        except KeyboardInterrupt:
            print("⚠️ Demo interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Show statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 Demo Statistics:")
        print(f"   • Duration: {elapsed:.1f}s")
        print(f"   • Frames processed: {frame_count}")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   • Total detections: {detection_count}")
        
        return True
    
    def video_demo(self, video_path: str):
        """Demonstrate on pre-recorded video"""
        print(f"\n🎥 Video Demo: {video_path}")
        print("-" * 30)
        
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return False
        
        # Process video
        results = self.engine.process_video_stream(
            source=video_path,
            show_display=True,
            save_path="skyguard_demo_output.mp4"
        )
        
        print(f"✅ Processed {len(results)} frames")
        print("💾 Output saved to skyguard_demo_output.mp4")
        
        return True
    
    def synthetic_demo(self, duration: int = 30):
        """Demonstrate with synthetic drone-like objects"""
        print(f"\n🎭 Synthetic Demo ({duration}s)")
        print("-" * 30)
        print("Simulating multiple moving objects...")
        
        # Create synthetic video
        width, height = 1280, 720
        fps = 30
        total_frames = duration * fps
        
        # Object parameters
        num_objects = 3
        objects = []
        
        for i in range(num_objects):
            obj = {
                'start_x': np.random.randint(50, width - 50),
                'start_y': np.random.randint(50, height - 50),
                'speed_x': np.random.uniform(-5, 5),
                'speed_y': np.random.uniform(-5, 5),
                'size': np.random.randint(30, 60),
                'color': tuple(np.random.randint(100, 255, 3).tolist())
            }
            objects.append(obj)
        
        print(f"🎯 Created {num_objects} synthetic objects")
        print("🎬 Starting synthetic video processing...")
        
        frame_count = 0
        detection_count = 0
        
        try:
            for frame_idx in range(total_frames):
                # Create frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add background texture
                noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                # Draw objects
                for obj in objects:
                    # Update position
                    x = int(obj['start_x'] + obj['speed_x'] * frame_idx)
                    y = int(obj['start_y'] + obj['speed_y'] * frame_idx)
                    
                    # Bounce off edges
                    if x < 0 or x > width - obj['size']:
                        obj['speed_x'] *= -1
                    if y < 0 or y > height - obj['size']:
                        obj['speed_y'] *= -1
                    
                    x = max(0, min(x, width - obj['size']))
                    y = max(0, min(y, height - obj['size']))
                    
                    # Draw object (simulate drone)
                    cv2.ellipse(frame, (x + obj['size']//2, y + obj['size']//2), 
                               (obj['size']//2, obj['size']//4), 0, 0, 360, obj['color'], -1)
                    
                    # Add rotor effect
                    for rotor in [(x + 10, y + 5), (x + obj['size'] - 10, y + 5),
                                 (x + 10, y + obj['size'] - 5), (x + obj['size'] - 10, y + obj['size'] - 5)]:
                        cv2.circle(frame, rotor, 3, (255, 255, 255), 1)
                
                # Process frame
                result = self.engine.process_frame(frame)
                
                # Update counters
                frame_count += 1
                detection_count += len(result.fused_detections)
                
                # Create display
                display_frame = self.engine.draw_results(frame, result)
                
                # Add progress info
                progress = (frame_idx + 1) / total_frames
                remaining_time = (total_frames - frame_idx - 1) / fps
                
                progress_text = f"Synthetic Demo - {remaining_time:.1f}s remaining ({progress*100:.0f}%)"
                cv2.putText(display_frame, progress_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('SkyGuard Synthetic Demo', display_frame)
                
                # Control playback speed
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    print("🛑 Demo stopped by user")
                    break
        
        except KeyboardInterrupt:
            print("⚠️ Demo interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
        
        print(f"\n📊 Synthetic Demo Results:")
        print(f"   • Frames processed: {frame_count}")
        print(f"   • Total detections: {detection_count}")
        print(f"   • Detection rate: {detection_count/frame_count:.2f} per frame")
        
        return True
    
    def benchmark_demo(self):
        """Performance benchmark demonstration"""
        print("\n⚡ Performance Benchmark")
        print("-" * 30)
        
        # Test different configurations
        test_configs = [
            {"size": (320, 240), "name": "Low Resolution"},
            {"size": (640, 480), "name": "Medium Resolution"},
            {"size": (1280, 720), "name": "High Resolution"},
        ]
        
        for config in test_configs:
            width, height = config["size"]
            name = config["name"]
            
            print(f"\n🔧 Testing {name} ({width}x{height})...")
            
            # Create test image
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some objects to detect
            for _ in range(3):
                x = np.random.randint(50, width - 100)
                y = np.random.randint(50, height - 100)
                cv2.rectangle(test_image, (x, y), (x + 80, y + 50), (255, 255, 255), -1)
            
            # Benchmark
            times = []
            num_runs = 50
            
            print(f"   Running {num_runs} iterations...")
            
            for i in range(num_runs):
                start = time.time()
                result = self.engine.process_frame(test_image)
                times.append(time.time() - start)
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{num_runs}")
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1 / avg_time if avg_time > 0 else 0
            
            print(f"   Results:")
            print(f"     • Average: {avg_time*1000:.1f}ms ({fps:.1f} FPS)")
            print(f"     • Min/Max: {min_time*1000:.1f}ms / {max_time*1000:.1f}ms")
            print(f"     • Standard deviation: {np.std(times)*1000:.1f}ms")
        
        # System performance summary
        perf_summary = self.engine.get_performance_summary()
        
        print(f"\n📊 System Performance Summary:")
        for key, value in perf_summary.items():
            if isinstance(value, dict):
                print(f"   • {key}: {len(value)} metrics")
            elif isinstance(value, float):
                print(f"   • {key}: {value:.3f}")
            else:
                print(f"   • {key}: {value}")
        
        return True
    
    def run_demo(self, mode: str, **kwargs):
        """Run specific demo mode"""
        if not self.initialize_engine():
            return False
        
        if mode not in self.demo_modes:
            print(f"❌ Unknown demo mode: {mode}")
            print(f"Available modes: {list(self.demo_modes.keys())}")
            return False
        
        print(f"🎬 Starting {mode} demo...")
        
        try:
            success = self.demo_modes[mode](**kwargs)
            
            if success:
                print(f"✅ {mode} demo completed successfully")
            else:
                print(f"❌ {mode} demo failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="SkyGuard Drone Detection Demo")
    parser.add_argument("mode", choices=['webcam', 'video', 'synthetic', 'benchmark'], 
                       help="Demo mode to run")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Demo duration in seconds (for webcam/synthetic)")
    parser.add_argument("--video-path", type=str, 
                       help="Path to video file (for video mode)")
    
    args = parser.parse_args()
    
    # Create demo
    demo = SkyGuardDemo()
    
    # Prepare arguments
    demo_kwargs = {}
    
    if args.mode in ['webcam', 'synthetic']:
        demo_kwargs['duration'] = args.duration
    elif args.mode == 'video':
        if not args.video_path:
            print("❌ Video path required for video mode")
            return False
        demo_kwargs['video_path'] = args.video_path
    
    # Run demo
    success = demo.run_demo(args.mode, **demo_kwargs)
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("💡 Try different modes to explore all features:")
        print("   • webcam: Live camera detection")
        print("   • synthetic: Simulated drone objects")
        print("   • benchmark: Performance testing")
        print("   • video: Process video files")
    else:
        print("\n❌ Demo failed. Check error messages above.")
    
    return success


if __name__ == "__main__":
    main()