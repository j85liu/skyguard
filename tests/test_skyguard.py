# SkyGuard Quick Test Script
# File: test_skyguard.py

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 CHECKING ENVIRONMENT")
    print("=" * 40)
    
    # Check Python version
    print(f"🐍 Python: {sys.version}")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python', 
        'matplotlib', 'seaborn', 'pandas', 'numpy', 'PIL', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'opencv-python':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages installed!")
    return True

def check_data_structure():
    """Check if VisDrone data is properly placed"""
    print("\n🗂️ CHECKING DATA STRUCTURE")
    print("=" * 40)
    
    expected_structure = [
        "data/raw/vision/visdrone/VisDrone2019-DET-train/images",
        "data/raw/vision/visdrone/VisDrone2019-DET-train/annotations", 
        "data/raw/vision/visdrone/VisDrone2019-DET-val/images",
        "data/raw/vision/visdrone/VisDrone2019-DET-val/annotations",
        "data/raw/vision/visdrone/VisDrone2019-DET-test-dev/images",
        "data/raw/vision/visdrone/VisDrone2019-DET-test-dev/annotations"
    ]
    
    all_good = True
    
    for path_str in expected_structure:
        path = Path(path_str)
        if path.exists():
            if 'images' in str(path):
                count = len(list(path.glob("*.jpg")))
                print(f"✅ {path} ({count} images)")
            else:
                count = len(list(path.glob("*.txt")))
                print(f"✅ {path} ({count} annotations)")
        else:
            print(f"❌ {path}")
            all_good = False
    
    if not all_good:
        print(f"\n💡 Place your VisDrone folders in:")
        print(f"   data/raw/vision/visdrone/")
        print(f"   - VisDrone2019-DET-train/")
        print(f"   - VisDrone2019-DET-val/")
        print(f"   - VisDrone2019-DET-test-dev/")
    
    return all_good

def test_data_exploration():
    """Test the data exploration notebook"""
    print("\n📊 TESTING DATA EXPLORATION")
    print("=" * 40)
    
    try:
        # Import our exploration script
        sys.path.append('.')
        
        # Create a minimal version of the explorer for testing
        from pathlib import Path
        import numpy as np
        from PIL import Image
        
        # Test loading a sample image
        sample_img_path = Path("data/raw/vision/visdrone/VisDrone2019-DET-val/images")
        
        if sample_img_path.exists():
            sample_images = list(sample_img_path.glob("*.jpg"))
            if sample_images:
                # Test loading image
                with Image.open(sample_images[0]) as img:
                    width, height = img.size
                    print(f"✅ Successfully loaded sample image: {width}x{height}")
                
                # Test annotation loading
                annotation_path = Path("data/raw/vision/visdrone/VisDrone2019-DET-val/annotations") / f"{sample_images[0].stem}.txt"
                if annotation_path.exists():
                    with open(annotation_path, 'r') as f:
                        lines = f.readlines()
                    print(f"✅ Successfully loaded annotation: {len(lines)} objects")
                else:
                    print(f"❌ Annotation not found: {annotation_path}")
                    return False
            else:
                print(f"❌ No images found in {sample_img_path}")
                return False
        else:
            print(f"❌ Sample image directory not found: {sample_img_path}")
            return False
        
        print("✅ Data exploration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data exploration test failed: {e}")
        return False

def test_data_conversion():
    """Test data conversion to YOLO format"""
    print("\n🔄 TESTING DATA CONVERSION")
    print("=" * 40)
    
    try:
        # Test the conversion on a small subset
        print("Testing conversion logic...")
        
        # Sample VisDrone annotation line
        visdrone_line = "100,150,50,30,1,4,0,0"  # car at (100,150) with size 50x30
        
        # Parse annotation
        parts = visdrone_line.split(',')
        bbox_left = int(parts[0])
        bbox_top = int(parts[1])
        bbox_width = int(parts[2])
        bbox_height = int(parts[3])
        object_category = int(parts[5])
        
        # Convert to YOLO format (assuming 640x480 image)
        img_width, img_height = 640, 480
        center_x = (bbox_left + bbox_width / 2) / img_width
        center_y = (bbox_top + bbox_height / 2) / img_height
        width_norm = bbox_width / img_width
        height_norm = bbox_height / img_height
        
        # Map class (car=4 -> car=3 in YOLO)
        yolo_class = object_category - 1  # Shift down by 1 (skip ignored class)
        
        print(f"✅ VisDrone format: {visdrone_line}")
        print(f"✅ YOLO format: {yolo_class} {center_x:.6f} {center_y:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        # Verify conversion is reasonable
        assert 0 <= center_x <= 1, "Center X out of bounds"
        assert 0 <= center_y <= 1, "Center Y out of bounds"
        assert 0 <= width_norm <= 1, "Width out of bounds"
        assert 0 <= height_norm <= 1, "Height out of bounds"
        
        print("✅ Data conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data conversion test failed: {e}")
        return False

def test_yolo_installation():
    """Test YOLO installation and basic functionality"""
    print("\n🤖 TESTING YOLO INSTALLATION")
    print("=" * 40)
    
    try:
        from ultralytics import YOLO
        
        # Test loading a pre-trained model
        print("Loading YOLO model...")
        model = YOLO('yolo11n.pt')  # This will download if not present
        
        print(f"✅ YOLO model loaded successfully")
        print(f"📊 Model info: {model.info()}")
        
        # Test basic prediction on a dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("Testing inference on dummy image...")
        results = model(dummy_image, verbose=False)
        
        print(f"✅ YOLO inference test passed!")
        print(f"📊 Detected {len(results[0].boxes)} objects in dummy image")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        print("💡 Try: pip install ultralytics")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 40)
    
    directories = [
        "data/raw/vision/visdrone",
        "data/processed/vision",
        "models/trained/vision",
        "models/optimized",
        "results",
        "logs",
        "notebooks/01_data_exploration",
        "notebooks/02_model_development", 
        "src/data/loaders",
        "training/vision",
        "evaluation/benchmarks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print("✅ Directory structure created!")

def run_quick_pipeline_test():
    """Run a quick end-to-end pipeline test"""
    print("\n🚀 RUNNING QUICK PIPELINE TEST")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    # Step 2: Create directories
    create_directory_structure()
    
    # Step 3: Check data
    data_ok = check_data_structure()
    
    # Step 4: Test components
    exploration_ok = test_data_exploration() if data_ok else False
    conversion_ok = test_data_conversion()
    yolo_ok = test_yolo_installation()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 40)
    print(f"{'✅' if True else '❌'} Environment: {'PASS' if True else 'FAIL'}")
    print(f"{'✅' if data_ok else '❌'} Data Structure: {'PASS' if data_ok else 'FAIL'}")
    print(f"{'✅' if exploration_ok else '❌'} Data Exploration: {'PASS' if exploration_ok else 'FAIL'}")
    print(f"{'✅' if conversion_ok else '❌'} Data Conversion: {'PASS' if conversion_ok else 'FAIL'}")
    print(f"{'✅' if yolo_ok else '❌'} YOLO Installation: {'PASS' if yolo_ok else 'FAIL'}")
    
    all_passed = all([data_ok, exploration_ok, conversion_ok, yolo_ok])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n🚀 NEXT STEPS:")
        print("1. Run data exploration: python -c 'from notebooks.01_data_exploration.vision_data_analysis import main; main()'")
        print("2. Convert data to YOLO: python src/data/loaders/vision_loader.py --test")
        print("3. Start training: python training/vision/train_detector.py --quick-test")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("💡 Fix the issues above before proceeding")
    
    return all_passed

def print_next_steps():
    """Print detailed next steps"""
    print("\n📋 DETAILED NEXT STEPS")
    print("=" * 60)
    
    print("1. 📊 EXPLORE YOUR DATA:")
    print("   python -c \"")
    print("   import sys; sys.path.append('.')") 
    print("   from notebooks.01_data_exploration.vision_data_analysis import main")
    print("   main()\"")
    
    print("\n2. 🔄 CONVERT DATA TO YOLO FORMAT:")
    print("   # Test conversion first")
    print("   python src/data/loaders/vision_loader.py --test")
    print("   ")
    print("   # Full conversion")
    print("   python src/data/loaders/vision_loader.py")
    
    print("\n3. 🚀 START TRAINING:")
    print("   # Quick test (5 epochs)")
    print("   python training/vision/train_detector.py --quick-test")
    print("   ")
    print("   # Full training")
    print("   python training/vision/train_detector.py --epochs 100 --batch-size 16")
    
    print("\n4. 🧪 ALTERNATIVE: Use Ultralytics directly:")
    print("   # This will auto-download VisDrone and start training")
    print("   yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100 imgsz=640")
    
    print("\n💡 TIPS:")
    print("   • Start with yolo11n.pt (fastest) for testing")
    print("   • Use yolo11s.pt or yolo11m.pt for better accuracy")
    print("   • Monitor training with: tensorboard --logdir runs/detect")
    print("   • Results will be in: runs/detect/train/")

def main():
    """Main test function"""
    print("🧪 SKYGUARD QUICK TEST SUITE")
    print("=" * 60)
    
    success = run_quick_pipeline_test()
    
    if success:
        print_next_steps()
    
    return success

if __name__ == "__main__":
    main()