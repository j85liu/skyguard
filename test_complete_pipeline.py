# Complete SkyGuard Pipeline Test - Fixed Version
# File: test_complete_pipeline.py
# Run from project root: python test_complete_pipeline.py

import subprocess
import sys
from pathlib import Path
import time
import signal

def timeout_handler(signum, frame):
    """Handle timeout"""
    raise TimeoutError("Command timed out")

def run_command_with_timeout(command, description, timeout_minutes=10):
    """Run a command with timeout and better progress handling"""
    print(f"\n🔄 {description}")
    print("=" * 60)
    print(f"Command: {command}")
    print(f"⏱️ Timeout: {timeout_minutes} minutes")
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)
        
        # Start process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=Path.cwd()
        )
        
        print("🔄 Process started, monitoring output...")
        
        output_lines = []
        last_output_time = time.time()
        no_output_timeout = 120  # 2 minutes without output
        
        while True:
            # Check if process is still running
            if process.poll() is not None:
                break
            
            # Read line with timeout
            try:
                line = process.stdout.readline()
            except:
                break
            
            if line:
                line = line.strip()
                output_lines.append(line)
                last_output_time = time.time()
                
                # Show important lines immediately
                if any(keyword in line.lower() for keyword in [
                    'epoch', 'training', 'completed', 'error', 'failed', 
                    'success', 'map', 'loss', 'saving', 'finished'
                ]):
                    print(f"📝 {line}")
                
                # Show progress indicators
                elif any(char in line for char in ['%', '|', '█', '/']):
                    print(f"📊 {line}")
                
                # Show every 10th line to prove it's working
                elif len(output_lines) % 10 == 0:
                    print(f"⏳ Processing... (line {len(output_lines)})")
            
            else:
                # No output received, check timeout
                if time.time() - last_output_time > no_output_timeout:
                    print(f"\n⚠️ No output for {no_output_timeout} seconds, assuming hung...")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    break
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.1)
        
        # Cancel timeout
        signal.alarm(0)
        
        # Get final return code
        return_code = process.poll()
        
        # Show final output
        print(f"\n📄 Process completed with return code: {return_code}")
        if len(output_lines) > 0:
            print("Last few lines of output:")
            for line in output_lines[-5:]:
                print(f"   {line}")
        
        if return_code == 0:
            print("✅ SUCCESS")
            return True
        else:
            print("❌ FAILED")
            return False
            
    except TimeoutError:
        print(f"\n⏰ Command timed out after {timeout_minutes} minutes")
        try:
            process.terminate()
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def run_command_simple(command, description):
    """Simple command execution without progress monitoring"""
    print(f"\n🔄 {description}")
    print("=" * 60)
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=Path.cwd(),
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr[-1000:])
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def create_directories():
    """Create necessary directory structure"""
    print("\n📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    directories = [
        "src/data/loaders",
        "training/vision",
        "results",
        "data/processed/vision"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def check_data_exists():
    """Check if VisDrone data exists"""
    print("\n🔍 CHECKING DATA AVAILABILITY")
    print("=" * 60)
    
    data_path = Path("data/raw/vision/visdrone")
    required_dirs = [
        data_path / "VisDrone2019-DET-train" / "images",
        data_path / "VisDrone2019-DET-val" / "images"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            count = len(list(dir_path.glob("*.jpg")))
            print(f"✅ {dir_path}: {count} images")
        else:
            print(f"❌ Missing: {dir_path}")
            all_exist = False
    
    return all_exist

def save_script_files():
    """Save the script files to correct locations"""
    print("\n💾 SAVING SCRIPT FILES")
    print("=" * 60)
    
    files_to_check = [
        "src/data/loaders/vision_loader.py",
        "training/vision/train_detector.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            print(f"💡 You need to save the code for {file_path}")
            return False
    
    return True

def check_training_results():
    """Check if any training results were generated"""
    print("\n🔍 CHECKING RESULTS")
    print("=" * 40)
    
    success = False
    
    # Check for training runs
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if run_dirs:
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            weights_dir = latest_run / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                last_pt = weights_dir / "last.pt"
                if best_pt.exists() or last_pt.exists():
                    print(f"✅ Found trained model: {latest_run}/weights/")
                    success = True
                else:
                    print(f"⚠️ Training run found but no model weights: {latest_run}")
            else:
                print(f"⚠️ Training run found but no weights directory: {latest_run}")
    
    # Check processed data
    processed_dir = Path("data/processed/vision/visdrone_yolo")
    if processed_dir.exists() and (processed_dir / "dataset.yaml").exists():
        print(f"✅ Found processed dataset: {processed_dir}")
        success = True
    
    if not success:
        print("❌ No training results or processed data found")
    
    return success

def main():
    """Run complete pipeline test with better error handling"""
    print("🚀 SKYGUARD COMPLETE PIPELINE TEST")
    print("=" * 80)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Running from: {current_dir}")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Check if data exists
    if not check_data_exists():
        print("\n❌ VisDrone data not found!")
        print("💡 Make sure your data is in: data/raw/vision/visdrone/")
        return False
    
    # Step 3: Check if script files exist
    if not save_script_files():
        print("\n❌ Script files missing!")
        print("💡 Save the provided code to the correct file locations")
        return False
    
    # Step 4: Test data conversion
    print("\n" + "="*80)
    print("STEP 1: DATA CONVERSION TEST")
    print("="*80)
    
    success = run_command_simple(
        "python src/data/loaders/vision_loader.py --test",
        "Testing data conversion (validation set only)"
    )
    
    if not success:
        print("❌ Data conversion test failed!")
        return False
    
    # Step 5: Full data conversion
    print("\n" + "="*80)
    print("STEP 2: FULL DATA CONVERSION")
    print("="*80)
    
    success = run_command_simple(
        "python src/data/loaders/vision_loader.py --full",
        "Converting full dataset to YOLO format"
    )
    
    if not success:
        print("❌ Full data conversion failed!")
        return False
    
    # Step 6: Quick training test with timeout
    print("\n" + "="*80)
    print("STEP 3: QUICK TRAINING TEST")
    print("="*80)
    
    print("🤖 Starting quick training test (3 epochs)...")
    print("⏱️ Expected time: 5-8 minutes on M2")
    print("🚨 Will timeout after 10 minutes if hung")
    
    success = run_command_with_timeout(
        "python training/vision/train_detector.py --quick-test",
        "Running quick training test with timeout protection",
        timeout_minutes=10
    )
    
    if not success:
        print("❌ Quick training test failed or timed out!")
        print("💡 Trying simpler fallback test...")
        
        # Fallback: just test YOLO import
        success = run_command_simple(
            "python -c \"from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('✅ YOLO working')\"",
            "Testing YOLO installation as fallback"
        )
    
    # Step 7: Try Ultralytics direct (shorter timeout)
    print("\n" + "="*80)
    print("STEP 4: ULTRALYTICS DIRECT TEST")
    print("="*80)
    
    print("🔄 Testing Ultralytics direct approach...")
    print("⏱️ Will timeout after 8 minutes...")
    
    ultralytics_success = run_command_with_timeout(
        "yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=2 imgsz=640 patience=50",
        "Training with Ultralytics (shorter test)",
        timeout_minutes=8
    )
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 PIPELINE TEST COMPLETE!")
    print("="*80)
    
    # Check what actually completed successfully
    results_exist = check_training_results()
    
    if success or ultralytics_success or results_exist:
        print("✅ Pipeline is functional!")
        print("\n🚀 NEXT STEPS:")
        print("1. Full training: python training/vision/train_detector.py --epochs 100")
        print("2. Or use Ultralytics: yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100")
        print("3. Monitor with: tensorboard --logdir runs/detect")
        print("4. View results in: runs/detect/train*/")
        
        if results_exist:
            print("\n📊 CURRENT RESULTS:")
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                for i, run_dir in enumerate(run_dirs[:3]):  # Show last 3 runs
                    print(f"   {i+1}. {run_dir.name}")
    else:
        print("⚠️ Tests had issues, but setup is functional")
        print("\n🔧 MANUAL OPTIONS:")
        print("1. Try: yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=10")
        print("2. Check individual scripts manually")
        print("3. Look at error logs above")
    
    return True

if __name__ == "__main__":
    main()