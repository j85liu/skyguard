# SkyGuard Project Configuration
# File: config/settings.py

import os
from pathlib import Path

class ProjectConfig:
    """
    Centralized project configuration and path management
    
    This follows industry best practices:
    1. All paths relative to project root
    2. Centralized configuration
    3. Easy to modify for different environments
    4. Handles both development and deployment scenarios
    """
    
    def __init__(self):
        # Find project root (directory containing this config)
        self.PROJECT_ROOT = self._find_project_root()
        
        # Data directories (relative to project root)
        self.DATA_ROOT = self.PROJECT_ROOT / "data"
        self.RAW_DATA_ROOT = self.DATA_ROOT / "raw"
        self.PROCESSED_DATA_ROOT = self.DATA_ROOT / "processed"
        
        # Vision data paths
        self.VISION_RAW = self.RAW_DATA_ROOT / "vision"
        self.VISDRONE_RAW = self.VISION_RAW / "visdrone"
        self.VISDRONE_PROCESSED = self.PROCESSED_DATA_ROOT / "vision" / "visdrone_yolo"
        
        # RF data paths  
        self.RF_RAW = self.RAW_DATA_ROOT / "rf"
        self.RF_PROCESSED = self.PROCESSED_DATA_ROOT / "rf"
        
        # Acoustic data paths
        self.ACOUSTIC_RAW = self.RAW_DATA_ROOT / "acoustic"
        self.ACOUSTIC_PROCESSED = self.PROCESSED_DATA_ROOT / "acoustic"
        
        # Model directories
        self.MODELS_ROOT = self.PROJECT_ROOT / "models"
        self.PRETRAINED_MODELS = self.MODELS_ROOT / "pretrained"
        self.TRAINED_MODELS = self.MODELS_ROOT / "trained"
        self.OPTIMIZED_MODELS = self.MODELS_ROOT / "optimized"
        
        # Results and logs
        self.RESULTS_ROOT = self.PROJECT_ROOT / "results"
        self.LOGS_ROOT = self.PROJECT_ROOT / "logs"
        
        # Training runs (Ultralytics convention)
        self.RUNS_ROOT = self.PROJECT_ROOT / "runs"
        
        # Notebooks and scripts
        self.NOTEBOOKS_ROOT = self.PROJECT_ROOT / "notebooks"
        self.SCRIPTS_ROOT = self.PROJECT_ROOT / "scripts"
        
    def _find_project_root(self):
        """
        Find the project root directory
        
        Looks for common project indicators:
        - .git directory
        - requirements.txt
        - setup.py
        - config/ directory
        
        Returns:
            Path: Project root directory
        """
        current_dir = Path(__file__).parent.absolute()
        
        # Look for project root indicators
        indicators = ['.git', 'requirements.txt', 'setup.py', 'README.md']
        
        # Start from current directory and go up
        search_dir = current_dir
        while search_dir != search_dir.parent:  # Not at filesystem root
            for indicator in indicators:
                if (search_dir / indicator).exists():
                    return search_dir
            search_dir = search_dir.parent
        
        # If not found, assume current working directory is project root
        return Path.cwd()
    
    def create_directories(self):
        """
        Create all necessary project directories
        """
        directories = [
            self.DATA_ROOT,
            self.RAW_DATA_ROOT,
            self.PROCESSED_DATA_ROOT,
            self.VISION_RAW,
            self.RF_RAW,
            self.ACOUSTIC_RAW,
            self.MODELS_ROOT,
            self.PRETRAINED_MODELS,
            self.TRAINED_MODELS,
            self.OPTIMIZED_MODELS,
            self.RESULTS_ROOT,
            self.LOGS_ROOT,
            self.RUNS_ROOT
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created project directories")
    
    def validate_visdrone_data(self):
        """
        Validate VisDrone data placement
        
        Returns:
            bool: True if VisDrone data is properly placed
        """
        required_dirs = [
            self.VISDRONE_RAW / "VisDrone2019-DET-train" / "images",
            self.VISDRONE_RAW / "VisDrone2019-DET-train" / "annotations",
            self.VISDRONE_RAW / "VisDrone2019-DET-val" / "images", 
            self.VISDRONE_RAW / "VisDrone2019-DET-val" / "annotations",
            self.VISDRONE_RAW / "VisDrone2019-DET-test-dev" / "images",
            self.VISDRONE_RAW / "VisDrone2019-DET-test-dev" / "annotations"
        ]
        
        print(f"🔍 Checking VisDrone data in: {self.VISDRONE_RAW}")
        
        all_good = True
        for dir_path in required_dirs:
            if dir_path.exists():
                if 'images' in str(dir_path):
                    count = len(list(dir_path.glob("*.jpg")))
                    print(f"✅ {dir_path.relative_to(self.PROJECT_ROOT)} ({count} images)")
                else:
                    count = len(list(dir_path.glob("*.txt"))) 
                    print(f"✅ {dir_path.relative_to(self.PROJECT_ROOT)} ({count} annotations)")
            else:
                print(f"❌ {dir_path.relative_to(self.PROJECT_ROOT)}")
                all_good = False
        
        if not all_good:
            print(f"\n💡 EXPECTED DATA STRUCTURE:")
            print(f"   {self.VISDRONE_RAW.relative_to(self.PROJECT_ROOT)}/")
            print(f"   ├── VisDrone2019-DET-train/")
            print(f"   │   ├── images/")
            print(f"   │   └── annotations/")
            print(f"   ├── VisDrone2019-DET-val/")
            print(f"   │   ├── images/") 
            print(f"   │   └── annotations/")
            print(f"   └── VisDrone2019-DET-test-dev/")
            print(f"       ├── images/")
            print(f"       └── annotations/")
        
        return all_good
    
    def get_absolute_path(self, relative_path):
        """
        Convert relative path to absolute path from project root
        
        Args:
            relative_path (str or Path): Relative path from project root
            
        Returns:
            Path: Absolute path
        """
        return self.PROJECT_ROOT / relative_path
    
    def print_config(self):
        """
        Print current configuration
        """
        print(f"📁 PROJECT CONFIGURATION")
        print(f"=" * 50)
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Data Root: {self.DATA_ROOT}")
        print(f"VisDrone Raw: {self.VISDRONE_RAW}")
        print(f"VisDrone Processed: {self.VISDRONE_PROCESSED}")
        print(f"Models: {self.TRAINED_MODELS}")
        print(f"Results: {self.RESULTS_ROOT}")


# Global configuration instance
config = ProjectConfig()

# Convenience functions for easy imports
def get_project_root():
    """Get project root directory"""
    return config.PROJECT_ROOT

def get_visdrone_path():
    """Get VisDrone dataset path"""
    return config.VISDRONE_RAW

def get_processed_data_path():
    """Get processed data path"""
    return config.VISDRONE_PROCESSED

def setup_project():
    """Setup project directories and validate data"""
    config.create_directories()
    return config.validate_visdrone_data()


if __name__ == "__main__":
    # Test configuration
    config.print_config()
    print()
    setup_project()