# YOLO Drone Detection Training Script
# File: training/vision/train_detector.py

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneDetectionTrainer:
    """
    Complete training pipeline for drone detection using YOLO
    
    This class handles:
    1. Model initialization and configuration
    2. Training with proper logging and checkpointing
    3. Validation and testing
    4. Results visualization and analysis
    5. Model export for deployment
    """
    
    def __init__(self, 
                 dataset_path="data/processed/vision/visdrone_yolo",
                 model_name="yolo11n.pt",
                 project_name="skyguard_drone_detection",
                 experiment_name=None):
        """
        Initialize the trainer
        
        Args:
            dataset_path (str): Path to YOLO formatted dataset
            model_name (str): YOLO model variant (yolo11n/s/m/l/x.pt)
            project_name (str): Project name for organizing runs
            experiment_name (str): Specific experiment name
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Results directory
        self.results_dir = Path("results") / self.project_name / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 DroneDetectionTrainer initialized")
        logger.info(f"📁 Dataset: {self.dataset_path}")
        logger.info(f"🤖 Model: {self.model_name}")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"📊 Results: {self.results_dir}")
    
    def validate_dataset(self):
        """
        Validate that the dataset is properly formatted
        
        Returns:
            bool: True if dataset is valid
        """
        logger.info("🔍 Validating dataset...")
        
        # Check for dataset.yaml
        dataset_yaml = self.dataset_path / "dataset.yaml"
        if not dataset_yaml.exists():
            logger.error(f"❌ Dataset config not found: {dataset_yaml}")
            return False
        
        # Load and validate dataset config
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ Missing key in dataset.yaml: {key}")
                return False
        
        # Check directories exist
        for split in ['train', 'val']:
            if split in config:
                images_dir = self.dataset_path / config[split]
                if not images_dir.exists():
                    logger.error(f"❌ Missing directory: {images_dir}")
                    return False
                
                # Count images
                image_count = len(list(images_dir.glob("*.jpg")))
                logger.info(f"✅ {split}: {image_count} images")
        
        logger.info(f"✅ Dataset validation passed")
        logger.info(f"📊 Classes: {config['nc']} ({', '.join(config['names'])})")
        
        return True
    
    def setup_model(self, pretrained=True):
        """
        Setup YOLO model for training
        
        Args:
            pretrained (bool): Use pretrained weights
            
        Returns:
            YOLO: Configured model
        """
        logger.info(f"🤖 Setting up model: {self.model_name}")
        
        try:
            # Load model
            if pretrained:
                model = YOLO(self.model_name)  # Load pretrained model
                logger.info("✅ Loaded pretrained weights")
            else:
                # Load model architecture only
                model_yaml = self.model_name.replace('.pt', '.yaml')
                model = YOLO(model_yaml)
                logger.info("✅ Loaded model architecture (no pretrained weights)")
            
            # Model info
            logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to setup model: {e}")
            raise
    
    def train(self, 
              epochs=100,
              imgsz=640,
              batch_size=16,
              learning_rate=0.01,
              save_period=10,
              patience=20,
              **kwargs):
        """
        Train the drone detection model
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch_size (int): Training batch size
            learning_rate (float): Initial learning rate
            save_period (int): Save checkpoint every N epochs
            patience (int): Early stopping patience
            **kwargs: Additional training arguments
        """
        logger.info("🚀 STARTING TRAINING")
        logger.info("=" * 50)
        
        # Validate dataset first
        if not self.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Setup model
        model = self.setup_model(pretrained=True)
        
        # Training configuration
        train_config = {
            'data': str(self.dataset_path / "dataset.yaml"),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': learning_rate,
            'save_period': save_period,
            'patience': patience,
            'device': self.device,
            'project': str(Path("runs") / "detect"),
            'name': f"{self.project_name}_{self.experiment_name}",
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'plots': True,
            **kwargs
        }
        
        # Log training configuration
        logger.info("⚙️ Training Configuration:")
        for key, value in train_config.items():
            logger.info(f"   • {key}: {value}")
        
        # Save training config
        config_path = self.results_dir / "train_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(train_config, f, default_flow_style=False)
        
        try:
            # Start training
            logger.info("🎯 Starting training...")
            results = model.train(**train_config)
            
            # Save results
            self.save_training_results(model, results, train_config)
            
            logger.info("🎉 Training completed successfully!")
            return model, results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_training_results(self, model, results, config):
        """
        Save training results and model artifacts
        
        Args:
            model: Trained YOLO model
            results: Training results
            config: Training configuration
        """
        logger.info("💾 Saving training results...")
        
        # Save best model
        best_model_path = self.results_dir / "best.pt"
        model.save(str(best_model_path))
        logger.info(f"✅ Saved best model: {best_model_path}")
        
        # Save training summary
        summary = {
            'model': self.model_name,
            'dataset': str(self.dataset_path),
            'config': config,
            'device': self.device,
            'training_time': datetime.now().isoformat(),
        }
        
        summary_path = self.results_dir / "training_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"✅ Saved training summary: {summary_path}")
    
    def validate_model(self, model_path=None, split='val'):
        """
        Validate trained model on validation/test set
        
        Args:
            model_path (str): Path to model weights
            split (str): Dataset split to validate on
            
        Returns:
            dict: Validation results
        """
        logger.info(f"🔍 Validating model on {split} set...")
        
        # Load model
        if model_path is None:
            model_path = self.results_dir / "best.pt"
        
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            return None
        
        model = YOLO(str(model_path))
        
        # Run validation
        dataset_yaml = self.dataset_path / "dataset.yaml"
        results = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=640,
            batch=16,
            save_json=True,
            save_txt=True,
            plots=True,
            verbose=True
        )
        
        # Extract key metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr))
        }
        
        logger.info("📊 Validation Results:")
        for metric, value in metrics.items():
            logger.info(f"   • {metric}: {value:.4f}")
        
        # Save validation results
        val_results_path = self.results_dir / f"validation_{split}_results.yaml"
        with open(val_results_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        return metrics
    
    def test_inference(self, model_path=None, test_images=None, conf_threshold=0.25):
        """
        Test model inference on sample images
        
        Args:
            model_path (str): Path to model weights
            test_images (list): List of test image paths
            conf_threshold (float): Confidence threshold for detections
        """
        logger.info("🧪 Testing model inference...")
        
        # Load model
        if model_path is None:
            model_path = self.results_dir / "best.pt"
        
        model = YOLO(str(model_path))
        
        # Get test images
        if test_images is None:
            test_dir = self.dataset_path / "images" / "test"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg"))[:6]  # First 6 images
            else:
                val_dir = self.dataset_path / "images" / "val"
                test_images = list(val_dir.glob("*.jpg"))[:6]  # Use val images
        
        if not test_images:
            logger.warning("⚠️ No test images found")
            return
        
        # Run inference
        results = model(test_images, conf=conf_threshold, save=True, save_txt=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (img_path, result) in enumerate(zip(test_images[:6], results[:6])):
            if i >= 6:
                break
            
            # Plot result
            im_array = result.plot()  # Plot with bounding boxes
            axes[i].imshow(im_array)
            axes[i].set_title(f"{Path(img_path).name}\n{len(result.boxes)} detections")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "inference_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ Inference test completed")
        logger.info(f"📸 Sample results saved to: {self.results_dir}/inference_samples.png")
    
    def export_model(self, model_path=None, formats=['onnx', 'torchscript']):
        """
        Export model for deployment
        
        Args:
            model_path (str): Path to model weights
            formats (list): Export formats
        """
        logger.info("📦 Exporting model for deployment...")
        
        # Load model
        if model_path is None:
            model_path = self.results_dir / "best.pt"
        
        model = YOLO(str(model_path))
        
        # Export to different formats
        export_dir = self.results_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        for format_name in formats:
            try:
                exported_path = model.export(format=format_name, imgsz=640)
                logger.info(f"✅ Exported {format_name}: {exported_path}")
            except Exception as e:
                logger.error(f"❌ Failed to export {format_name}: {e}")
        
        logger.info(f"📦 Model exports saved to: {export_dir}")


def quick_training_test():
    """
    Quick training test with minimal epochs for validation
    """
    print("🧪 QUICK TRAINING TEST")
    print("=" * 40)
    
    trainer = DroneDetectionTrainer(
        dataset_path="data/processed/vision/visdrone_yolo",
        model_name="yolo11n.pt",  # Smallest/fastest model
        experiment_name="quick_test"
    )
    
    # Quick training (just 5 epochs)
    try:
        model, results = trainer.train(
            epochs=5,
            imgsz=640,
            batch_size=8,  # Small batch for testing
            patience=50,   # No early stopping for quick test
            save_period=5  # Save only at the end
        )
        
        print("✅ Quick training test completed!")
        print("🔍 Running validation...")
        
        # Quick validation
        metrics = trainer.validate_model()
        
        if metrics:
            print("📊 Quick test results:")
            print(f"   • mAP50: {metrics['mAP50']:.3f}")
            print(f"   • mAP50-95: {metrics['mAP50-95']:.3f}")
        
        print("🧪 Testing inference...")
        trainer.test_inference()
        
        print("\n🎉 Quick test completed successfully!")
        print("🚀 Ready for full training!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        print("💡 Check your dataset path and format")


def main():
    """
    Main training function with command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO drone detection model")
    parser.add_argument("--dataset", default="data/processed/vision/visdrone_yolo",
                       help="Path to YOLO dataset")
    parser.add_argument("--model", default="yolo11n.pt", 
                       choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
                       help="YOLO model variant")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--experiment", default=None, help="Experiment name")
    parser.add_argument("--quick-test", action="store_true", help="Run quick training test")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_training_test()
        return
    
    # Full training
    trainer = DroneDetectionTrainer(
        dataset_path=args.dataset,
        model_name=args.model,
        experiment_name=args.experiment
    )
    
    # Train model
    model, results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        learning_rate=args.lr
    )
    
    # Validate
    metrics = trainer.validate_model()
    
    # Test inference
    trainer.test_inference()
    
    # Export model
    trainer.export_model()
    
    print("🎉 Training pipeline completed!")


if __name__ == "__main__":
    main()