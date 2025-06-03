# YOLO Drone Detection Training Script
# File: training/vision/train_detector.py
# Run from project root: python training/vision/train_detector.py

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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
                 dataset_path=None,
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
        # Get project root
        self.project_root = Path.cwd()
        
        # Set default dataset path
        if dataset_path is None:
            dataset_path = "data/processed/vision/visdrone_yolo"
        self.dataset_path = self.project_root / dataset_path
        
        self.model_name = model_name
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Results directory
        self.results_dir = self.project_root / "results" / self.project_name / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 DroneDetectionTrainer initialized")
        logger.info(f"📁 Project root: {self.project_root}")
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
            logger.info("💡 Run data conversion first: python src/data/loaders/vision_loader.py --full")
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
                
                if image_count == 0:
                    logger.error(f"❌ No images found in {split} set")
                    return False
        
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
            param_count = sum(p.numel() for p in model.model.parameters())
            logger.info(f"📊 Model parameters: {param_count:,}")
            
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
            'project': str(self.project_root / "runs" / "detect"),
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
        try:
            model.save(str(best_model_path))
            logger.info(f"✅ Saved best model: {best_model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save model: {e}")
        
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
        try:
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
            # Calculate F1 score
            if metrics['precision'] > 0 and metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0.0
        except Exception as e:
            logger.warning(f"⚠️ Could not extract metrics: {e}")
            metrics = {'error': str(e)}
        
        logger.info("📊 Validation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   • {metric}: {value:.4f}")
            else:
                logger.info(f"   • {metric}: {value}")
        
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
        
        if not Path(model_path).exists():
            logger.warning(f"⚠️ Model not found: {model_path}")
            return
        
        model = YOLO(str(model_path))
        
        # Get test images
        if test_images is None:
            test_dir = self.dataset_path / "images" / "test"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg"))[:6]  # First 6 images
            else:
                val_dir = self.dataset_path / "images" / "val"
                if val_dir.exists():
                    test_images = list(val_dir.glob("*.jpg"))[:6]  # Use val images
                else:
                    logger.warning("⚠️ No test images found")
                    return
        
        if not test_images:
            logger.warning("⚠️ No test images found")
            return
        
        logger.info(f"🖼️ Testing on {len(test_images)} images")
        
        # Run inference
        try:
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
            inference_plot_path = self.results_dir / "inference_samples.png"
            plt.savefig(inference_plot_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"✅ Inference test completed")
            logger.info(f"📸 Sample results saved to: {inference_plot_path}")
            
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")


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
    
    # Quick training (just 3 epochs)
    try:
        model, results = trainer.train(
            epochs=3,
            imgsz=640,
            batch_size=8,  # Small batch for testing
            patience=50,   # No early stopping for quick test
            save_period=1  # Save every epoch for quick test
        )
        
        print("✅ Quick training test completed!")
        print("🔍 Running validation...")
        
        # Quick validation
        metrics = trainer.validate_model()
        
        if metrics and 'error' not in metrics:
            print("📊 Quick test results:")
            print(f"   • mAP50: {metrics.get('mAP50', 'N/A'):.3f}")
            print(f"   • mAP50-95: {metrics.get('mAP50-95', 'N/A'):.3f}")
        
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
    
    print("🎉 Training pipeline completed!")


if __name__ == "__main__":
    main()