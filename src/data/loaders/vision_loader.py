# VisDrone to YOLO Format Converter
# File: src/data/loaders/vision_loader.py

import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisDroneToYOLO:
    """
    Convert VisDrone dataset format to YOLO format for training
    
    This class handles the complete conversion pipeline:
    1. Parse VisDrone annotations
    2. Convert to YOLO format (normalized coordinates)
    3. Create proper directory structure
    4. Generate dataset configuration files
    """
    
    def __init__(self, visdrone_root="data/raw/vision/visdrone", 
                 yolo_output="data/processed/vision/visdrone_yolo"):
        """
        Initialize the converter
        
        Args:
            visdrone_root (str): Path to VisDrone dataset
            yolo_output (str): Path for YOLO formatted output
        """
        self.visdrone_root = Path(visdrone_root)
        self.yolo_output = Path(yolo_output)
        
        # VisDrone class mapping (excluding ignored regions)
        # Original: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 
        #          6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
        # YOLO:    0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
        #          5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
        
        self.class_mapping = {
            1: 0,   # pedestrian
            2: 1,   # people  
            3: 2,   # bicycle
            4: 3,   # car
            5: 4,   # van
            6: 5,   # truck
            7: 6,   # tricycle
            8: 7,   # awning-tricycle
            9: 8,   # bus
            10: 9   # motor
            # Note: class 0 (ignored) is skipped
        }
        
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        
        logger.info(f"📁 VisDrone root: {self.visdrone_root}")
        logger.info(f"📁 YOLO output: {self.yolo_output}")
        logger.info(f"🏷️ Classes: {len(self.class_names)} ({', '.join(self.class_names)})")
    
    def create_directory_structure(self):
        """
        Create YOLO dataset directory structure
        """
        logger.info("📂 Creating YOLO directory structure...")
        
        # Create main directories
        dirs_to_create = [
            self.yolo_output / "images" / "train",
            self.yolo_output / "images" / "val", 
            self.yolo_output / "images" / "test",
            self.yolo_output / "labels" / "train",
            self.yolo_output / "labels" / "val",
            self.yolo_output / "labels" / "test"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {dir_path}")
    
    def convert_bbox_format(self, bbox, img_width, img_height):
        """
        Convert VisDrone bbox format to YOLO format
        
        VisDrone: (left, top, width, height) in pixels
        YOLO: (center_x, center_y, width, height) normalized to [0,1]
        
        Args:
            bbox (dict): VisDrone bbox with left, top, width, height
            img_width (int): Image width in pixels
            img_height (int): Image height in pixels
            
        Returns:
            tuple: (center_x, center_y, width, height) normalized
        """
        # Extract VisDrone coordinates
        left = bbox['bbox_left']
        top = bbox['bbox_top'] 
        width = bbox['bbox_width']
        height = bbox['bbox_height']
        
        # Calculate center coordinates
        center_x = left + width / 2
        center_y = top + height / 2
        
        # Normalize by image dimensions
        center_x_norm = center_x / img_width
        center_y_norm = center_y / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Ensure coordinates are within [0, 1]
        center_x_norm = max(0, min(1, center_x_norm))
        center_y_norm = max(0, min(1, center_y_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        return center_x_norm, center_y_norm, width_norm, height_norm
    
    def parse_visdrone_annotation(self, annotation_path):
        """
        Parse VisDrone annotation file
        
        Args:
            annotation_path (Path): Path to annotation file
            
        Returns:
            list: List of annotation dictionaries
        """
        annotations = []
        
        if not annotation_path.exists():
            return annotations
        
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        annotation = {
                            'bbox_left': int(parts[0]),
                            'bbox_top': int(parts[1]),
                            'bbox_width': int(parts[2]),
                            'bbox_height': int(parts[3]),
                            'score': float(parts[4]) if parts[4] != '' else 1.0,
                            'object_category': int(parts[5]),
                            'truncation': int(parts[6]) if len(parts) > 6 and parts[6] != '' else 0,
                            'occlusion': int(parts[7]) if len(parts) > 7 and parts[7] != '' else 0
                        }
                        annotations.append(annotation)
                    except ValueError as e:
                        logger.warning(f"⚠️ Skipping invalid annotation line: {line} - Error: {e}")
        
        return annotations
    
    def convert_split(self, split_name):
        """
        Convert a single dataset split to YOLO format
        
        Args:
            split_name (str): Name of split (train/val/test-dev)
            
        Returns:
            tuple: (converted_count, skipped_count, error_count)
        """
        logger.info(f"🔄 Converting {split_name} split...")
        
        # Map split names
        split_mapping = {
            'train': 'train',
            'val': 'val', 
            'test-dev': 'test'
        }
        yolo_split = split_mapping.get(split_name, split_name)
        
        # Source directories
        visdrone_split_dir = self.visdrone_root / f"VisDrone2019-DET-{split_name}"
        images_dir = visdrone_split_dir / "images"
        annotations_dir = visdrone_split_dir / "annotations"
        
        # Destination directories
        yolo_images_dir = self.yolo_output / "images" / yolo_split
        yolo_labels_dir = self.yolo_output / "labels" / yolo_split
        
        if not images_dir.exists():
            logger.error(f"❌ Images directory not found: {images_dir}")
            return 0, 0, 1
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg"))
        logger.info(f"📸 Found {len(image_files)} images to convert")
        
        converted_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Converting {split_name}"):
            try:
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Copy image to YOLO directory
                dest_img_path = yolo_images_dir / img_path.name
                shutil.copy2(img_path, dest_img_path)
                
                # Load corresponding annotation
                annotation_path = annotations_dir / f"{img_path.stem}.txt"
                annotations = self.parse_visdrone_annotation(annotation_path)
                
                # Convert annotations to YOLO format
                yolo_annotations = []
                
                for ann in annotations:
                    object_category = ann['object_category']
                    
                    # Skip ignored regions (class 0)
                    if object_category == 0:
                        continue
                    
                    # Skip objects not in our class mapping
                    if object_category not in self.class_mapping:
                        continue
                    
                    # Convert bbox format
                    center_x, center_y, width, height = self.convert_bbox_format(
                        ann, img_width, img_height
                    )
                    
                    # Skip invalid bboxes
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Map to YOLO class ID
                    yolo_class_id = self.class_mapping[object_category]
                    
                    # Create YOLO annotation line
                    yolo_line = f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    yolo_annotations.append(yolo_line)
                
                # Write YOLO annotation file
                dest_label_path = yolo_labels_dir / f"{img_path.stem}.txt"
                with open(dest_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                converted_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {img_path.name}: {e}")
                error_count += 1
        
        logger.info(f"✅ {split_name} conversion complete:")
        logger.info(f"   • Converted: {converted_count}")
        logger.info(f"   • Skipped: {skipped_count}")
        logger.info(f"   • Errors: {error_count}")
        
        return converted_count, skipped_count, error_count
    
    def create_dataset_yaml(self):
        """
        Create YOLO dataset configuration file
        """
        logger.info("📝 Creating dataset.yaml configuration...")
        
        # Create dataset configuration
        dataset_config = {
            'path': str(self.yolo_output.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),  # number of classes
            'names': self.class_names
        }
        
        # Write YAML file
        yaml_path = self.yolo_output / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Created dataset.yaml at {yaml_path}")
        
        # Also create a classes.txt file for reference
        classes_path = self.yolo_output / "classes.txt"
        with open(classes_path, 'w') as f:
            for i, class_name in enumerate(self.class_names):
                f.write(f"{i}: {class_name}\n")
        
        logger.info(f"✅ Created classes.txt at {classes_path}")
    
    def create_file_lists(self):
        """
        Create train.txt, val.txt, test.txt files with image paths
        """
        logger.info("📋 Creating file lists...")
        
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_output / "images" / split
            
            if not images_dir.exists():
                continue
            
            # Get all image files
            image_files = list(images_dir.glob("*.jpg"))
            
            # Create file list
            file_list_path = self.yolo_output / f"{split}.txt"
            with open(file_list_path, 'w') as f:
                for img_path in sorted(image_files):
                    # Write relative path from dataset root
                    rel_path = img_path.relative_to(self.yolo_output)
                    f.write(f"{rel_path}\n")
            
            logger.info(f"✅ Created {split}.txt with {len(image_files)} images")
    
    def validate_conversion(self):
        """
        Validate the converted dataset
        """
        logger.info("🔍 Validating conversion...")
        
        validation_results = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_output / "images" / split
            labels_dir = self.yolo_output / "labels" / split
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg"))
            label_files = list(labels_dir.glob("*.txt"))
            
            # Check for matching image/label pairs
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            
            missing_labels = image_stems - label_stems
            orphaned_labels = label_stems - image_stems
            
            validation_results[split] = {
                'images': len(image_files),
                'labels': len(label_files),
                'missing_labels': len(missing_labels),
                'orphaned_labels': len(orphaned_labels)
            }
            
            logger.info(f"📊 {split.upper()} validation:")
            logger.info(f"   • Images: {len(image_files)}")
            logger.info(f"   • Labels: {len(label_files)}")
            if missing_labels:
                logger.warning(f"   • Missing labels: {len(missing_labels)}")
            if orphaned_labels:
                logger.warning(f"   • Orphaned labels: {len(orphaned_labels)}")
        
        return validation_results
    
    def get_dataset_statistics(self):
        """
        Generate statistics about the converted dataset
        """
        logger.info("📊 Generating dataset statistics...")
        
        stats = {
            'total_images': 0,
            'total_objects': 0,
            'class_counts': {class_name: 0 for class_name in self.class_names},
            'split_stats': {}
        }
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.yolo_output / "labels" / split
            
            if not labels_dir.exists():
                continue
            
            split_objects = 0
            split_class_counts = {class_name: 0 for class_name in self.class_names}
            
            label_files = list(labels_dir.glob("*.txt"))
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                    split_class_counts[class_name] += 1
                                    split_objects += 1
            
            stats['split_stats'][split] = {
                'images': len(label_files),
                'objects': split_objects,
                'class_counts': split_class_counts
            }
            
            stats['total_images'] += len(label_files)
            stats['total_objects'] += split_objects
            
            for class_name, count in split_class_counts.items():
                stats['class_counts'][class_name] += count
        
        # Print statistics
        logger.info("📈 DATASET STATISTICS:")
        logger.info("=" * 50)
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Total objects: {stats['total_objects']}")
        logger.info(f"Average objects per image: {stats['total_objects']/stats['total_images']:.1f}")
        
        logger.info("\nClass distribution:")
        for class_name, count in stats['class_counts'].items():
            percentage = count / stats['total_objects'] * 100 if stats['total_objects'] > 0 else 0
            logger.info(f"  • {class_name}: {count} ({percentage:.1f}%)")
        
        logger.info("\nSplit breakdown:")
        for split, split_stats in stats['split_stats'].items():
            logger.info(f"  • {split.upper()}: {split_stats['images']} images, {split_stats['objects']} objects")
        
        return stats
    
    def convert_full_dataset(self):
        """
        Convert the complete VisDrone dataset to YOLO format
        """
        logger.info("🚀 STARTING FULL DATASET CONVERSION")
        logger.info("=" * 60)
        
        # Step 1: Create directory structure
        self.create_directory_structure()
        
        # Step 2: Convert each split
        total_converted = 0
        total_errors = 0
        
        for split in ['train', 'val', 'test-dev']:
            converted, skipped, errors = self.convert_split(split)
            total_converted += converted
            total_errors += errors
        
        # Step 3: Create configuration files
        self.create_dataset_yaml()
        self.create_file_lists()
        
        # Step 4: Validate conversion
        validation_results = self.validate_conversion()
        
        # Step 5: Generate statistics
        stats = self.get_dataset_statistics()
        
        logger.info("🎉 CONVERSION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"✅ Total images converted: {total_converted}")
        logger.info(f"❌ Total errors: {total_errors}")
        logger.info(f"📁 Output directory: {self.yolo_output}")
        logger.info(f"📝 Dataset config: {self.yolo_output}/dataset.yaml")
        
        if total_errors == 0:
            logger.info("🌟 Conversion completed successfully!")
            logger.info("🚀 Ready for YOLO training!")
        else:
            logger.warning(f"⚠️ Conversion completed with {total_errors} errors")
        
        return stats


def quick_test_conversion():
    """
    Quick test function to convert a small subset for testing
    """
    print("🧪 QUICK TEST: Converting small subset for validation")
    print("=" * 60)
    
    # Create a test converter
    converter = VisDroneToYOLO(
        visdrone_root="data/raw/vision/visdrone",
        yolo_output="data/processed/vision/visdrone_yolo_test"
    )
    
    # Test conversion on validation set only (smallest)
    converter.create_directory_structure()
    converted, skipped, errors = converter.convert_split('val')
    
    if converted > 0:
        converter.create_dataset_yaml()
        converter.create_file_lists()
        converter.validate_conversion()
        
        print(f"\n✅ Quick test completed successfully!")
        print(f"📁 Test output: {converter.yolo_output}")
        print(f"🔍 Check the output to verify conversion is working correctly")
        print(f"🚀 If everything looks good, run full conversion")
    else:
        print(f"\n❌ Quick test failed - check your data paths")


def main():
    """
    Main function for dataset conversion
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert VisDrone to YOLO format")
    parser.add_argument("--test", action="store_true", help="Run quick test conversion")
    parser.add_argument("--visdrone-root", default="data/raw/vision/visdrone", 
                       help="Path to VisDrone dataset")
    parser.add_argument("--output", default="data/processed/vision/visdrone_yolo",
                       help="Output path for YOLO dataset")
    
    args = parser.parse_args()
    
    if args.test:
        quick_test_conversion()
    else:
        # Full conversion
        converter = VisDroneToYOLO(args.visdrone_root, args.output)
        stats = converter.convert_full_dataset()


if __name__ == "__main__":
    main()