#!/usr/bin/env python3
"""
YOLO11 OBB Model Testing Script
Test trained YOLO OBB models on individual images

Usage:
    python test_model.py --model path/to/best.pt --image path/to/image.jpg
    python test_model.py --model models/yolo11n_obb_custom_*.pt --image datasets/moorebot_v2/test/images/
    python test_model.py --model runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt --source datasets/moorebot_v2/test/images/
    python test_model.py --help

Features:
    - Test single images or entire directories
    - Visualize oriented bounding boxes
    - Save annotated results
    - Display confidence scores
    - Support for both image files and directories
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import glob

def check_and_install_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages!")
        print("\nPlease install the missing packages by running:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

# Check dependencies first
check_and_install_dependencies()

# Now import the packages
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import numpy as np


def find_best_model():
    """Find the most recent best.pt model in runs/train"""
    train_dirs = glob.glob('runs/train/*/weights/best.pt')
    if not train_dirs:
        return None
    
    # Sort by modification time
    train_dirs.sort(key=os.path.getmtime, reverse=True)
    return train_dirs[0]


def get_image_files(source):
    """Get list of image files from source (file or directory)"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    source_path = Path(source)
    
    if source_path.is_file():
        if source_path.suffix.lower() in image_extensions:
            return [str(source_path)]
        else:
            print(f"Error: {source} is not a valid image file")
            return []
    
    elif source_path.is_dir():
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        return [str(f) for f in sorted(image_files)]
    
    else:
        print(f"Error: {source} is not a valid file or directory")
        return []


def draw_obb_on_image(image_path, results, save_path=None, show=True):
    """
    Draw oriented bounding boxes on image with labels and confidence scores
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Get OBB results
    if len(results) > 0 and results[0].obb is not None:
        obb = results[0].obb
        
        # Get boxes, confidences, and class IDs
        boxes = obb.xyxyxyxy.cpu().numpy() if obb.xyxyxyxy is not None else []
        confs = obb.conf.cpu().numpy() if obb.conf is not None else []
        cls_ids = obb.cls.cpu().numpy() if obb.cls is not None else []
        
        # Get class names
        class_names = results[0].names
        
        # Draw each oriented bounding box
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            # Convert box coordinates to integer points
            points = box.reshape((-1, 1, 2)).astype(np.int32)
            
            # Choose color based on class
            cls_id_int = int(cls_id)
            if cls_id_int == 0:  # box
                color = (0, 255, 0)  # Green
            elif cls_id_int == 1:  # robot
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red for unknown
            
            # Draw the oriented bounding box
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            
            # Prepare label with class name and confidence
            class_name = class_names[cls_id_int]
            label = f"{class_name}: {conf:.2f}"
            
            # Get label size for background rectangle
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Find the top-left point of the box for label placement
            top_left = np.min(points, axis=0)[0]
            label_x, label_y = int(top_left[0]), int(top_left[1])
            
            # Ensure label is within image bounds
            label_y = max(label_height + 5, label_y)
            
            # Draw background rectangle for label
            cv2.rectangle(
                img,
                (label_x, label_y - label_height - 5),
                (label_x + label_width, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (label_x, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        print(f"  Detected {len(boxes)} object(s):")
        for i, (conf, cls_id) in enumerate(zip(confs, cls_ids)):
            class_name = class_names[int(cls_id)]
            print(f"    - {class_name}: {conf:.3f}")
    else:
        print("  No objects detected")
    
    # Save annotated image if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"  Saved annotated image to: {save_path}")
    
    # Display image if requested
    if show:
        # Resize if image is too large
        max_display_size = 1200
        if max(img.shape[:2]) > max_display_size:
            scale = max_display_size / max(img.shape[:2])
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            display_img = cv2.resize(img, (new_width, new_height))
        else:
            display_img = img
        
        window_name = f"Detection: {Path(image_path).name}"
        cv2.imshow(window_name, display_img)
        print(f"  Press any key to continue (window: {window_name})...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


def test_model_on_images(model_path, image_sources, conf_threshold=0.25, 
                         iou_threshold=0.45, save_results=True, show_results=True,
                         output_dir='runs/test'):
    """
    Test model on multiple images
    
    Args:
        model_path: Path to trained model (.pt file)
        image_sources: List of image paths or directories
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_results: Whether to save annotated images
        show_results: Whether to display results
        output_dir: Directory to save results
    """
    print("="*80)
    print("YOLO11 OBB Model Testing")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path}")
    
    # Load model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Check device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get all image files
    all_images = []
    for source in image_sources:
        images = get_image_files(source)
        all_images.extend(images)
    
    if not all_images:
        print("No valid images found!")
        sys.exit(1)
    
    print(f"\nFound {len(all_images)} image(s) to process")
    
    # Create output directory if saving results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_output_dir = os.path.join(output_dir, f"test_{timestamp}")
        os.makedirs(test_output_dir, exist_ok=True)
        print(f"Results will be saved to: {test_output_dir}")
    
    # Process each image
    print("\n" + "="*80)
    print("Processing images...")
    print("="*80)
    
    detection_summary = {
        'total_images': len(all_images),
        'images_with_detections': 0,
        'total_detections': 0,
        'class_counts': {}
    }
    
    for idx, image_path in enumerate(all_images, 1):
        print(f"\n[{idx}/{len(all_images)}] Processing: {Path(image_path).name}")
        
        try:
            # Run inference
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )
            
            # Count detections
            if len(results) > 0 and results[0].obb is not None:
                num_detections = len(results[0].obb.conf)
                if num_detections > 0:
                    detection_summary['images_with_detections'] += 1
                    detection_summary['total_detections'] += num_detections
                    
                    # Count by class
                    cls_ids = results[0].obb.cls.cpu().numpy()
                    class_names = results[0].names
                    for cls_id in cls_ids:
                        class_name = class_names[int(cls_id)]
                        detection_summary['class_counts'][class_name] = \
                            detection_summary['class_counts'].get(class_name, 0) + 1
            
            # Prepare save path
            save_path = None
            if save_results:
                save_path = os.path.join(test_output_dir, f"annotated_{Path(image_path).name}")
            
            # Draw and display results
            draw_obb_on_image(image_path, results, save_path=save_path, show=show_results)
            
        except Exception as e:
            print(f"  Error processing image: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("Testing Summary")
    print("="*80)
    print(f"Total images processed: {detection_summary['total_images']}")
    print(f"Images with detections: {detection_summary['images_with_detections']}")
    print(f"Total detections: {detection_summary['total_detections']}")
    if detection_summary['class_counts']:
        print("\nDetections by class:")
        for class_name, count in detection_summary['class_counts'].items():
            print(f"  {class_name}: {count}")
    
    if save_results:
        print(f"\nAnnotated images saved to: {test_output_dir}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Test YOLO11 OBB model on individual images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Test on a single image with default model
        python test_model.py --image datasets/moorebot_v2/test/images/image1.jpg
        
        # Test on specific model
        python test_model.py --model runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt --image test.jpg
        
        # Test on all images in a directory
        python test_model.py --model models/yolo11n_obb_custom_20250927_124420.pt --source datasets/moorebot_v2/test/images/
        
        # Test without showing images (faster for batch processing)
        python test_model.py --source datasets/moorebot_v2/test/images/ --no-show
        
        # Test with custom confidence threshold
        python test_model.py --image test.jpg --conf 0.5
        """
    )

    parser.add_argument('--model', type=str, default="./runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt",
                        help='Path to trained model (.pt file). If not specified, uses most recent best.pt')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image file (alternative to --source)')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to image file or directory containing images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save annotated images')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display images (useful for batch processing)')
    parser.add_argument('--output', type=str, default='runs/test',
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model is None:
        print("No model specified, searching for most recent best.pt...")
        args.model = find_best_model()
        if args.model is None:
            print("\nError: No trained model found!")
            print("Please specify a model with --model flag or train a model first.")
            print("\nAvailable models:")
            print("  - runs/train/*/weights/best.pt (training runs)")
            print("  - models/*.pt (saved models)")
            sys.exit(1)
        print(f"Found model: {args.model}")
    
    # Determine image sources
    image_sources = []
    if args.image:
        image_sources.append(args.image)
    if args.source:
        image_sources.append(args.source)
    
    if not image_sources:
        print("\nError: No image source specified!")
        print("Please provide either --image or --source")
        print("\nExamples:")
        print("  python test_model.py --image path/to/image.jpg")
        print("  python test_model.py --source path/to/images/directory/")
        sys.exit(1)
    
    # Run testing
    test_model_on_images(
        model_path=args.model,
        image_sources=image_sources,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_results=not args.no_save,
        show_results=not args.no_show,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
