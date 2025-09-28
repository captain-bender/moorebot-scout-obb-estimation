#!/usr/bin/env python3
"""
YOLO11 OBB (Oriented Bounding Box) Custom Training Script
Train a YOLO11n-obb model on custom dataset

Usage:
    python training.py                           # Use default settings
    python training.py --epochs 200 --batch-size 32  # Custom settings
    python training.py --help                    # Show all options

First time setup:
    pip install ultralytics torch torchvision
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
from datetime import datetime

def check_and_install_dependencies():
    """Check if required packages are installed and provide installation guidance"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'yaml': 'pyyaml'
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
        print("\nFor a complete installation with all dependencies:")
        print("pip install ultralytics torch torchvision")
        print("\nThen run this script again.")
        sys.exit(1)
    
    print("All required packages are installed!")

# Check dependencies first
check_and_install_dependencies()

# Now import the packages
from ultralytics import YOLO
import torch
import yaml

def setup_directories():
    """Create necessary directories for training outputs"""
    directories = ['runs', 'runs/train', 'models']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def check_dataset_config(data_path):
    """Verify dataset configuration and paths"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset configuration:")
    print(f"  Classes: {config.get('nc', 'Unknown')}")
    print(f"  Names: {config.get('names', 'Unknown')}")
    
    # Verify dataset paths exist - handle relative paths correctly
    dataset_dir = os.path.dirname(data_path)
    for split in ['train', 'val', 'test']:
        if split in config:
            # Handle relative paths that might start with ../
            raw_path = config[split]
            if raw_path.startswith('../'):
                split_path = os.path.join(dataset_dir, raw_path)
            else:
                split_path = os.path.join(dataset_dir, raw_path)
            
            split_path = os.path.normpath(split_path)  # Clean up path
            
            if os.path.exists(split_path):
                image_count = len([f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {split.capitalize()}: {image_count} images found at {split_path}")
            else:
                print(f"  Warning: {split} path not found: {split_path}")
    
    return config

def main():
    # Display helpful banner
    print("YOLO11 OBB Custom Training")
    print("=" * 60)
    print("Training YOLO11 nano OBB model on your custom dataset")
    print("Classes: box, robot")
    print("=" * 60)
    print()
    
    parser = argparse.ArgumentParser(
        description='Train YOLO11 OBB model on custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training.py                                    # Default settings
  python training.py --epochs 200                      # Train for 200 epochs  
  python training.py --batch-size 32 --epochs 150      # Larger batch, more epochs
        """
    )
    parser.add_argument('--data', type=str, default='./datasets/moorebot_v1/data.yaml', help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolo11n-obb.pt', help='Model to use (yolo11n-obb.pt, yolo11s-obb.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100,help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=1920,help='Image size for training')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, 0, 1, etc.). Empty for auto-detect')
    parser.add_argument('--workers', type=int, default=16, help='Number of workers')

    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=25, help='Save checkpoint every N epochs')
    parser.add_argument('--cache', type=str, default='ram', help='Cache images in ram/disk/False')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup augmentation probability (disabled for OBB)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')
    parser.add_argument('--degrees', type=float, default=10.0, help='Image rotation degrees')
    parser.add_argument('--translate', type=float, default=0.2, help='Image translation fraction')
    parser.add_argument('--scale', type=float, default=0.9, help='Image scale gain')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')
    
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='yolo11n-moorebot_v1-obb-v2', help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Check CUDA availability
    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify dataset
    print("\nDataset verification:")
    dataset_config = check_dataset_config(args.data)
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    model = YOLO(args.model)  # Load pretrained model
    
    # Print model info
    # print(f"Model architecture: {model.model}")
    
    # Training configuration
    train_config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # Better for high-res training
        'lr0': args.lr0,
        'patience': args.patience,
        'save_period': args.save_period,
        'cache': args.cache,
        'mixup': args.mixup,
        'mosaic': args.mosaic,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'fliplr': args.fliplr,
        'close_mosaic': 15,  # Stop mosaic in last 15 epochs
        'amp': True,  # Automatic Mixed Precision for memory efficiency
        'fraction': 1.0,  # Use full dataset
        'seed': 42,  # Reproducible results
    }
    
    print("\nTraining configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    # Start training
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    try:
        results = model.train(**train_config)
        
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"{'='*60}")
        print(f"Results saved to: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal metrics:")
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"models/yolo11n_obb_custom_{timestamp}.pt"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()