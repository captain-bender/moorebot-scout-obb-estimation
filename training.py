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
    
    # Verify dataset paths exist
    dataset_dir = os.path.dirname(data_path)
    for split in ['train', 'val', 'test']:
        if split in config:
            split_path = os.path.join(dataset_dir, config[split])
            if os.path.exists(split_path):
                image_count = len([f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {split.capitalize()}: {image_count} images found")
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
  python training.py --name my_experiment --lr0 0.001  # Custom name and learning rate
  python training.py --resume runs/train/exp/weights/last.pt  # Resume training

Common configurations:
  Fast training:    --epochs 50 --batch-size 8
  Balanced:         --epochs 100 --batch-size 16 (default)
  High quality:     --epochs 200 --batch-size 32
        """
    )
    parser.add_argument('--data', type=str, default='datasets/moorebot_v1/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolo11n-obb.pt',
                       help='Model to use (yolo11n-obb.pt, yolo11s-obb.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=1280,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (cpu, 0, 1, etc.). Empty for auto-detect')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save model every N epochs')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='yolo11n_obb_custom',
                       help='Experiment name')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    print("=" * 60)
    print("YOLO11 OBB Custom Training")
    print("=" * 60)
    
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
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
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
        'lr0': args.lr0,
        'patience': args.patience,
        'save_period': args.save_period,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,  # Disable mosaic augmentation for final epochs
        'resume': bool(args.resume),
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
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