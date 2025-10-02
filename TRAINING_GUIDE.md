# YOLO11 OBB Training Guide - Detailed Documentation

This comprehensive guide covers advanced training configurations, optimization strategies, and troubleshooting for YOLO11 Oriented Bounding Box detection on the Moorebot dataset.

## Training Configuration

### Current Optimized Settings
The training script is configured for **high-accuracy** oriented bounding box detection:

```bash
# Default high-accuracy training
python training.py
```

| Parameter | Default Value | Description | Rationale |
|-----------|---------------|-------------|-----------|
| `--img-size` | 1920 | Native resolution training | Matches 1920x1080 source images |
| `--batch-size` | 4 | Small batch for high-res | Memory optimization for large images |
| `--epochs` | 100 | Training duration | Balanced training time |
| `--lr0` | 0.001 | Initial learning rate | Lower LR for stable high-res training |
| `--optimizer` | AdamW | Optimizer choice | Better convergence for complex geometry |
| `--workers` | 16 | Data loading threads | Fast data pipeline |
| `--cache` | ram | Data caching | Speed up training |
| `--mixup` | 0.0 | Mixup augmentation | Disabled for OBB precision |

### Training Focus: **Accuracy over Speed**
- High-resolution native image processing (1980px)
- Precision-oriented augmentation strategy
- Memory-optimized batch processing
- OBB-specific parameter tuning

## Dataset Information

### Dataset Versions

#### **moorebot_v1** (Grayscale Dataset)
- **Images**: 429 total
- **Classes**: `box`, `robot`
- **Preprocessing**: Converted to grayscale (CRT phosphor)
- **Use Case**: Memory-efficient training, geometric focus
- **Path**: `datasets/moorebot_v1/`

#### **moorebot_v2** (RGB Dataset) - **Currently Used**
- **Images**: 178 total  
- **Classes**: `box`, `robot`
- **Preprocessing**: None (original RGB preserved)
- **Use Case**: Production-ready, full color information
- **Path**: `datasets/moorebot_v2/`

### Dataset Structure
```
datasets/moorebot_v2/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images (RGB)
│   └── labels/        # OBB annotation files
├── valid/
│   ├── images/        # Validation images
│   └── labels/        # OBB annotation files
└── test/
    ├── images/        # Test images
    └── labels/        # OBB annotation files
```

## ⚙️ Advanced Parameters

### Augmentation Settings (OBB-Optimized)
```python
# Current optimized settings
mosaic: 1.0        # Effective for OBB training
degrees: 10.0      # Rotation helps OBB learn orientations  
translate: 0.2     # Spatial variation
scale: 0.9         # Size variation
fliplr: 0.5        # Horizontal symmetry
mixup: 0.0         # Disabled for OBB precision
```

### Training Hyperparameters
```python
# Optimizer settings
optimizer: 'AdamW'           # Best for high-res OBB training
lr0: 0.001                   # Conservative learning rate
weight_decay: 0.0005         # Regularization
warmup_epochs: 3.0           # Gradual learning rate increase

# Training strategy
patience: 50                 # Early stopping
save_period: 25              # Checkpoint frequency
close_mosaic: 15             # Stop complex augmentation late
amp: True                    # Mixed precision training
```
