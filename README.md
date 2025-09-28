# moorebotOBB - Moorebot Robot Pose usine YOLO11 Oriented Bounding Boxes

A custom oriented bounding box project using YOLO11 for detecting and tracking Moorebot scout robot for robotic applications and automation.
The custom moorebot dataset contains 2 classes: `box` and `robot`.

### Training (Optimized for High Accuracy)
```bash
# High-accuracy training with default settings (recommended)
python training.py
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4 | Batch size (optimized for high-res images) |
| `--img-size` | 1280 | Input image size (high-resolution for accuracy) |
| `--device` | auto | Device (auto-detects GPU/CPU) |
| `--workers` | 8 | Number of data loading workers |
| `--name` | `yolo11n-moorebot_v1-obb-v1` | Experiment name |

### Datasets

The datasets can be found in Roboflow Universe:

- [Version1](https://app.roboflow.com/moorebot-scout/moorebot-obb-ncoi8/2): Full training dataset applying the usual preprocessing and augmentation steps