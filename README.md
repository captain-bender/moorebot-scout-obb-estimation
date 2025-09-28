# moorebotOBB - Moorebot Robot Pose usine YOLO11 Oriented Bounding Boxes

A custom oriented bounding box project using YOLO11 for detecting and tracking Moorebot scout robot for robotic applications and automation.
The custom moorebot dataset contains 2 classes: `box` and `robot`.

### Training
```bash
# Basic training (recommended for first run)
python training.py

# With custom parameters
python training.py --epochs 200 --batch-size 32 --name my_experiment
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size (reduce if GPU memory issues) |
| `--img-size` | 640 | Input image size |
| `--lr0` | 0.01 | Initial learning rate |
| `--name` | `yolo11n_obb_custom` | Experiment name |
| `--patience` | 50 | Early stopping patience |
| `--device` | auto | Device (cpu, 0, 1, etc.) |

### Datasets

The datasets can be found in Roboflow Universe:

- [Version1](https://app.roboflow.com/moorebot-scout/moorebot-obb-ncoi8/2): Full training dataset applying the usual preprocessing and augmentation steps