# moorebotOBB - YOLO11 Oriented Bounding Box Detection

A custom oriented bounding box detection project using YOLO11 for detecting and tracking Moorebot scout robots and boxes for robotic applications and automation.

**Classes**: `box`, `robot`  
**Model**: YOLO11n-obb (nano)  
**Dataset**: RGB images (1920x1080 native resolution)

## Quick Start

### Install Dependencies
```bash
pip install ultralytics torch torchvision
```

## Training

**[Complete Training Guide](TRAINING_GUIDE.md)** - Detailed documentation.

### View Results
Training results are saved in `runs/train/[experiment_name]/`

## Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4 | Batch size (optimized for high-res) |
| `--img-size` | 1920 | Input image size (native resolution) |
| `--lr0` | 0.001 | Initial learning rate |
| `--device` | auto | Device (auto-detects GPU/CPU) |

## Dataset
- [Roboflow Universe - Moorebot OBB v2](https://app.roboflow.com/moorebot-scout/moorebot-obb-ncoi8/3)
- [Roboflow Universe - Moorebot OBB v1](https://app.roboflow.com/moorebot-scout/moorebot-obb-ncoi8/2)

## Testing Trained Models (`test_model.py`)

After training, you can run inference on individual images or an entire folder of images using the testing script.

### Basic Usage
```bash
python test_model.py --image datasets/moorebot_v2/test/images/your_image.jpg
```
If you do not pass `--model`, the script will automatically pick the most recent `best.pt` in `runs/train/*/weights/`.

### Specify a Model Explicitly
```bash
python test_model.py \
	--model runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt \
	--image datasets/moorebot_v2/test/images/your_image.jpg
```

### Run on a Folder of Images
```bash
python test_model.py --model models/yolo11n_obb_custom_20250927_124420.pt \
	--source datasets/moorebot_v2/test/images/
```

### Useful Flags
| Flag | Description |
|------|-------------|
| `--conf 0.5` | Set confidence threshold (default 0.25) |
| `--iou 0.45` | IoU threshold for NMS |
| `--no-show` | Do not open image windows (batch mode) |
| `--no-save` | Do not save annotated images |
| `--output runs/test` | Base directory for annotated outputs |

### Output
Annotated images are saved to: `runs/test/test_<timestamp>/annotated_<original_name>.jpg` (unless `--no-save` is used). Console output lists detections with class and confidence.

### Classes & Colors (default)
- box → green oriented box
- robot → blue oriented box
 
## Evaluate on Test Split (`evaluate_test.py`)

Run quantitative metrics (Precision, Recall, mAP, F1, curves) on the dataset `test` split defined in your `data.yaml`.

### Basic Usage
```bash
python evaluate_test.py
```
Automatically finds the most recent `runs/train/*/weights/best.pt` and uses `datasets/moorebot_v2/data.yaml`.

### Specify Model & Dataset
```bash
python evaluate_test.py \
	--model runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt \
	--data datasets/moorebot_v2/data.yaml
```

### Additional Options
| Option | Description |
|--------|-------------|
| `--imgsz 1920` | Image size for evaluation |
| `--batch 4` | Batch size |
| `--conf 0.001` | Confidence threshold (keep low for metrics) |
| `--iou 0.7` | IoU threshold for NMS |
| `--save-json` | Export COCO-style JSON predictions |
| `--save-txt` | Save per-image YOLO-format predictions |
| `--project runs/obb` | Output base directory |
| `--name test` | Base run name (timestamp appended) |

### Outputs
Metrics and artifacts saved to: `runs/obb/test_<timestamp>/`

Includes (when generated):
```
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png / BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png
results.png / results.csv
metrics_summary.csv (concise extracted metrics)
```

Use this after training to track performance progression across versions.
---
