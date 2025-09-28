# moorebotOBB - Moorebot Robot Pose usine YOLO11 Oriented Bounding Boxes

Train a YOLO11 OBB (Oriented Bounding Box) model on your custom moorebot dataset with 2 classes: `box` and `robot`.

### 2. Start Training
```bash
# Basic training (recommended for first run)
python training.py

# With custom parameters
python training.py --epochs 200 --batch-size 32 --name my_experiment
```

### 3. View Results
Training results are saved in `runs/train/[experiment_name]/`

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size (reduce if GPU memory issues) |
| `--img-size` | 640 | Input image size |
| `--lr0` | 0.01 | Initial learning rate |
| `--name` | `yolo11n_obb_custom` | Experiment name |
| `--patience` | 50 | Early stopping patience |
| `--device` | auto | Device (cpu, 0, 1, etc.) |
