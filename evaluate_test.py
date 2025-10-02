#!/usr/bin/env python3
"""
YOLO11 OBB Test Split Evaluation Script
Evaluate a trained YOLO11 Oriented Bounding Box model on the dataset test split
to obtain quantitative performance metrics (Precision, Recall, mAP, F1, etc.).

Usage Examples:
  python evaluate_test.py                                  # Auto-detect latest best.pt, default dataset
  python evaluate_test.py --model runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt
  python evaluate_test.py --data datasets/moorebot_v2/data.yaml --imgsz 1920 --batch 4
  python evaluate_test.py --save-json --save-txt          # Export COCO-style JSON + per-image txt predictions

Outputs:
  - Metrics printed to console
  - Ultralytics standard artifacts (confusion matrix, PR curves, etc.) in runs/obb/test*/
  - metrics_summary.csv (concise CSV of primary metrics)

Notes:
  - Requires a dataset YAML containing a 'test:' entry
  - For oriented bounding box models, the task is inferred from weights
  - Set --device '' to auto-detect GPU/CPU
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import csv


def check_and_install_dependencies():
    """Check if required packages are installed and exit with guidance if not."""
    required_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'yaml': 'pyyaml'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required packages:\n")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


check_and_install_dependencies()

from ultralytics import YOLO  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import math
import cv2
import numpy as np

# ---- Angle Error Utilities (defined early so main can call) -----------------

def polygon_angle_deg(pts: np.ndarray) -> float:
    """Compute rectangle orientation angle in degrees using longest edge."""
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v = p2 - p1
        length = np.linalg.norm(v)
        if length > 0:
            edges.append((length, v))
    if not edges:
        return 0.0
    v = max(edges, key=lambda x: x[0])[1]
    angle = math.degrees(math.atan2(v[1], v[0]))
    if angle < 0:
        angle += 180
    return angle


def polygon_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    a = pts_a.astype(np.float32)
    b = pts_b.astype(np.float32)
    def order(pts):
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
        idx = np.argsort(angles)
        return pts[idx]
    a = order(a)
    b = order(b)
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    retval, inter = cv2.intersectConvexConvex(a, b)
    if retval <= 0:
        return 0.0
    inter_area = cv2.contourArea(inter)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def load_gt_labels(label_path: Path, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array(coords, dtype=float).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            ang = polygon_angle_deg(pts)
            objs.append({'cls': cls, 'pts': pts, 'angle': ang})
    return objs


def compute_angle_error(model, images_dir: Path, iou_thresh: float, device):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = sorted([p for p in Path(images_dir).iterdir() if p.suffix.lower() in image_exts])
    if not images:
        print(f"No images found for angle metric in {images_dir}")
        return None
    labels_dir = images_dir.parent / 'labels'
    if not labels_dir.exists():
        print(f"Labels directory not found for angle metric: {labels_dir}")
        return None
    all_errors = []
    for img_path in images:
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        gt_objs = load_gt_labels(label_path, img_path)
        if not gt_objs:
            continue
        preds = model.predict(source=str(img_path), device=device, verbose=False)
        if not preds or preds[0].obb is None:
            continue
        obb = preds[0].obb
        if obb.xyxyxyxy is None:
            continue
        pred_polys = obb.xyxyxyxy.cpu().numpy()
        pred_cls = obb.cls.cpu().numpy().astype(int)
        pred_conf = obb.conf.cpu().numpy() if obb.conf is not None else np.ones(len(pred_cls))
        pred_items = []
        for poly, c, conf in zip(pred_polys, pred_cls, pred_conf):
            pts = poly.reshape(4, 2)
            ang = polygon_angle_deg(pts)
            pred_items.append({'cls': c, 'pts': pts, 'angle': ang, 'conf': conf})
        matched_gt = set()
        for p in sorted(pred_items, key=lambda x: x['conf'], reverse=True):
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_objs):
                if idx in matched_gt:
                    continue
                if gt['cls'] != p['cls']:
                    continue
                iou = polygon_iou(p['pts'], gt['pts'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                matched_gt.add(best_gt_idx)
                gt_angle = gt['angle']
                pred_angle = p['angle']
                diff = abs(pred_angle - gt_angle)
                diff = min(diff, 180 - diff)
                if diff > 90:
                    diff = 180 - diff
                all_errors.append(diff)
    if not all_errors:
        print("No matched predictions for angle metric (check IoU threshold or predictions).")
        return None
    arr = np.array(all_errors, dtype=float)
    stats = {
        'count': float(len(arr)),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(arr.max())
    }
    return stats


def find_latest_best(weights_glob: str = 'runs/train/*/weights/best.pt') -> str | None:
    paths = [p for p in Path('.').glob(weights_glob)]
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(paths[0])


def load_dataset_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def verify_test_split(data_yaml: str):
    cfg = load_dataset_yaml(data_yaml)
    if 'test' not in cfg:
        raise ValueError("Dataset YAML has no 'test' entry. Add test: path/to/test/images")
    base = Path(data_yaml).parent
    test_rel = cfg['test']
    test_path = (base / test_rel).resolve() if not os.path.isabs(test_rel) else Path(test_rel)
    if not test_path.exists():
        # Attempt recovery if legacy '../' style was used incorrectly
        recovery_note = ''
        if test_rel.startswith('../'):
            alt = (base / test_rel.lstrip('../')).resolve()
            if alt.exists():
                print(f"Warning: Adjusted test path from '{test_rel}' to '{alt}' (legacy '../' removed)")
                test_path = alt
                recovery_note = ' (recovered)'
        if not test_path.exists():
            raise FileNotFoundError(f"Test images path not found: {test_path} (derived from '{test_rel}')\n"
                                    f"Hint: If your dataset folders are inside the same directory as the YAML, use 'test: test/images'")
    img_count = len([f for f in test_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if img_count == 0:
        print(f"Warning: No images found in test directory: {test_path}")
    return test_path, img_count, cfg


def summarize_metrics(results) -> dict:
    """Extract key metrics from Ultralytics val results object."""
    metrics = {}
    # Ultralytics stores metrics in results.results_dict or results.metrics
    if hasattr(results, 'results_dict') and results.results_dict:
        raw = results.results_dict
        # Filter only numeric
        for k, v in raw.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
    else:
        # Fallback keys (YOLOv8 style)
        possible = [
            'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'fitness'
        ]
        for k in possible:
            if hasattr(results, k.replace('/', '_').replace('(', '_').replace(')', '_')):
                metrics[k] = float(getattr(results, k))
    return metrics


def write_metrics_csv(save_dir: Path, metrics: dict):
    csv_path = save_dir / 'metrics_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{v:.6f}"])
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO11 OBB model on test split to report metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_test.py
  python evaluate_test.py --model models/yolo11n_obb_custom_20250927_124420.pt
  python evaluate_test.py --data datasets/moorebot_v2/data.yaml --imgsz 1920 --batch 4
  python evaluate_test.py --conf 0.25 --iou 0.45 --save-json
        """
    )
    parser.add_argument('--model', type=str, default='./runs/train/yolo11n-moorebot_v2-obb-v1/weights/best.pt', help='Path to trained weights .pt (auto-detect latest if omitted)')
    parser.add_argument('--data', type=str, default='datasets/moorebot_v2/data.yaml', help='Dataset YAML with test split')
    parser.add_argument('--imgsz', type=int, default=1920, help='Image size for evaluation')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold (lower for metrics aggregation)')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS during evaluation')
    parser.add_argument('--device', type=str, default='', help='Device ("", cpu, 0, 1, etc.)')
    parser.add_argument('--save-json', action='store_true', help='Save COCO-style JSON results (if supported)')
    parser.add_argument('--save-txt', action='store_true', help='Save per-image prediction .txt files')
    parser.add_argument('--project', type=str, default='runs/obb', help='Base project directory for outputs')
    parser.add_argument('--name', type=str, default='test', help='Base run name (timestamp appended)')
    parser.add_argument('--verbose', action='store_true', help='Verbose Ultralytics logging')
    parser.add_argument('--angle-metric', action='store_true', help='Compute oriented box angle error Δθ (additional pass)')
    parser.add_argument('--angle-iou-thresh', type=float, default=0.1, help='IoU threshold to match prediction to GT for angle error')

    args = parser.parse_args()

    # Resolve / find model
    if not args.model:
        print("No --model specified, searching for latest runs/train/*/weights/best.pt ...")
        latest = find_latest_best()
        if not latest:
            print("Error: Could not find any best.pt under runs/train/*/weights/. Provide --model explicitly.")
            sys.exit(1)
        args.model = latest
        print(f"Using model: {args.model}")
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    # Verify test split
    try:
        test_path, img_count, cfg = verify_test_split(args.data)
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        sys.exit(1)

    print("\nTest Split Information:")
    print(f"  test images dir : {test_path}")
    print(f"  image count     : {img_count}")
    print(f"  classes (nc)    : {cfg.get('nc')} -> {cfg.get('names')}")

    # Device
    device = args.device if args.device else ('0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available() and device != 'cpu':
        print(f"CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading model ...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    print("Model loaded successfully.")

    # Prepare run naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.name}_{timestamp}"

    print("\nStarting evaluation on test split ...")
    try:
        results = model.val(
            data=args.data,
            split='test',
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            conf=args.conf,
            iou=args.iou,
            save_json=args.save_json,
            save_txt=args.save_txt,
            project=args.project,
            name=run_name,
            exist_ok=True,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path(args.project) / run_name
    print(f"\nArtifacts saved to: {save_dir}")

    # Summarize metrics
    metrics = summarize_metrics(results)
    if not metrics:
        print("Warning: Could not extract metrics dictionary from results object.")
    else:
        print("\nPrimary Metrics:")
        # Common nice-order keys if they exist
        preferred_order = [
            'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'box/precision', 'box/recall', 'box/mAP50', 'box/mAP50-95',
            'fitness'
        ]
        printed = set()
        for key in preferred_order:
            if key in metrics:
                print(f"  {key:22s}: {metrics[key]:.4f}")
                printed.add(key)
        # Print any remaining metrics
        for key, val in sorted(metrics.items()):
            if key not in printed:
                print(f"  {key:22s}: {val:.4f}")

        csv_path = write_metrics_csv(save_dir, metrics)
        print(f"\nCSV summary written: {csv_path}")

    # Helpful pointers
    print("\nAdditional Outputs (if generated):")
    for fname in [
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'BoxPR_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 'BoxF1_curve.png',
        'results.png', 'results.csv'
    ]:
        p = save_dir / fname
        if p.exists():
            print(f"  - {p.name}")

    print("\nEvaluation complete.")

    # Optional angle error computation
    if args.angle_metric:
        try:
            print("\nComputing angle error (Δθ) across test set ...")
            angle_stats = compute_angle_error(
                model=model,
                images_dir=test_path,
                iou_thresh=args.angle_iou_thresh,
                device=device
            )
            if angle_stats:
                print("\nAngle Error (Δθ) Statistics (degrees):")
                for k, v in angle_stats.items():
                    print(f"  {k:12s}: {v:.4f}")
                # Append to metrics_summary.csv if it exists
                csv_path = save_dir / 'metrics_summary.csv'
                try:
                    with open(csv_path, 'a', newline='') as f:
                        f.write('\n')
                        for k, v in angle_stats.items():
                            f.write(f"angle/{k},{v:.6f}\n")
                    print(f"Δθ stats appended to {csv_path}")
                except Exception as e:
                    print(f"Could not append angle metrics to CSV: {e}")
        except Exception as e:
            print(f"Angle metric computation failed: {e}")
    else:
        print("(Skip angle metric: enable with --angle-metric)")


if __name__ == '__main__':
    main()

# ---- Angle Error Utilities -------------------------------------------------

def compute_angle_error(model, images_dir: Path, iou_thresh: float, device):
    """Compute oriented box angle error Δθ.

    Steps:
      1. For each test image, load GT polygons from labels directory (YOLO OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4 normalized).
      2. Run inference to get predicted polygons (results[0].obb.xyxyxyxy).
      3. Greedy match predictions to GT of same class by maximum polygon IoU (>= iou_thresh).
      4. Angle = orientation of longest edge; Δθ normalized to [0, 90] by min(|Δθ|, 180-|Δθ|).
      5. Aggregate statistics.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = sorted([p for p in Path(images_dir).iterdir() if p.suffix.lower() in image_exts])
    if not images:
        print(f"No images found for angle metric in {images_dir}")
        return None

    # Infer labels directory relative to images directory
    labels_dir = images_dir.parent / 'labels'
    if not labels_dir.exists():
        print(f"Labels directory not found for angle metric: {labels_dir}")
        return None

    all_errors = []

    for img_path in images:
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        gt_objs = load_gt_labels(label_path, img_path)
        if not gt_objs:
            continue

        # Predict
        preds = model.predict(source=str(img_path), device=device, verbose=False)
        if not preds or preds[0].obb is None:
            continue
        obb = preds[0].obb
        if obb.xyxyxyxy is None:
            continue
        pred_polys = obb.xyxyxyxy.cpu().numpy()  # shape (N,8)
        pred_cls = obb.cls.cpu().numpy().astype(int)
        pred_conf = obb.conf.cpu().numpy() if obb.conf is not None else np.ones(len(pred_cls))

        pred_items = []
        for poly, c, conf in zip(pred_polys, pred_cls, pred_conf):
            pts = poly.reshape(4, 2)
            ang = polygon_angle_deg(pts)
            pred_items.append({'cls': c, 'pts': pts, 'angle': ang, 'conf': conf})

        # Match predictions to GT greedily by IoU per class
        matched_gt = set()
        for p in sorted(pred_items, key=lambda x: x['conf'], reverse=True):
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_objs):
                if idx in matched_gt:
                    continue
                if gt['cls'] != p['cls']:
                    continue
                iou = polygon_iou(p['pts'], gt['pts'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                matched_gt.add(best_gt_idx)
                gt_angle = gt['angle']
                pred_angle = p['angle']
                diff = abs(pred_angle - gt_angle)
                # Orientation symmetry (rectangle) -> normalize
                diff = min(diff, 180 - diff)
                # Occasionally angle definitions differ by 90; tighten to <=90
                if diff > 90:
                    diff = 180 - diff
                all_errors.append(diff)

    if not all_errors:
        print("No matched predictions for angle metric (check IoU threshold or predictions).")
        return None

    arr = np.array(all_errors, dtype=float)
    stats = {
        'count': float(len(arr)),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(arr.max())
    }
    return stats


def load_gt_labels(label_path: Path, img_path: Path):
    """Load ground truth OBB labels (YOLO normalized quadrilateral format)."""
    try:
        import cv2  # Local ensure
    except Exception:
        pass
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    objs = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array(coords, dtype=float).reshape(4, 2)
            # Denormalize
            pts[:, 0] *= w
            pts[:, 1] *= h
            ang = polygon_angle_deg(pts)
            objs.append({'cls': cls, 'pts': pts, 'angle': ang})
    return objs


def polygon_angle_deg(pts: np.ndarray) -> float:
    """Compute rectangle orientation angle in degrees using longest edge."""
    # Ensure shape (4,2)
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)
    # Compute edge vectors (cyclic)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v = p2 - p1
        length = np.linalg.norm(v)
        if length > 0:
            edges.append((length, v))
    if not edges:
        return 0.0
    # Longest edge defines orientation
    v = max(edges, key=lambda x: x[0])[1]
    angle = math.degrees(math.atan2(v[1], v[0]))
    # Normalize to [0,180)
    if angle < 0:
        angle += 180
    return angle


def polygon_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """IoU for two convex quadrilaterals using cv2.intersectConvexConvex."""
    a = pts_a.astype(np.float32)
    b = pts_b.astype(np.float32)
    # Ensure consistent point ordering (cv2 expects either clockwise or ccw). We'll assume given order works; if fails, reorder by centroid angle.
    def order(pts):
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
        idx = np.argsort(angles)
        return pts[idx]
    a = order(a)
    b = order(b)
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    retval, inter = cv2.intersectConvexConvex(a, b)
    if retval <= 0:
        return 0.0
    inter_area = cv2.contourArea(inter)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)
