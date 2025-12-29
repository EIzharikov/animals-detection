import yaml
from pathlib import Path
from collections import Counter

import torch
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO
from animals_detect.constants import PROJECT_ROOT

def load_dataset(data_yml):
    with open(data_yml) as f:
        data_cfg = yaml.safe_load(f)
    classes = data_cfg["names"]
    val_files = data_cfg.get("val", [])
    if not val_files:
        raise ValueError("Val split missing in YAML")
    val_files = [PROJECT_ROOT / Path(p) for p in val_files]
    return classes, val_files

def compute_stats(classes, val_files):
    stats = {cls: {"encounters": 0, "total": 0, "max_per_image": 0} for cls in classes}
    for img_path in val_files:
        ann_file = PROJECT_ROOT / "data/labels" / img_path.parent.name / (img_path.stem + ".txt")
        img_counter = Counter()
        if ann_file.exists():
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        img_counter[cls_id] += 1
        for cls_id, cls_name in classes.items():
            if img_counter.get(cls_id, 0) > 0:
                stats[cls_name]["encounters"] += 1
                stats[cls_name]["total"] += img_counter[cls_id]
                stats[cls_name]["max_per_image"] = max(stats[cls_name]["max_per_image"], img_counter[cls_id])
    return stats

def compute_roc_auc(model_path, classes, val_files):
    model = YOLO(model_path)
    y_true, y_score = [], []
    for img_path in val_files:
        ann_file = PROJECT_ROOT / "data/labels" / img_path.parent.name / (img_path.stem + ".txt")
        gt_cls = set()
        if ann_file.exists():
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_cls.add(int(parts[0]))
        pred = model.predict(source=str(img_path), imgsz=640, conf=0.001, verbose=False)[0]
        scores = [0.0] * len(classes)
        if pred.boxes:
            for cls_id, conf in zip(pred.boxes.cls.tolist(), pred.boxes.conf.tolist()):
                scores[int(cls_id)] = max(scores[int(cls_id)], conf)
        y_true.append([1 if i in gt_cls else 0 for i in range(len(classes))])
        y_score.append(scores)
    y_true = torch.tensor(y_true).numpy()
    y_score = torch.tensor(y_score).numpy()
    roc_auc = {}
    for i, cls_name in classes.items():
        try:
            roc_auc[cls_name] = roc_auc_score(y_true[:, i], y_score[:, i])
        except ValueError:
            roc_auc[cls_name] = None
    return roc_auc

def print_stats_table(stats, roc_auc):
    print(f"{'Class':<20} {'Encounters':<10} {'Total':<10} {'Max per image':<15} {'ROC AUC':<10}")
    for cls_name, s in stats.items():
        auc = f"{roc_auc.get(cls_name, 'N/A'):.3f}" if roc_auc.get(cls_name) is not None else "N/A"
        print(f"{cls_name:<20} {s['encounters']:<10} {s['total']:<10} {s['max_per_image']:<15} {auc:<10}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    classes, val_files = load_dataset(args.data)
    stats = compute_stats(classes, val_files)
    roc_auc = compute_roc_auc(args.weights, classes, val_files)
    print_stats_table(stats, roc_auc)
