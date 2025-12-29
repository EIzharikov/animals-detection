import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ultralytics import YOLO


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_names(names):
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: v for i, v in enumerate(names)}


def collect_images(val_entry, data_root):
    val_path = (data_root / val_entry).resolve()
    print(val_path)
    if val_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(p for p in val_path.rglob("*") if p.suffix.lower() in exts)
    if val_path.is_file():
        with open(val_path) as f:
            return [
                (data_root.parent / line.strip()).resolve()
                for line in f
                if line.strip()
            ]
    raise ValueError("Invalid val path")


def label_path_from_image(img_path):
    parts = list(img_path.parts)
    if "images" not in parts:
        raise ValueError(f"Invalid YOLO image path: {img_path}")
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def compute_table(classes, images):
    stats = {
        name: {"encounters": 0, "total": 0, "max_per_image": 0}
        for name in classes.values()
    }
    for img in images:
        ann = label_path_from_image(img)
        counter = Counter()
        if ann.exists():
            with open(ann) as f:
                for line in f:
                    cid = int(line.split()[0])
                    counter[cid] += 1
        for cid, cname in classes.items():
            n = counter.get(cid, 0)
            if n > 0:
                stats[cname]["encounters"] += 1
                stats[cname]["total"] += n
                stats[cname]["max_per_image"] = max(stats[cname]["max_per_image"], n)
    return stats


def compute_roc_auc(weights, classes, images):
    model = YOLO(weights)
    y_true = []
    y_score = []

    for img in tqdm(images):
        ann = label_path_from_image(img)
        gt = set()
        if ann.exists():
            with open(ann) as f:
                for line in f:
                    gt.add(int(line.split()[0]))

        result = model.predict(
            source=str(img),
            imgsz=640,
            conf=0.001,
            device=model.device,
            verbose=False,
        )[0]

        scores = np.zeros(len(classes), dtype=float)
        if result.boxes is not None and len(result.boxes) > 0:
            for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                c = int(c)
                scores[c] = max(scores[c], s)

        y_true.append([1 if i in gt else 0 for i in range(len(classes))])
        y_score.append(scores)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    auc = {}
    for i, name in classes.items():
        try:
            auc[name] = roc_auc_score(y_true[:, i], y_score[:, i])
        except ValueError:
            auc[name] = None
    return auc


def print_table(stats, auc):
    header = f"{'Class':<20} {'Encounters':<12} {'Total':<10} {'Max/Image':<12} {'ROC AUC':<8}"
    print(header)
    print("-" * len(header))
    for cls, s in stats.items():
        a = auc.get(cls)
        a = f"{a:.3f}" if a is not None else "N/A"
        print(
            f"{cls:<20} "
            f"{s['encounters']:<12} "
            f"{s['total']:<10} "
            f"{s['max_per_image']:<12} "
            f"{a:<8}"
        )


def save_excel(stats, auc, output_path):
    rows = []
    for cls, s in stats.items():
        rows.append(
            {
                "Species": cls,
                "Encounters": s["encounters"],
                "Total individuals": s["total"],
                "Max individuals per encounter": s["max_per_image"],
                "ROC AUC": auc.get(cls),
            }
        )

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="evaluation.xlsx")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    data_root = data_yaml.parent

    cfg = load_yaml(data_yaml)
    classes = resolve_names(cfg["names"])
    images = collect_images(cfg["val"], data_root)

    stats = compute_table(classes, images)
    auc = compute_roc_auc(args.weights, classes, images)

    print_table(stats, auc)
    save_excel(stats, auc, args.output)

    print(f"\nSaved Excel table to: {Path(args.output).resolve()}")
