import yaml
from pathlib import Path
from collections import Counter
from animals_detect.constants import PROJECT_ROOT

def load_dataset(data_yml):
    with open(data_yml) as f:
        data_cfg = yaml.safe_load(f)
    classes = data_cfg["names"]
    img_files = data_cfg.get("train", []) + data_cfg.get("val", [])
    img_files = [PROJECT_ROOT / Path(p) for p in img_files]
    return classes, img_files

def compute_stats(classes, img_files):
    stats = {cls: {"encounters": 0, "total": 0, "max_per_image": 0} for cls in classes}
    for img_path in img_files:
        ann_file = PROJECT_ROOT / "data/labels" / img_path.parent.name / (img_path.stem + ".txt")
        img_counter = Counter()
        print(ann_file)
        break
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

def print_stats_table(stats):
    print(f"{'Class':<20} {'Encounters':<10} {'Total':<10} {'Max per image':<15}")
    for cls_name, s in stats.items():
        print(f"{cls_name:<20} {s['encounters']:<10} {s['total']:<10} {s['max_per_image']:<15}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    classes, img_files = load_dataset(args.data)
    stats = compute_stats(classes, img_files)
    print_stats_table(stats)
