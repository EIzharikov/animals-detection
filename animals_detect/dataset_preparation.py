import yaml
from pathlib import Path
from collections import Counter
import pandas as pd
from animals_detect.constants import PROJECT_ROOT

DATA_YML = PROJECT_ROOT / "data/data.yaml"
OUTPUT_XLSX = PROJECT_ROOT / "results/class_stats.xlsx"

def load_classes(data_yml):
    with open(data_yml) as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg["names"]

def gather_all_images(data_yml):
    with open(data_yml) as f:
        data_cfg = yaml.safe_load(f)
    images = []
    for split in ["train", "val", "test"]:
        if split in data_cfg:
            for line in Path(data_cfg[split]).read_text().splitlines():
                images.append(line.strip())
    return images

def compute_class_stats(img_list, classes):
    counter = Counter()
    max_per_img = {cls_name: 0 for cls_name in classes.values()}
    encounters = {cls_name: 0 for cls_name in classes.values()}
    for img in img_list:
        img_path = Path(img)
        ann_file = PROJECT_ROOT / Path("data/labels") / img_path.parent.name / (img_path.stem + ".txt")
        img_counter = Counter()
        if ann_file.exists():
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        counter[cls_id] += 1
                        img_counter[cls_id] += 1
        for cls_id, cls_name in classes.items():
            if img_counter.get(cls_id, 0) > 0:
                encounters[cls_name] += 1
                max_per_img[cls_name] = max(max_per_img[cls_name], img_counter[cls_id])
    rows = []
    for cls_id, cls_name in classes.items():
        rows.append([cls_name, encounters[cls_name], counter.get(cls_id,0), max_per_img[cls_name]])
    return rows

if __name__ == "__main__":
    classes = load_classes(DATA_YML)
    all_images = gather_all_images(DATA_YML)
    stats = compute_class_stats(all_images, classes)

    df = pd.DataFrame(stats, columns=["Class", "Encounters", "Total", "Max per image"])
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Class statistics saved to {OUTPUT_XLSX}")
