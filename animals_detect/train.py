from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from animals_detect.constants import DATA_YAML_RELATIVE, PROJECT_NAME, RESULTS_PATH, get_device


def train(args):
    model = YOLO(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RESULTS_PATH}/{Path(args.model).name}_epochs{args.epochs}_imgsize{args.imgsize}_batch{args.batch}_freeze{args.freeze}_{timestamp}"

    model.train(
        data=DATA_YAML_RELATIVE,  # путь к data.yml для животных
        epochs=args.epochs,
        imgsz=args.imgsize,
        batch=args.batch,
        device=get_device(),
        project=PROJECT_NAME,
        name=run_name,
        plots=True,
        seed=42,
        freeze=args.freeze,
    )
    print(f"Training completed. Results saved in project '{run_name}'")
