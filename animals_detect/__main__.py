import argparse
from pathlib import Path

from animals_detect.train import train


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ====== TRAIN ======
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--model", help="YOLO model name")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--imgsize", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument(
        "--freeze", type=int, default=0, help="Amount of freezed layers"
    )

    # ====== INFER ======
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--weights", required=True, help="Path to trained weights"
    )
    infer_parser.add_argument(
        "--source", required=True, help="Image or folder to infer on"
    )

    # ====== EVALUATE ======
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluate")
    eval_parser.add_argument(
        "--weights-dir", type=Path, required=True, help="Path to trained weights"
    )
    eval_parser.add_argument(
        "--output", type=Path, required=True, help="Path to save table"
    )
    eval_parser.add_argument("--imgsizez", type=int, default=640, help="Image size")

    args = parser.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
