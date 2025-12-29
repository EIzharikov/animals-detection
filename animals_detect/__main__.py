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

    args = parser.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
