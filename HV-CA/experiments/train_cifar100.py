import sys
from pathlib import Path
from trainers.train_cls import main as run

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--dataset_root", str(Path(__file__).resolve().parents[1] / "data" / "cifar100"),
        "--model", "resnet50",
        "--epochs", "100",
        "--batch_size", "128",
        "--lr", "1e-2",
        "--img_size", "224",
        "--reduction", "16",
        "--attn", "hvca",
    ]
    run()
