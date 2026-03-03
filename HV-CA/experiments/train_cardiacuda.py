import sys
from pathlib import Path
from trainers.train_seg import main as run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data" / "cardiacuda"
    sys.argv = [
        sys.argv[0],
        "--images", str(root / "images"),
        "--masks", str(root / "masks"),
        "--epochs", "200",
        "--batch_size", "1",
        "--lr", "1e-2",
        "--img_size", "256",
        "--num_classes", "4",
        "--reduction", "16",
        "--model", "swinunet",
    ]
    run()
