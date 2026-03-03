import sys
from pathlib import Path
from trainers.train_medical_kfold import main as run

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--dataset_root", str(Path(__file__).resolve().parents[1] / "data" / "skin_cancer"),
        "--model", "resnet18",
        "--epochs", "30",
        "--batch_size", "64",
        "--lr", "1e-2",
        "--img_size", "224",
        "--reduction", "16",
        "--attn", "hvca",
        "--folds", "5",
        "--seed", "42",
    ]
    run()
