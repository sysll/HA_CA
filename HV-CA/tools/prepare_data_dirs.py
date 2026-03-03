from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def main() -> int:
    (DATA / "skin_cancer").mkdir(parents=True, exist_ok=True)
    (DATA / "sakha_tb").mkdir(parents=True, exist_ok=True)
    (DATA / "cifar100").mkdir(parents=True, exist_ok=True)
    (DATA / "imagenet100").mkdir(parents=True, exist_ok=True)
    (DATA / "cardiacuda" / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "cardiacuda" / "masks").mkdir(parents=True, exist_ok=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
