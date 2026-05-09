import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "dataset.yaml"
RUN_NAME = "yolo8s"
RUNS_DIR = ROOT / "runs"


def log_environment(out_dir: Path) -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(info, indent=2))
    print("=" * 60)
    print("ENVIRONMENT")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return info


def main():
    run_dir = RUNS_DIR / RUN_NAME
    log_environment(run_dir)

    model = YOLO("yolov8s.pt")

    train_results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        workers=4,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        seed=42,
        deterministic=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,
        save_json=True,
        val=True,
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION (per-class metrics)")
    print("=" * 60)
    val_results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(RUNS_DIR),
        name=f"{RUN_NAME}_val",
        exist_ok=True,
    )

    summary = {
        "run_name": RUN_NAME,
        "weights_best": str(run_dir / "weights" / "best.pt"),
        "weights_last": str(run_dir / "weights" / "last.pt"),
        "metrics_overall": {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "fitness": float(val_results.fitness),
        },
        "metrics_per_class": {
            str(i): {
                "name": val_results.names[i],
                "mAP50": float(val_results.box.ap50[idx]),
                "mAP50-95": float(val_results.box.ap[idx]),
                "precision": float(val_results.box.p[idx]),
                "recall": float(val_results.box.r[idx]),
            }
            for idx, i in enumerate(val_results.box.ap_class_index)
        },
        "speed_ms_per_image": dict(val_results.speed),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary saqlandi:", run_dir / "summary.json")
    print("Best weights:", summary["weights_best"])


if __name__ == "__main__":
    main()
