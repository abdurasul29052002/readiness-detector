import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_YAML = ROOT / "dataset.yaml"
RUN_NAME = "yolo8s"
RUNS_DIR = ROOT / "runs"
run_dir = RUNS_DIR / RUN_NAME
best = run_dir / "weights" / "best.pt"

model = YOLO(str(best))
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
    "stopped_at_epoch": 36,
    "weights_best": str(best),
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
out = run_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print("\nSummary saqlandi:", out)
