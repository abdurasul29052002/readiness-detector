"""
FastAPI AI server — YOLOv8 model orqali o'quvchi xatti-harakatini aniqlash.

Endpoints:
  POST /predict       — rasm qabul qilib, bounding boxlar qaytaradi
  GET  /health        — server holati
  GET  /models        — mavjud modellar ro'yxati
  POST /models/switch — faol modelni almashtirish
  POST /models/upload — yangi model yuklash
  POST /predict/video — video faylni kadr-kadr tahlil qilish
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent / "models"
METADATA_FILE = MODELS_DIR / "metadata.json"

CLASS_NAMES = {
    0: "hand-raising",
    1: "read",
    2: "write",
    3: "discuss",
    4: "bow-head",
    5: "turn-head",
    6: "standing",
}

# Diqqatli (attentive) va chalg'igan (distracted) guruhlari
ATTENTIVE_CLASSES = {0, 1, 2}
DISTRACTED_CLASSES = {3, 4, 5, 6}

app = FastAPI(title="Student Behavior Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model registry
models_registry: dict[str, dict] = {}
active_model_version: str | None = None
model: YOLO | None = None


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata():
    data = {}
    for ver, info in models_registry.items():
        data[ver] = info["metadata"]
    METADATA_FILE.write_text(json.dumps(data, indent=2, default=str))


@app.on_event("startup")
def load_models():
    global model, active_model_version

    metadata = load_metadata()

    for pt_file in sorted(MODELS_DIR.glob("*.pt")):
        version = pt_file.stem
        meta = metadata.get(version, {})
        models_registry[version] = {
            "path": pt_file,
            "model": None,
            "metadata": {
                "version": version,
                "training_date": meta.get("training_date"),
                "accuracy": meta.get("accuracy"),
                "description": meta.get("description"),
                "filename": pt_file.name,
            },
        }

    # Load default model
    default_version = "best" if "best" in models_registry else next(iter(models_registry), None)
    if default_version:
        entry = models_registry[default_version]
        entry["model"] = YOLO(str(entry["path"]))
        model = entry["model"]
        active_model_version = default_version
        print(f"Model loaded: {default_version} ({entry['path']})")
    else:
        print("WARNING: No models found in models/ directory")


def run_detection(img_array, confidence: float) -> dict:
    """Rasmda detection ishga tushiradi va natijani qaytaradi."""
    if model is None:
        return {"error": "Model not loaded"}

    results = model.predict(img_array, conf=confidence, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, "unknown"),
            "confidence": round(conf, 3),
            "group": "attentive" if cls_id in ATTENTIVE_CLASSES else "distracted",
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        })

    attentive_count = sum(1 for d in detections if d["group"] == "attentive")
    distracted_count = sum(1 for d in detections if d["group"] == "distracted")
    total = attentive_count + distracted_count

    return {
        "detections": detections,
        "summary": {
            "total": total,
            "attentive": attentive_count,
            "distracted": distracted_count,
            "attentive_percent": round(attentive_count / total * 100, 1) if total > 0 else 0,
            "distracted_percent": round(distracted_count / total * 100, 1) if total > 0 else 0,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "active_model_version": active_model_version,
        "available_models": len(models_registry),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.5,
):
    if model is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    return run_detection(img_array, confidence)


# --- Model Versioning ---

@app.get("/models")
def list_models():
    return {
        "active_version": active_model_version,
        "models": [
            {
                "version": ver,
                "filename": info["metadata"]["filename"],
                "loaded": info["model"] is not None,
                "training_date": info["metadata"].get("training_date"),
                "accuracy": info["metadata"].get("accuracy"),
                "description": info["metadata"].get("description"),
            }
            for ver, info in models_registry.items()
        ],
    }


@app.post("/models/switch")
def switch_model(version: str):
    global active_model_version, model

    if version not in models_registry:
        raise HTTPException(404, f"Model version '{version}' not found")

    entry = models_registry[version]
    if entry["model"] is None:
        entry["model"] = YOLO(str(entry["path"]))

    model = entry["model"]
    active_model_version = version
    return {"status": "switched", "active_version": version}


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    version: str = Form(...),
    training_date: str = Form(None),
    accuracy: float = Form(None),
    description: str = Form(None),
):
    dest = MODELS_DIR / f"{version}.pt"
    content = await file.read()
    dest.write_bytes(content)

    models_registry[version] = {
        "path": dest,
        "model": None,
        "metadata": {
            "version": version,
            "training_date": training_date,
            "accuracy": accuracy,
            "description": description,
            "filename": f"{version}.pt",
        },
    }
    save_metadata()

    return {"status": "uploaded", "version": version}


# --- Video Processing ---

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    frame_interval: int = 30,
):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        import cv2
    except ImportError:
        raise HTTPException(500, "opencv-python-headless is required for video processing")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Video faylini ochib bo'lmadi")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps

        frame_results = []
        frame_num = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = run_detection(rgb, confidence)
                frame_results.append({
                    "frame_number": frame_num,
                    "timestamp_seconds": round(frame_num / fps, 2),
                    **result,
                })
                processed += 1

            frame_num += 1

        cap.release()

        # Overall summary
        if frame_results:
            avg_att = sum(r["summary"]["attentive_percent"] for r in frame_results) / len(frame_results)
            avg_dist = sum(r["summary"]["distracted_percent"] for r in frame_results) / len(frame_results)
            avg_total = sum(r["summary"]["total"] for r in frame_results) / len(frame_results)
        else:
            avg_att = avg_dist = avg_total = 0

        return {
            "total_frames": total_frames,
            "processed_frames": processed,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "frame_results": frame_results,
            "overall_summary": {
                "avg_attentive_percent": round(avg_att, 1),
                "avg_distracted_percent": round(avg_dist, 1),
                "avg_total_detected": round(avg_total, 1),
            },
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
