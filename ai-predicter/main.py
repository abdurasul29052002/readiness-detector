"""
FastAPI AI server — YOLOv8 model orqali o'quvchi xatti-harakatini aniqlash.

Endpoints:
  POST /predict  — rasm qabul qilib, bounding boxlar qaytaradi
  GET  /health   — server holati
"""

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"

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

# Model yuklash
model: YOLO | None = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}, /predict will not work")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.5,
):
    if model is None:
        return {"error": "Model not loaded"}

    # Rasmni o'qish
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    # YOLO inference
    results = model.predict(img_array, conf=confidence, verbose=False)
    result = results[0]

    # Natijalarni formatlash
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

    # Statistika
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
